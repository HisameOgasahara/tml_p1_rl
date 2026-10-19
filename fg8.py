"""
Flow-GRPO on 2D 8-Gaussians (+ PPO / DGPO / DiffusionNFT for comparison)
=========================================================================

Pairs with DESIGN.md. Formulas follow ARIS-in-AI-Offer
docs/tutorials/modern_diffusion_post_training_tutorial_en.md (cited as §x.y):

    x_t = (1-t) x_0 + t eps,  t=0 data, t=1 noise, generation runs t: 1 -> 0
    target v = eps - x_0                                          (§2.1)
    SDE step (h > 0):  mu = x - h [v + g^2/(2t) (x + (1-t) v)]    (§3.2)
                       x' = mu + g sqrt(h) xi,  g^2 = a^2 t/(1-t)
    log p(x'|x) = N(mu, g^2 h I), summed over dims, ratio vs OLD  (§3.3)
    loss = PPO-clip + beta_KL * ||mu_theta - mu_ref||^2/(2 g^2 h) (§3.4)

The base model is ARIS code/flow_matching.py's TimeMLP with the time
convention flipped to the one above. Everything runs on CPU: the network
is a 2D MLP and batches are a few hundred points, so GPU launch/transfer
overhead would dominate (ARIS's toys are CPU too).
"""
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- data & rewards ------------------------------------------------------------

R_RING, SIGMA = 2.0, 0.1
MODES = torch.tensor([[R_RING * math.cos(2 * math.pi * k / 8),
                       R_RING * math.sin(2 * math.pi * k / 8)] for k in range(8)])
TARGET = (0, 1)                                   # the two right-hand modes


def sample_data(n):
    k = torch.randint(0, 8, (n,))
    return MODES[k] + SIGMA * torch.randn(n, 2)


def reward_dense(x):
    """minus the distance to the nearer of the two target modes"""
    return -torch.cdist(x, MODES[list(TARGET)]).min(1).values


def reward_sparse(x, radius=SIGMA):
    """1 inside a small disc around mode 0, else 0 (rare under the base model)"""
    return ((x - MODES[0]).norm(dim=1) < radius).float()


REWARDS = {"dense": reward_dense, "sparse": reward_sparse}


def metrics(x, reward_fn):
    d, k = torch.cdist(x, MODES).min(1)
    masses = torch.bincount(k, minlength=8).float() / len(x)
    on = d < 3 * SIGMA
    within = ((x - MODES[k])[on] ** 2).mean().sqrt().item() if on.any() else float("nan")
    return dict(reward=reward_fn(x).mean().item(),
                target_mass=masses[list(TARGET)].sum().item(),
                masses=masses.tolist(),
                off_support=(~on).float().mean().item(),
                within_std=within)


# --- model -----------------------------------------------------------------------

class TimeMLP(nn.Module):
    """(x, t) -> v (or a scalar value for the PPO critic). Same as ARIS flow_matching.py."""

    def __init__(self, out=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out))

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, t.reshape(-1, 1)], -1))


def train_base(steps=8000, bs=1024, lr=1e-3, seed=0, log_every=1000):
    """Rectified-flow pretraining: regress v = eps - x0."""
    torch.manual_seed(seed)
    model = TimeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x0, eps, t = sample_data(bs), torch.randn(bs, 2), torch.rand(bs)
        xt = (1 - t)[:, None] * x0 + t[:, None] * eps
        loss = ((model(xt, t) - (eps - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log_every and step % log_every == 0:
            print(f"  base step {step:5d}  loss {loss.item():.4f}")
    return model.requires_grad_(False)


def get_base(path="base.pt", **kw):
    model = TimeMLP()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.requires_grad_(False)
    model = train_base(**kw)
    torch.save(model.state_dict(), path)
    return model


# --- samplers (§3.2, §3.3) -------------------------------------------------------

def tgrid(T):
    return torch.linspace(1, 0, T + 1)


def g2_at(t, a, t_cap):
    """g(t)^2 = a^2 t/(1-t). At t=1 it is infinite; like the official code, the first
    step uses the neighbouring grid point t_cap = t_1 (§3.2, A.3)."""
    t = min(t, t_cap)
    return a * a * t / (1 - t)


def sde_mean(model, x, t, h, g2):
    v = model(x, t)
    mu = x - h * (v + g2 / (2 * t) * (x + (1 - t) * v))
    return mu, v


def gauss_logp(x_next, mu, var):
    d = x_next.shape[-1]
    return -((x_next - mu) ** 2).sum(-1) / (2 * var) - 0.5 * d * math.log(2 * math.pi * var)


def step_consts(ts, k, a):
    t = ts[k].item()
    h = (ts[k] - ts[k + 1]).item()
    return t, h, g2_at(t, a, ts[1].item())


@torch.no_grad()
def ode_sample(model, z, T):
    ts, x = tgrid(T), z
    for k in range(T):
        t, h = ts[k].item(), (ts[k] - ts[k + 1]).item()
        x = x - h * model(x, t)
    return x


@torch.no_grad()
def rollout_sde(model, z, T, a, generator=None):
    """Returns states xs [T+1, B, 2] and per-step log-probs [T, B]."""
    ts, x = tgrid(T), z
    xs, lps = [z], []
    for k in range(T):
        t, h, g2 = step_consts(ts, k, a)
        mu, _ = sde_mean(model, x, t, h, g2)
        x = mu + math.sqrt(g2 * h) * torch.randn(x.shape, generator=generator)
        xs.append(x); lps.append(gauss_logp(x, mu, g2 * h))
    return torch.stack(xs), torch.stack(lps)


# --- advantages --------------------------------------------------------------------

def group_adv(R, G):
    """GRPO advantage (§1.1); a group with zero std gets A = 0 (A.3)."""
    Rg = R.view(-1, G)
    std = Rg.std(1, unbiased=False, keepdim=True)
    A = torch.where(std > 1e-8, (Rg - Rg.mean(1, keepdim=True)) / (std + 1e-8), torch.zeros_like(Rg))
    return A.reshape(-1), (std.squeeze(1) <= 1e-8).float().mean().item()


# --- training --------------------------------------------------------------------------

@dataclass
class Cfg:
    algo: str = "flow_grpo"      # flow_grpo | ppo | dgpo | nft
    reward: str = "dense"        # dense | sparse
    iters: int = 300
    batch: int = 384             # trajectories (= reward calls) per iteration, same for every algo
    G: int = 24
    T_train: int = 10
    T_eval: int = 40
    a: float = 0.7
    clip: float = 0.2
    beta_kl: float = 0.04
    lr: float = 3e-4
    n_minibatch: int = 4
    eval_every: int = 20
    n_eval: int = 4000
    seed: int = 0
    critic_lr: float = 1e-3      # ppo
    dgpo_beta_T: float = 1.0     # dgpo: beta*T folded together (§4.4; scale is convention-dependent)
    dgpo_t_min: float = 0.1      # dgpo timestep clipping (§4.7)
    n_noise: int = 8             # dgpo / nft: (t, eps) draws per sample per iteration
    nft_beta: float = 1.0        # nft mixing beta (§5.5)
    nft_ema: float = 0.5         # nft: old <- ema*old + (1-ema)*theta after each iteration (§5.8)


def _pg_update(policy, ref, critic, opt, copt, xs, logp_old, A, R, cfg, log):
    """Flow-GRPO / PPO update on stored SDE trajectories (§3.4, §3.6)."""
    T, B = logp_old.shape
    ts = tgrid(T)
    clip_hits, kls, nfe = torch.zeros(T), [], 0
    for idx in torch.randperm(B).chunk(cfg.n_minibatch):
        loss, closs = 0.0, 0.0
        for k in range(T):
            t, h, g2 = step_consts(ts, k, cfg.a)
            x_k, x_k1 = xs[k, idx], xs[k + 1, idx]
            mu, _ = sde_mean(policy, x_k, t, h, g2)
            rho = torch.exp(gauss_logp(x_k1, mu, g2 * h) - logp_old[k, idx])
            adv = A[k, idx]
            pg = -torch.min(rho * adv, rho.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            clip_hits[k] += ((rho - 1).abs() > cfg.clip).float().sum()
            nfe += len(idx)
            if cfg.beta_kl > 0:
                with torch.no_grad():
                    mu_ref, _ = sde_mean(ref, x_k, t, h, g2)
                kl = ((mu - mu_ref) ** 2).sum(-1).mean() / (2 * g2 * h)
                kls.append(kl.item()); nfe += len(idx)
                pg = pg + cfg.beta_kl * kl
            loss = loss + pg / T
            if critic is not None:
                closs = closs + ((critic(x_k, t).squeeze(-1) - R[idx]) ** 2).mean() / T
                nfe += len(idx)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if critic is not None:
            copt.zero_grad(); closs.backward(); copt.step()
    log["clip_by_t"].append((clip_hits / B).tolist())
    log["kl"].append(sum(kls) / len(kls) if kls else 0.0)
    return nfe


def _dgpo_update(policy, ref, opt, x0, A, cfg):
    """DGPO (§4.3, §4.4, §7.2): group DSM differences inside one sigmoid."""
    G = cfg.G
    x0g, Ag = x0.view(-1, G, 2), A.view(-1, G)
    nfe = 0
    for gidx in torch.randperm(x0g.shape[0]).chunk(cfg.n_minibatch):
        xg, ag, ng = x0g[gidx], Ag[gidx], len(gidx)
        loss = 0.0
        for _ in range(cfg.n_noise):
            t = cfg.dgpo_t_min + (1 - cfg.dgpo_t_min) * torch.rand(ng, 1, 1)
            eps = torch.randn(ng, 1, 2).expand(ng, G, 2)          # shared t, eps within a group
            xt, v = (1 - t) * xg + t * eps, eps - xg
            tt = t.expand(ng, G, 1).reshape(-1)
            l_th = ((policy(xt.reshape(-1, 2), tt) - v.reshape(-1, 2)) ** 2).mean(-1).view(ng, G)
            with torch.no_grad():
                l_ref = ((ref(xt.reshape(-1, 2), tt) - v.reshape(-1, 2)) ** 2).mean(-1).view(ng, G)
            lam = 2 * t.view(ng, 1) ** 2                          # D t^2 with D = 2 (§7.2 scale note)
            logit = -(lam * cfg.dgpo_beta_T * ag * (l_th - l_ref)).sum(1)
            loss = loss - F.logsigmoid(logit).mean() / cfg.n_noise
            nfe += 2 * ng * G
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    return nfe


def _nft_update(policy, old, opt, x0, A, cfg):
    """DiffusionNFT (§5.3, §5.6, §7.3): reflected two-branch regression around old."""
    r = 0.5 + 0.5 * A.clamp(-1, 1)
    beta, nfe = cfg.nft_beta, 0
    for idx in torch.randperm(len(x0)).chunk(cfg.n_minibatch):
        xb, rb = x0[idx].repeat(cfg.n_noise, 1), r[idx].repeat(cfg.n_noise)
        t = torch.rand(len(xb)).clamp(0.01, 0.99)
        eps = torch.randn_like(xb)
        xt, v = (1 - t)[:, None] * xb + t[:, None] * eps, eps - xb
        with torch.no_grad():
            v_old = old(xt, t)
        d = policy(xt, t) - v_old
        v_pos, v_neg = v_old + beta * d, v_old - beta * d
        loss = (rb * ((v_pos - v) ** 2).mean(-1) + (1 - rb) * ((v_neg - v) ** 2).mean(-1)).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        nfe += 2 * len(xb)
    return nfe


def train(base, cfg, verbose=True):
    torch.manual_seed(cfg.seed)
    policy = copy.deepcopy(base).requires_grad_(True)
    ref = copy.deepcopy(base).requires_grad_(False)
    old = copy.deepcopy(base).requires_grad_(False) if cfg.algo == "nft" else None
    critic = TimeMLP(out=1) if cfg.algo == "ppo" else None
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    copt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr) if critic is not None else None
    reward_fn = REWARDS[cfg.reward]
    eval_z = torch.randn(cfg.n_eval, 2, generator=torch.Generator().manual_seed(1234))
    B, T = cfg.batch, cfg.T_train
    pg_algo = cfg.algo in ("flow_grpo", "ppo")
    log = dict(cfg=asdict(cfg), it=[], nfe=[], time=[], ode=[], sde_reward=[],
               zero_std=[], clip_by_t=[], kl=[],
               buffer_floats=((T + 1) * B * 2 + T * B) if pg_algo else (B * 2 + B))
    nfe, t_train = 0, 0.0
    for it in range(cfg.iters + 1):
        if it % cfg.eval_every == 0 or it == cfg.iters:
            with torch.no_grad():
                m = metrics(ode_sample(policy, eval_z, cfg.T_eval), reward_fn)
                xs_e, _ = rollout_sde(policy, eval_z, T, cfg.a, torch.Generator().manual_seed(99))
            log["it"].append(it); log["nfe"].append(nfe); log["time"].append(t_train)
            log["ode"].append(m); log["sde_reward"].append(reward_fn(xs_e[-1]).mean().item())
            if verbose:
                print(f"  [{cfg.algo}] it {it:4d}  ode reward {m['reward']:+.4f}  target mass {m['target_mass']:.3f}"
                      f"  off-support {m['off_support']:.3f}  within-std {m['within_std']:.3f}")
        if it == cfg.iters:
            break
        t0 = time.perf_counter()
        z = torch.randn(B, 2)
        if pg_algo:
            xs, logp_old = rollout_sde(policy, z, T, cfg.a)        # policy == old at rollout time
            nfe += B * T
            R = reward_fn(xs[-1])
            A_grp, zf = group_adv(R, cfg.G)
            if cfg.algo == "flow_grpo":
                A = A_grp[None].expand(T, B)                       # one advantage for every step
            else:
                with torch.no_grad():
                    V = critic(xs[:-1].reshape(-1, 2), tgrid(T)[:-1].repeat_interleave(B)).view(T, B)
                A = R[None] - V
                A = (A - A.mean()) / (A.std() + 1e-8)
                nfe += B * T
            nfe += _pg_update(policy, ref, critic, opt, copt, xs, logp_old, A, R, cfg, log)
        else:
            x0 = ode_sample(old if old is not None else policy, z, T)   # any sampler works (§4.6, §5.2)
            nfe += B * T
            R = reward_fn(x0)
            A_grp, zf = group_adv(R, cfg.G)
            if cfg.algo == "dgpo":
                nfe += _dgpo_update(policy, ref, opt, x0, A_grp, cfg)
            else:
                nfe += _nft_update(policy, old, opt, x0, A_grp, cfg)
                with torch.no_grad():
                    for po, pn in zip(old.parameters(), policy.parameters()):
                        po.mul_(cfg.nft_ema).add_(pn, alpha=1 - cfg.nft_ema)
        log["zero_std"].append(zf)
        t_train += time.perf_counter() - t0
    log["final_samples"] = ode_sample(policy, eval_z[:1000], cfg.T_eval).tolist()
    log["policy"] = policy
    return log


def run(base, cfg, name, results_dir="results", verbose=False):
    """train() with a json cache keyed by name and seed (the policy is not cached)."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}_s{cfg.seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    log = train(base, cfg, verbose=verbose)
    log.pop("policy")
    with open(path, "w") as f:
        json.dump(log, f)
    return log


# --- 1-① implementation checks (no training) ---------------------------------------

def check_implementation(base, a=0.7, T=10, n=4000):
    torch.manual_seed(0)
    out = {}
    # (a) base reproduces the 8 modes
    z = torch.randn(n, 2)
    m_ode = metrics(ode_sample(base, z, 40), reward_dense)
    assert all(0.08 < p < 0.17 for p in m_ode["masses"]), m_ode["masses"]
    assert m_ode["off_support"] < 0.1, m_ode["off_support"]
    out["base_ode"] = m_ode

    # (b) ratio = 1 when theta == old, recomputed through the training code path
    xs, lp_old = rollout_sde(base, z[:512], T, a)
    ts = tgrid(T)
    max_dev = 0.0
    for k in range(T):
        t, h, g2 = step_consts(ts, k, a)
        mu, _ = sde_mean(base, xs[k], t, h, g2)
        rho = torch.exp(gauss_logp(xs[k + 1], mu, g2 * h) - lp_old[k])
        max_dev = max(max_dev, (rho - 1).abs().max().item())
    assert max_dev < 1e-4, max_dev
    out["ratio_max_dev"] = max_dev

    # (c) gauss_logp == torch Normal log-prob summed over dims (§3.3)
    mu, x, var = torch.randn(64, 2), torch.randn(64, 2), 0.37
    ref_lp = torch.distributions.Normal(mu, math.sqrt(var)).log_prob(x).sum(-1)
    assert torch.allclose(gauss_logp(x, mu, var), ref_lp, atol=1e-5)

    # (d) KL: zero at theta == ref; closed form (§3.4) == torch kl_divergence for a perturbed policy.
    # float64: mu_p - mu_r is a small difference of O(1) numbers, so float32 cancellation
    # alone gives ~1e-2 relative error on samples where the KL is tiny.
    ref64 = copy.deepcopy(base).double()
    pert = copy.deepcopy(ref64)
    with torch.no_grad():
        for p in pert.parameters():
            p.add_(0.01 * torch.randn_like(p))
    xs64 = xs.double()
    worst = 0.0
    for k in range(T):
        t, h, g2 = step_consts(ts, k, a)
        g = math.sqrt(g2)
        mu_r, v_r = sde_mean(ref64, xs64[k], t, h, g2)
        mu_0, _ = sde_mean(copy.deepcopy(ref64), xs64[k], t, h, g2)
        assert ((mu_0 - mu_r) ** 2).sum(-1).max().item() == 0.0
        mu_p, v_p = sde_mean(pert, xs64[k], t, h, g2)
        kl_mean = ((mu_p - mu_r) ** 2).sum(-1) / (2 * g2 * h)
        kl_v = (h / 2) * (1 / g + g * (1 - t) / (2 * t)) ** 2 * ((v_p - v_r) ** 2).sum(-1)
        std = math.sqrt(g2 * h)
        kl_torch = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_p, std), torch.distributions.Normal(mu_r, std)).sum(-1)
        worst = max(worst, ((kl_mean - kl_torch).abs() / (kl_torch + 1e-12)).max().item(),
                    ((kl_v - kl_torch).abs() / (kl_torch + 1e-12)).max().item())
    assert worst < 1e-3, worst
    out["kl_rel_err"] = worst

    # (e) SDE(T) vs ODE samples of the base: recorded, not asserted (Q24)
    xs_full, _ = rollout_sde(base, z, T, a)
    out["base_sde"] = metrics(xs_full[-1], reward_dense)
    return out


# --- 1-② one update moves the transition mean the right way (§3.6) -------------------

def check_one_update(base, a=0.7, T=10, lr=1e-4, clip=0.2):
    torch.manual_seed(0)
    z = torch.randn(1, 2)
    xs, lp_old = rollout_sde(base, z, T, a)
    k = T // 2
    t, h, g2 = step_consts(tgrid(T), k, a)
    out = {}
    for adv in (+1.0, -1.0):
        m = copy.deepcopy(base).requires_grad_(True)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        with torch.no_grad():
            d0 = (sde_mean(m, xs[k], t, h, g2)[0] - xs[k + 1]).norm().item()
        mu, _ = sde_mean(m, xs[k], t, h, g2)
        rho = torch.exp(gauss_logp(xs[k + 1], mu, g2 * h) - lp_old[k])
        loss = -torch.min(rho * adv, rho.clamp(1 - clip, 1 + clip) * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d1 = (sde_mean(m, xs[k], t, h, g2)[0] - xs[k + 1]).norm().item()
        out[adv] = (d0, d1)
    assert out[1.0][1] < out[1.0][0], out
    assert out[-1.0][1] > out[-1.0][0], out
    return out


# --- 2-① zero-std groups vs theory p^G + (1-p)^G ---------------------------------------

@torch.no_grad()
def check_zero_std(base, Gs=(4, 24), a=0.7, T=10, n=96_000):
    torch.manual_seed(0)
    xs, _ = rollout_sde(base, torch.randn(n, 2), T, a)
    hit = reward_sparse(xs[-1])
    p = hit.mean().item()
    out = {"p": p}
    for G in Gs:
        _, zf = group_adv(hit, G)
        theory = p ** G + (1 - p) ** G
        n_groups = n // G
        se = math.sqrt(theory * (1 - theory) / n_groups) + 1e-9
        assert abs(zf - theory) < 4 * se + 0.01, (G, zf, theory)
        out[G] = dict(measured=zf, theory=theory)
    return out
