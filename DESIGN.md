# Flow-GRPO on 2D 8-Gaussians — 설계

근거: ARIS `docs/tutorials/modern_diffusion_post_training_tutorial_en.md` (이하 §번호), `code/flow_matching.py`, `code/diffusion_online_rl.py`
형식: ARIS toy처럼 스크립트 1개, 각 실험은 assert로 확인. CPU 수 분.

## 설정 (축소)

| | 논문 (SD3.5-M) | 축소판 |
|---|---|---|
| 데이터 | 이미지 latent, prompt | 2D 위 봉우리 8개 혼합 가우시안 (반지름 2, std 0.1), 무조건부 |
| 모델 | MMDiT | TimeMLP (`flow_matching.py`, 시간 규약은 §2.1로 뒤집음) |
| 보상 | GenEval / OCR / PickScore | 오른쪽 봉우리 2개 중 가까운 쪽까지 거리의 음수 |

알고리즘은 §3 그대로: §3.2 SDE 스텝($h>0$, $g=a\sqrt{t/(1-t)}$, $t=1$ 첫 스텝 처리), §3.3 차원 합 log-prob·ratio 분모는 old, §3.4 PPO-clip + 닫힌형 KL, §3.6 고정/재계산 경계, §3.5 학습 SDE $T=10$ / 평가 ODE $T=40$.
$G=24$, $a=0.7$, $\varepsilon_c=0.2$, $\beta_\text{KL}=0.04$ (§3.7).

축소로 잃는 것: 고차원, prompt별 그룹, 보상 모델 해킹 여지 → 이 toy에서 KL은 다양성 유지 역할로만 확인된다.

## 1단계. 이해와 검증

**1-① 작게 구현한 게 논문대로인가** — 학습 없이, 구현이 §3 식과 일치하는지 확인

- base가 8봉우리를 재현 (봉우리별 질량 ≈ 1/8)
- $\theta=\text{old}$에서 ratio = 1, $\theta=\text{ref}$에서 KL = 0
- `gauss_logp`가 `torch.distributions.Normal` log-prob 합과 일치 (§3.3)
- §3.4 닫힌형 KL이 두 가우시안 평균으로 직접 계산한 KL과 일치
- base로 SDE($T=10$) 샘플 vs ODE 샘플이 비슷 — 엄밀한 일치는 기대하지 않음 (Q24), 차이를 기록

**1-② 메커니즘이 작동하는지, 비용 순서대로**

1. 위 1-① (학습 없음, 초)
2. 업데이트 1회: $A>0$ 샘플에서 $\mu_\theta$가 실제 샘플된 $x_{k+1}$ 쪽으로 이동 (§3.6)
3. 전체 학습 (분)

**1-③ 재현 판단: baseline과 지표**

- baseline: base 모델 (같은 평가 노이즈, ODE 40스텝)
- 지표 (§8: raw reward + 샘플 + 다양성): ODE raw reward, 봉우리별 질량, 3σ 밖 비율
- 재현할 논문 주장:
  - §3.7 SDE로 학습해도 ODE 평가 보상이 오른다 → reward > base
  - §3.7 KL이 품질을 지킨다 → $\beta=0$ 대비 3σ 밖 비율·봉우리 내부 std 수축이 덜함
  - §3.5 $T_\text{train}=10$이 $T_\text{train}=40$과 비슷한 보상을 더 싸게 → 둘 다 학습, 보상과 NFE 비교

## 2단계. 적용 조건

**2-① 언제 이식 가능한가 — 전제, 실패, 신호** (§8, A.3, Q24)

| 전제 | 깨지는 조건 | 신호 | 출처 |
|---|---|---|---|
| 그룹 안 보상에 분산이 있다 | 0/1 보상, 목표를 봉우리 하나의 반경 σ로 좁힘 (base 적중률 ≈ 5%), $G\in\{4,24\}$, 총 샘플 수 고정 | std=0 그룹 비율 ≈ $p^G+(1-p)^G$, 보상 정체 | A.3 |
| 학습 SDE와 평가 ODE의 분포가 같다 | 유한 스텝·학습 오차 | 같은 policy의 SDE 평가 vs ODE 평가 보상 차이 | Q24, §3.5 |
| clip이 모든 $t$에서 작동한다 | ratio 분포가 $t$마다 다름 | $t$ 구간별 clip 비율 | §8 (GRPO-Guard) |
| 보상 상승 = 품질 상승 | KL 없음 | raw reward ↑ 동시에 다양성 ↓ | §8 |

신호는 1-③ 학습에서 같이 기록한다. 추가 학습은 첫 줄(희소 보상)만.

**2-② 대안 중 언제 고르고 언제 피하나** (§1.2, §6, Q25)

같은 base·보상·총 샘플 수로 학습해 비교한다. 대안은 둘로 나뉜다.

**(a) PPO — advantage 계산법만 다른 대안** (§1.1: GRPO = PPO에서 critic을 그룹 평균으로 대체)

SDE rollout, ratio, clip, KL은 Flow-GRPO와 동일하고, advantage만 critic $V(x_t,t)$로 계산한다.

| 기준 | 측정 |
|---|---|
| 추가 모델(critic) 비용 vs 그룹 $G$개 샘플 비용 | 같은 보상 도달까지 샘플 수·시간 |
| 희소 보상에서 학습 신호 유지 | 2-① 희소 보상 설정에서 보상 곡선 |
| 결과 | 최종 보상, 봉우리별 질량 |

주의: 2D에서는 critic 학습이 쉬워 PPO가 이미지보다 유리하게 나올 수 있다 (축소로 잃는 것).

**(b) DGPO, DiffusionNFT — 튜토리얼에 선택 기준이 있는 대안** (§6, Q25, 코드 §7.2·§7.3)

| 선택 기준 (Q25) | 측정 |
|---|---|
| 샘플러: Flow-GRPO만 SDE rollout 필요 | 같은 보상 도달까지 NFE·시간 |
| 저장: 궤적 $O(PGTd)$ vs 깨끗한 샘플 $O(PGd)$ | rollout 버퍼 크기 |
| 결과 | 최종 보상, 봉우리별 질량 |

판단 기준(튜토리얼 정답): SDE 비용을 낼 수 있고 스텝별 likelihood가 필요하면 Flow-GRPO, ODE만 쓰거나 저장·속도가 중요하면 DGPO/NFT. toy 결과가 이 기준과 맞는지 확인하고, 논문 효율 수치(~20×, ~25×)는 특정 보상 곡선 비교이므로(Q22, Q25) 직접 잰다.

**2-③ (선택) 스케일링** — 생략.
