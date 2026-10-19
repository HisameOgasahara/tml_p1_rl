# Flow-GRPO on 2D 8-Gaussians

관점1(이해·검증 → 적용 조건) 기준으로 Flow-GRPO를 2D 봉우리 8개 혼합 가우시안에서 재현하는 toy. 설계는 [DESIGN.md](DESIGN.md).

| 파일 | 내용 |
|---|---|
| `fg8.py` | 데이터·보상·base(RF)·Flow-GRPO·PPO·DGPO·DiffusionNFT·검증 함수 |
| `01_stage1_verify.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tml_p1_rl/blob/main/01_stage1_verify.ipynb) | 1-① 구현 검증, 1-② 업데이트 1회, 1-③ 재현 판단 |
| `02_stage2_conditions.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tml_p1_rl/blob/main/02_stage2_conditions.ipynb) | 2-① 전제·실패 신호, 2-② PPO / DGPO / NFT 비교 |

Colab에서 노트북을 열고 **CPU 런타임**으로 위에서부터 실행. 첫 셀이 레포를 clone합니다.
식과 기본 HP는 ARIS-in-AI-Offer `modern_diffusion_post_training_tutorial` §3–§7을 따릅니다.
