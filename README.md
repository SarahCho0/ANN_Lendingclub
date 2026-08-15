<div align="center">

<br/>

# 💰 LendingClub Sharpe Optimizer

**ANN default prediction · Sharpe-ratio optimal investment threshold search**

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat-square&logo=polars&logoColor=white)](https://pola.rs)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)

<br/>

[![📐 Algorithm](https://img.shields.io/badge/📐_Core-Algorithm-2E7D32?style=for-the-badge)](#core-algorithm)
[![⚙️ Model](https://img.shields.io/badge/⚙️_Model-Config-1565C0?style=for-the-badge)](#model-configuration)

<br/>

</div>

---

## Overview

**LendingClub Sharpe Optimizer** trains a **scikit-learn `MLPClassifier` (ANN)** on Lending Club loan data to predict default probability, then searches the investment threshold **θ\*** that maximizes portfolio **Sharpe ratio** — computed from IRR-based cash flows — and validates it with a **1,000-sample bootstrap** for a 95% confidence interval, benchmarked against an approve-all-loans strategy.

| | Train (60%) | Validation (20%) | Test (20%) |
|---|---|---|---|
| **Purpose** | Fit the ANN | Search θ\* — max Sharpe over a 990-point grid (1%–99%, 0.1% steps) | Final Sharpe + 95% CI (1,000-sample bootstrap) |
| **Output** | `MLPClassifier` | `optimal_threshold.png` | Benchmark comparison |

---

## Quick Start

**1 · Environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2 · Preprocess**

```bash
cd src
python preprocess_pipeline.py
```

**3 · Train**

```bash
python train_models.py
```

> Outputs: `models/ANN_Validation.pkl`, `models/ANN_Test.pkl`, `models/optimal_threshold.png`, `experiments_log.csv`

---

## Project Structure

```
ANN_Lendingclub/
├── src/
│   ├── config.py                 # Central config — paths, hyperparameters, column lists
│   ├── utils.py                  # Logger, Sharpe/IRR/bootstrap helpers
│   ├── preprocess_pipeline.py    # Data preprocessing
│   └── train_models.py           # Train → validate (θ*) → test → benchmark
├── notebooks/
│   ├── 01_label_definition.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_modeling.ipynb
│   └── 03_modeling_result.ipynb
├── models/
│   ├── ANN_Validation.pkl
│   ├── ANN_Test.pkl
│   └── optimal_threshold.png
├── 260129_feature 변수후보.xlsx   # Feature candidate list
└── requirements.txt
```

---

## Core Algorithm

**Sharpe ratio** (IRR-based)

$$\text{Sharpe} = \frac{E[R_p] - E[R_f]}{\sigma[R_p]}$$

**Investment decision logic**

```python
invest = pred_prob < θ           # only invest below the default-probability threshold

portfolio_return = (
    IRR        if invest          # actual internal rate of return
    else rf_return                # risk-free (treasury) return otherwise
)

Sharpe = (mean(portfolio_return) - mean(rf_return)) / std(portfolio_return)
```

**IRR**, computed from an amortizing-loan cash flow (equal monthly payments, early termination on default), then annualized: `(1 + monthly_irr)^12 - 1`.

**θ\* search** — on the validation set, evaluate Sharpe at 990 threshold values (1%–99%, 0.1% steps) and keep the one that maximizes it. On the test set, apply θ\* and bootstrap 1,000 times to get a 95% confidence interval (2.5th–97.5th percentile of the Sharpe distribution).

---

## Model Configuration

Values as defined in `src/config.py`.

| Parameter | Value |
|---|---|
| `hidden_layer_sizes` | `(256, 128, 64)` |
| `activation` | `relu` |
| `learning_rate_init` | `0.001` |
| `alpha` (L2 regularization) | `0.001` |
| `batch_size` | `512` |
| `max_iter` | `1000` |
| `early_stopping` | `True` |
| `random_state` | `42` |

**Default definition**

```python
# Target = 1 (default)
["Charged Off", "Default", "Late (31-120 days)"]

# Target = 0 (performing)
["Fully Paid"]
```

**Key settings**

| Setting | Value | Description |
|---|---|---|
| `INVESTMENT_THRESHOLD` | `0.15` | Fallback reject threshold before θ\* search |
| `USE_SAMPLE` / `SAMPLE_FRAC` | `False` / `0.3` | Test-mode subsampling |
| `DEFAULT_RF_3YR` / `DEFAULT_RF_5YR` | `0.061` / `0.104` | Fallback treasury yields if the FRED fetch fails |

---

## Evaluation

- **ROC-AUC** — classification performance for default prediction
- **Sharpe ratio** — risk-adjusted portfolio return at θ\*
- **Approval rate** — share of loans invested in (`pred_prob < θ*`)
- **Benchmark comparison** — Sharpe of the ANN + θ\* strategy vs. an approve-all-loans strategy
- Every run is logged to `experiments_log.csv` (`Date, Model, Split, AUC, Sharpe, Avg_Return, Duration, Params, Memo`)

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn (`MLPClassifier`) |
| Data | pandas · Polars · NumPy |
| Financial calc | numpy-financial (IRR) · pandas-datareader (FRED treasury rates) |
| Viz | matplotlib · seaborn |
| Notebooks | Jupyter / JupyterLab |

---

<details>
<summary>🇰🇷 &nbsp;한국어 설명 보기</summary>
<br/>

## 개요

**LendingClub Sharpe Optimizer**는 Lending Club 대출 데이터로 **scikit-learn `MLPClassifier`(ANN)**를 학습해 부도 확률을 예측하고, IRR 기반 현금흐름으로 계산한 포트폴리오 **Sharpe Ratio**를 최대화하는 투자 임계값 **θ\***를 탐색합니다. 1,000회 부트스트래핑으로 95% 신뢰구간을 산출하며, 모든 대출을 승인하는 벤치마크와 성과를 비교합니다.

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src
python preprocess_pipeline.py
python train_models.py
```

## 핵심 알고리즘

- **Train(60%)**: ANN 학습
- **Validation(20%)**: 990개 θ값(1%~99%, 0.1% 단위) 중 Sharpe 최대인 θ\* 선택
- **Test(20%)**: θ\* 적용 후 1,000회 부트스트래핑으로 95% 신뢰구간 산출, 벤치마크와 비교

부도 확률이 θ 미만인 대출만 투자하며, 투자 시 IRR, 거절 시 무위험수익률을 포트폴리오 수익률로 사용해 Sharpe Ratio를 계산합니다.

## 모델 설정 (`src/config.py`)

은닉층 `(256, 128, 64)` · 활성화함수 `relu` · L2 정규화(`alpha`) `0.001` · `batch_size` `512` · `max_iter` `1000` · `early_stopping` `True`

</details>
