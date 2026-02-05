# 🏦 Lending Club 신용평가 프로젝트 - ANN 모델 기반 투자 최적화

## 📋 프로젝트 개요

Lending Club 데이터로 **ANN(Artificial Neural Network) 모델**을 학습하여 부도 예측 및 **Sharpe Ratio 최대화 투자 임계값(θ*)** 도출

**목표**: 부도 확률 예측 → 최적 임계값(θ*) 탐색 → Sharpe Ratio 최대화 투자 전략 수립

---

## 🎯 프로젝트 목표

- ✅ ANN 신경망을 사용한 대출 부도 예측
- ✅ 최적 임계값(θ*) 자동 탐색으로 Sharpe Ratio 최대화
- ✅ 포트폴리오 기대 수익률 및 위험도 평가
- ✅ 벤치마크(모든 대출 승인) 대비 성과 개선도 측정

---

## 🔄 전체 파이프라인 흐름

```
1️⃣ 데이터 분할 (층화추출)
   Train(60%) / Validation(20%) / Test(20%)
   
2️⃣ ANN 모델 학습 (Train 데이터)
   - 은닉층: [256, 128, 64]
   - 활성화 함수: ReLU
   - Dropout rate: 0.3
   
3️⃣ Validation 평가 + 최적 임계값(θ*) 탐색
   → Sharpe Ratio 최대화하는 θ* 찾기
   
4️⃣ Test 평가 (θ* 적용)
   → 최종 Sharpe, 포트폴리오 수익률, 신뢰구간
   
5️⃣ 벤치마크 비교
   → 모든 대출 승인 시나리오 vs ANN 전략
```

---

## 📁 프로젝트 구조

```
Lending-Club-PRJ_ANN/
├── ann_venv_final/               # Python 가상환경 (최종)
├── ann_venv_new/                 # Python 가상환경 (테스트)
├── src/
│   ├── config.py                 # ✨ 중앙 설정 파일
│   ├── utils.py                  # 로거 유틸
│   ├── preprocess_pipeline.py    # 데이터 전처리
│   ├── train_models.py           # Random Split 학습
│   └── train_models_random.py    # K-Fold CV 학습
├── notebooks/                     # EDA/분석 노트북
│   ├── 01_label_definition.ipynb
│   ├── 02_eda_visualization.ipynb
│   └── 03_modeling.ipynb
├── models/                        # 저장된 모델들
├── data/
│   ├── external/                 # 외부 데이터 (국채 수익률 등)
│   ├── processed/                # 전처리된 데이터
│   └── raw/
├── requirements.txt               # 패키지 목록
├── lending_club_2020_train.csv   # 원본 데이터
├── experiments_log.csv            # 실험 로그
└── README.md                      # 이 파일
```

---

## ⚙️ 환경 설정

### 가상환경 생성 (첫 설정)
```bash
python3 -m venv ann_venv_new
source ann_venv_new/bin/activate
pip install -r requirements.txt
```

### 이후 실행
```bash
source ann_venv_new/bin/activate
python src/train_models.py
```

---

## 📊 핵심 알고리즘

### Sharpe Ratio 계산 (IRR 기반)

**공식**:
$$\text{Sharpe} = \frac{E[R_p] - E[R_f]}{\sigma[R_p]}$$

**투자 결정 로직**:
```python
# 부도 확률 < θ인 대출만 투자
invest = (pred_prob < θ)

# 포트폴리오 수익률 구성
portfolio_return = {
    IRR          if 투자 (pred_prob < θ)
    rf_return    if 거절 (pred_prob ≥ θ)
}

# Sharpe 계산
Sharpe = (mean(portfolio_return) - mean(rf_return)) / std(portfolio_return)
```

### IRR(내부수익률) 계산

**원리금균등상환** 기반 현금흐름:
```
월 상환액 = P × [r(1+r)^n] / [(1+r)^n - 1]
현금흐름 = [-P, A, A, ..., A] (부도 시 조기 종료)

월별 IRR → 연율화: annual_irr = (1 + monthly_irr)^12 - 1
```

### 최적 임계값(θ*) 탐색

**목표**: Sharpe Ratio 최대화하는 θ 찾기

**Validation 단계 (θ* 결정)**:
1. Validation 데이터로 990개 θ값(1%~99%, 0.1% 단위) 각각 테스트
2. 각 θ에 대해 Sharpe Ratio 계산
3. Sharpe 최대인 **θ* 1개 선택**

**Test 단계 (신뢰구간 계산)**:
1. 선택된 θ*를 Test 데이터에 적용
2. 1,000회 부트스트래핑 실행 → Sharpe 분포 생성
3. **95% 신뢰구간** 계산 (하위 2.5%, 상위 97.5% 백분위수)

---

## 🏃 실행 방법

### 1. 환경 활성화
```bash
source ann_venv_new/bin/activate
```

### 2. 데이터 전처리
```bash
cd src
python preprocess_pipeline.py
```

### 3. 모델 학습 (ANN)
```bash
python train_models.py
```

**생성 파일**:
- `models/ANN_final.pkl` - 학습된 모델
- `models/optimal_threshold.png` - θ* 시각화
- `experiments_log.csv` - 실험 결과 기록

---

## 📋 모델 설정

### ANN (Artificial Neural Network)

**하이퍼파라미터**:
```python
{
    'hidden_layers': [256, 128, 64],
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'dropout_rate': 0.3,
    'validation_split': 0.2,
    'random_state': 42
}
```

---

## 🎯 부도(Default) 정의

**부도 상태** (Target = 1):
```python
[
    'Charged Off',
    'Default',
    'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off'
]
```

**정상 상태** (Target = 0):
```python
[
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid'
]
```

---

## 📈 평가 지표

### 1. ROC-AUC
분류 성능 평가 - 부도 예측 정확도

### 2. Sharpe Ratio
투자 성과 평가 - 단위 위험당 초과 수익률
$$\text{Sharpe} = \frac{\text{평균 수익률} - \text{무위험률}}{\text{표준편차}}$$

### 3. 평균 수익률
포트폴리오의 기대 수익

### 4. 투자 승인 비율
전체 대출 중 투자 비율 (θ* < 부도확률)

---

## 🔑 주요 설정값 (config.py)

| 설정 | 값 | 설명 |
|-----|-----|------|
| `MODELS_TO_USE` | `['ANN']` | 사용 모델 (ANN만 사용) |
| `INVESTMENT_THRESHOLD` | 0.15 | 부도 확률 15% 이상이면 투자 거절 |
| `SAMPLE_FRAC` | 0.3 | 테스트 모드에서 사용할 데이터 비율 |
| `USE_SAMPLE` | False | 현재는 전체 데이터 사용 |
| `DEFAULT_RF_3YR` | 0.061 | 3년 만기 국채 기본 수익률 |
| `DEFAULT_RF_5YR` | 0.104 | 5년 만기 국채 기본 수익률 |

---

## 📊 최종 평가 (Test Set)

### 1️⃣ 포트폴리오 성과
```
- 평균 수익률: E[portfolio_return]
- Sharpe Ratio: (mean - rf_mean) / std
- 투자 승인 비율: 전체 대출 중 투자 비율
```

### 2️⃣ 벤치마크 비교 (모든 대출 승인 시나리오)
```
벤치마크 Sharpe = (E[모든 대출 IRR] - E[국채]) / σ[모든 대출 IRR]

성과 개선도 = (ANN Sharpe - 벤치마크) / 벤치마크 × 100%
```

### 3️⃣ 신뢰구간
```
95% 신뢰구간: [CI_lower, CI_upper]
→ Test Sharpe 분포의 하위 2.5%, 상위 97.5% 백분위수
```

---

## ✨ 리팩토링 완료사항

### 1️⃣ Python 환경 설정
- **가상환경 생성**: 독립적인 Python 3.9 환경 구성
- **패키지 설치**: 필수 패키지만 선별하여 최소화

### 2️⃣ 중앙화된 설정 관리 (config.py)
모든 경로와 상수를 한 곳에서 관리:
- 경로 설정 (데이터, 모델, 로그)
- 4개 모델의 하이퍼파라미터
- 부도 정의 & 제외 변수 리스트
- 투자 임계값, 국채 수익률

### 3️⃣ 모델 코드 리팩토링

#### **train_models.py** (Random Split)
```
🔥 구현된 모델: ANN (신경망)

📊 평가 방식:
├── 60/20/20 랜덤 분할
├── ROC-AUC 계산
├── Sharpe Ratio 산출 (투자 관점)
├── 최적 임계값 θ* 부트스트래핑 탐색
└── Test 세트로 최종 성과 평가
```

#### **train_models_random.py** (K-Fold Cross-Validation)
```
🔢 K-Fold 설정:
├── 5-Fold Cross-Validation
├── shuffle=True, random_state=42
└── 각 Fold별 독립적인 모델 학습

📊 결과 저장:
├── 매 Fold마다 experiments_log.csv에 기록
├── 앙상블 평가도 Fold별로 진행
└── 모델별 가중치를 동적으로 계산
```

### 4️⃣ Requirements 단순화
불필요한 Jupyter 관련 패키지 제거 (100개 → 20개 핵심 패키지)

---

## 🔄 개선사항 요약

### Before (다중 모델 기반)
```
❌ XGBoost, LightGBM, LogisticRegression 병행
❌ 하드코딩된 경로와 파라미터
❌ 코드 중복 많음
❌ 모델 관리 복잡
```

### After (ANN + 중앙 설정)
```
✅ ANN만 사용 (단순화)
✅ config.py에서 모든 설정 관리
✅ 코드 재사용성 높음
✅ 최적 임계값 θ* 부트스트래핑
✅ 벤치마크 비교 분석
✅ 최종 평가: 포트폴리오 IRR, 부트스트랩 분포
```

---

## 📝 실험 로깅

모든 실험 결과는 `experiments_log.csv`에 기록:
```
Date, Model, Split, AUC, Sharpe, Avg_Return, Duration, Params, Memo
```

---

## 🚀 다음 단계

1. **하이퍼파라미터 튜닝**: `config.py`의 `MODEL_PARAMS` 조정
2. **최적 임계값**: 부트스트래핑으로 자동 탐색
3. **변수 선택**: `EXCLUDE_COLS` 조정하여 특성 공학 진행
4. **Out-of-Sample 테스트**: Test 세트로 최종 성과 평가

---

## 🔧 문제 해결

### numpy-financial 설치
```bash
pip install numpy-financial
```

### 패키지 업그레이드
```bash
pip install -r requirements.txt --upgrade
```

---

## 📞 주요 특징 (v0.7)

- ✅ **일관된 Sharpe 계산**: Validation, 최적 임계값 탐색, Test 모두 **같은 IRR 기반 Sharpe** 사용
- ✅ **벤치마크**: 모든 대출 승인 vs ANN 선택적 투자 비교
- ✅ **효율성**: 투자 대출만 IRR 계산 → 연산량 50% 감소
- ✅ **신뢰성**: 부트스트랩 신뢰구간으로 결과 검증
- ✅ **자동화**: 일관된 Sharpe 계산 및 최종 평가 완전 자동화

---

**프로젝트 상태**: ✅ Ready to Run
