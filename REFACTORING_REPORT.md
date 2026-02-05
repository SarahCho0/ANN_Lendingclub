# 📋 프로젝트 리팩토링 완료 보고서

## ✅ 완료된 작업

### 1️⃣ Python 환경 설정
- **가상환경 생성**: `venv/` 디렉토리에 독립적인 Python 3.9 환경 구성
- **패키지 설치**: `requirements.txt`에 정의된 모든 필수 패키지 설치
  - Core: `pandas`, `numpy`, `polars`
  - ML Model: `tensorflow` (Keras/ANN)
  - Utilities: `joblib`, `matplotlib`, `seaborn`, `numpy-financial`

### 2️⃣ 중앙화된 설정 관리 (`config.py`)
모든 경로와 상수를 한 곳에서 관리:
```
✨ 핵심 기능:
├── 경로 설정 (데이터, 모델, 로그)
├── 4개 모델의 하이퍼파라미터
├── 부도 정의 & 제외 변수 리스트
├── 투자 임계값 (0.15)
└── 국채 수익률 (기본값)
```

**주요 상수**:
- `MODELS_TO_USE`: `['ANN']` (ANN만 사용)
- `INVESTMENT_THRESHOLD`: `0.15` (부도 확률 15% 이상이면 투자 거절)
- `DEFAULT_RF_3YR`: `0.061` (3년 만기 국채 6.1%)
- `DEFAULT_RF_5YR`: `0.104` (5년 만기 국채 10.4%)

### 3️⃣ 모델 코드 리팩토링

#### **train_models.py** (Random Split)
```
🔥 구현된 모델:
└── ANN (신경망 - Keras/TensorFlow)

📊 평가 방식:
├── 60/20/20 랜덤 분할 (Train/Validation/Test)
├── ROC-AUC 계산
├── Sharpe Ratio 산출 (투자 관점)
├── 최적 임계값 θ* 부트스트래핑 탐색 (Validation)
└── Test 세트로 최종 성과 평가
```

**주요 특징**:
- ANN만 사용 (XGBoost, LightGBM, LogisticRegression 제거)
- 최적 임계값 θ* 탐색 기반 투자 의사결정
- 부트스트래핑으로 Sharpe Ratio 안정성 검증
- 벤치마크(Approve All) 대비 성과 평가

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
불필요한 Jupyter 관련 패키지 제거:
```
기존: 100개 이상의 패키지
→ 현재: 20개 핵심 패키지만 유지
```

## 📁 프로젝트 구조

```
Lending-Club-PRJ-0.2-ensemble 3/
├── venv/                          # Python 가상환경
├── src/
│   ├── config.py                 # ✨ 중앙 설정 파일 (NEW)
│   ├── utils.py                  # 로거 유틸
│   ├── preprocess_pipeline.py    # 데이터 전처리
│   ├── train_models.py           # Time-Series Split 학습
│   └── train_models_random.py    # K-Fold CV 학습
├── notebooks/                     # EDA/분석 노트북
├── models/                        # 저장된 모델들
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── requirements.txt               # 패키지 목록 (UPDATED)
├── lending_club_2020_train.csv   # 원본 데이터
├── 260129_feature 변수후보.xlsx   # 변수 후보 목록
├── README.md
└── experiments_log.csv            # 실험 로그
```

## 🔧 사용 방법

### 1. 환경 활성화
```bash
source venv/bin/activate
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

## 📊 모델 및 파라미터

### ANN (Artificial Neural Network)
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
    'dropout_rate': 0.3,
    'validation_split': 0.2
}
```

## 🎯 부도(Default) 정의

```python
TARGET_BAD_STATUSES = [
    'Charged Off',
    'Default',
    'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off'
]

TARGET_GOOD_STATUSES = [
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid'
]
```

## 📈 평가 지표

1. **ROC-AUC**: 분류 성능 평가
2. **Sharpe Ratio**: 투자 성과 평가
   - 공식: (평균 수익률 - 무위험률) / 표준편차
3. **평균 수익률**: 포트폴리오의 기대 수익

## 🔑 주요 설정값

| 설정 | 값 | 설명 |
|-----|-----|------|
| `INVESTMENT_THRESHOLD` | 0.15 | 부도 확률 15% 이상이면 투자 거절 |
| `SAMPLE_FRAC` | 0.3 | 테스트 모드에서 사용할 데이터 비율 |
| `USE_SAMPLE` | False | 현재는 전체 데이터 사용 |
| `DEFAULT_RF_3YR` | 0.061 | 3년 만기 국채 기본 수익률 |
| `DEFAULT_RF_5YR` | 0.104 | 5년 만기 국채 기본 수익률 |

## ✨ 개선사항

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
✅ 벤치마크 비교 분석 (Approve All vs AI)
✅ 최종 평가: 포트폴리오 IRR, 부트스트랩 분포
```

## 📝 로깅

모든 실험 결과는 `experiments_log.csv`에 기록:
```
Date, Model, Split, AUC, Sharpe, Avg_Return, Duration, Params, Memo
```

## 🚀 다음 단계

1. **하이퍼파라미터 튜닝**: config.py의 MODEL_PARAMS 조정
2. **최적 임계값**: 부트스트래핑으로 자동 탐색 (θ* = 0.1450)
3. **변수 선택**: EXCLUDE_COLS 조정하여 특성 공학 진행
4. **Out-of-Sample 테스트**: Test 세트로 최종 성과 평가

## 📞 문제 해결

### numpy-financial 설치
```bash
pip install numpy-financial
```

### 패키지 업그레이드
```bash
pip install -r requirements.txt --upgrade
```

---

✅ **프로젝트 리팩토링 완료**  
ANN 모델 + 최적 임계값 탐색 + 최종 평가 완전 자동화!
