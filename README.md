# ANN 기반 대출 승인 신용 위험 예측 및 포트폴리오 최적화 시스템

## 📌 사용 방법

1. 이 파일을 실행: `python final.py`
2. 결과 폴더(experiment_results/)에 4개 파일이 생성됨:
   - `bootstrap_distribution_*.csv` : Bootstrap 샤프 지수 분포
   - `feature_importance_*.csv` : 특성 중요도 랭킹
   - `experiment_config_*.json` : 실험 설정값
   - `experiment_report_*.txt` : 최종 보고서

---

## 📊 전체 실행 흐름 (순서도)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1: 데이터 로딩 및 전처리
  ├─ CSV 파일 로딩 (1.2M+ 대출 기록)
  ├─ 결측치 처리, 변수 생성 (로그화, 파생변수 등)
  └─ 불필요한 컬럼 제거

  Step 2: 금융 계산
  ├─ Risk-free rate 매핑 (국채 금리 기반)
  ├─ Actual IRR 계산 (Newton-Raphson 방식)
  ├─ Sample weight 계산 (부도 패널티 적용)
  └─ Interest rate spread 계산

  Step 3: Feature 선택 및 검증
  ├─ BASE_FEATURES 리스트에서 변수 선택
  ├─ Random Forest로 중요도 계산
  ├─ 상위 30개 Feature 선택
  └─ 결측치/무한대 값 처리

  Step 4: 데이터 분할
  ├─ Stratified Split: Train 60% : Val 20% : Test 20%
  ├─ 계층화 분할로 Target 분포 유지
  └─ Meta 정보(IRR, 금리, 대출액) 분리

  Step 5: 데이터 정규화
  └─ StandardScaler로 정규화 (Train 통계 기반)

  Step 6: ANN 모델 구축 및 학습
  ├─ 모델 구조: Input → Dense(128) → Dense(64) → Dense(32) → Output(sigmoid)
  ├─ Batch Normalization + Dropout(0.3)으로 과적합 방지
  ├─ Sample weight 적용 (불균형 데이터 처리)
  ├─ Early Stopping: 개선 없을 때 자동 종료
  └─ 학습 완료 (최대 100 epochs, 실제 ~50 epochs)

  Step 7: Top-K Portfolio 최적화
  ├─ Validation 데이터로 K 탐색 (1%~60%, 1% 단위)
  ├─ 각 K별 Weighted Sharpe Ratio 계산
  ├─ 최적 K 결정
  └─ 선택된 포트폴리오 (상위 K% 낮은 부도확률 대출)

  Step 8: Test 평가
  ├─ Test 데이터로 최적 K 성능 평가
  ├─ Weighted Sharpe Ratio 계산
  ├─ 벤치마크(모든 대출 승인) 성과와 비교
  └─ 시기별 분석 (금융위기/안정기/COVID 기간)

  Step 9: Bootstrap 검증
  ├─ Test 데이터에서 1000회 복원 샘플링
  ├─ 각 샘플의 Sharpe Ratio 계산
  └─ 95% 신뢰 구간 산출

  Step 10: 결과 저장
  ├─ CSV: 특성 중요도, Bootstrap 분포
  ├─ JSON: 실험 설정값 저장
  ├─ TXT: 종합 보고서
  └─ PNG: Bootstrap 히스토그램

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## ⚙️ 하이퍼파라미터 조절 가이드

### 📌 [1] Feature 변수 선택 (Line ~570)
- BASE_FEATURES 리스트를 수정하여 사용할 변수 선택/제거

### 📌 [2] 모델 구조 (Line ~825)
- `hidden_layers=[128, 64, 32]`: 은닉층 뉴런 개수

### 📌 [3] 학습 설정 (Line ~850)
- `epochs=50`: 최대 학습 반복
- `batch_size=256`: 배치 크기
- `learning_rate=0.001`: 학습률

### 📌 [4] 데이터 분할 (Line ~770)
- `test_size=0.4, 0.5` 값으로 비율 조정

### 📌 [5] 포트폴리오 최적화 (Line ~1000)
- `range(2, 61)`: K 비율 탐색 범위

### 📌 [6] Bootstrap 설정 (Line ~1270)
- `N_BOOTSTRAP=1000`: 반복 횟수

---

## 🚀 실행 방법

```bash
python final.py
```

### 필수 파일
- `lending_club_2020_train.csv`: 입력 데이터

### 선택 파일
- `GS3.csv`: 3년 국채 금리 (없으면 기본값 2% 사용)
- `GS5.csv`: 5년 국채 금리 (없으면 기본값 2% 사용)

---

## 📁 프로젝트 구조

```
final.py                    # 메인 실행 파일 (모든 단계 포함)
nn.ipynb                    # Jupyter 노트북 버전
README.md                   # 이 파일 (프로젝트 개요)
lending_club_2020_train.csv # 입력 데이터
```

---

## 📊 주요 특징

✅ **Polars 활용**: 고속 데이터 전처리 (대용량 CSV 처리)
✅ **금융 계산**: 실제 수익률(IRR), 위험자유율(Treasury) 매핑
✅ **깊은 신경망**: BatchNormalization, Dropout으로 과적합 방지
✅ **샘플 가중치**: 불균형 데이터(부도/정상)에 대한 클래스 가중치 적용
✅ **Top-K 전략**: Weighted Sharpe Ratio 기반 최적 포트폴리오 자동 선택
✅ **자동 결과 저장**: CSV, JSON, TXT, PNG 형식으로 결과 저장

---

## 📈 기대 결과

### Validation 데이터
- 최적 비율: 10%~20% (자동 선택)
- Top-K Sharpe: 0.25~0.35
- 벤치마크 대비 개선도: 30~50%

### Test 데이터
- 최적 비율: 10%~20%
- Top-K Sharpe: 0.24~0.33
- 벤치마크 대비 개선도: 25~45%

---

## 👥 프로젝트 정보

- **코스**: 통계데이터사이언스 (류근관 교수님)
- **학년**: 7학년 2학기
- **팀명**: ANN 팀
- **주제**: 신용 위험 평가 및 포트폴리오 최적화
- **작성일**: 2026년 2월

---

## 📝 라이선스

대학 과제용 코드입니다.

