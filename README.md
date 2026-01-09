# 📊 Olist E-commerce 데이터 분석 프로젝트

## 1. 프로젝트 개요 (Project Overview)

### 🎯 목적 및 배경
1인 프로젝트로 브라질의 Olist 이커머스 데이터를 활용하여 **매출 증대 및 효율적인 투자 전략을 수립**하는 것을 목표로 합니다. 데이터 기반의 의사결정을 위해 고객 행동 패턴(RFM), 배송 프로세스, 리뷰 텍스트(NLP)를 다각도로 분석하였습니다.

### 🏆 주요 목표
*   **고객 세분화:** RFM 분석을 통해 VIP, 충성 고객, 이탈 위험 고객 등을 분류하고 타겟 마케팅 전략 도출
*   **배송 프로세스 최적화:** 배송 지연과 고객 만족도(리뷰 점수) 간의 상관관계 규명
*   **리뷰 감성 분석:** 자연어 처리(NLP)를 통해 이탈 고객과 잠재 우수 고객의 니즈 파악
--
### 데이터셋
Brazilian E-Commerce Public Dataset by Olist (Kaggle)
이 데이터셋은 브라질 전자상거래 플랫폼 Olist에서 제공하는 공개 데이터입니다. 2016년부터 2018년 사이에 이루어진 주문 데이터를 포함하며 주문, 고객, 판매자, 제품, 결제, 리뷰, 지리정보 등을 분석할 수 있는 다차원적인 구조로 되어 있습니다.
<img width="2486" height="1496" alt="image" src="https://github.com/user-attachments/assets/459cee62-46ea-420f-a009-5db5b29d1d2f" />

# 데이터 구성 개요
파일 구성: 총 9개의 CSV 파일로 구성
총 주문 수 (orders): 약 100,000건의 주문 데이터 포함  
데이터 구조: 관계형 데이터베이스(Relational Database) 구조로, 각 테이블이 order_id, customer_id 등의 키(key)를 통해 서로 연결되어 있습니다. (Star Schema 형태)
전체 컬럼 수: 약 54개 (모든 테이블 합계)  

📌 테이블 개요: 주문, 고객, 결제, 리뷰, 제품, 판매자, 지리정보, 카테고리명 번역  

| 파일명 (Dataset Name) | 행 (Rows) | 열 (Cols) | 설명 |
| :--- | :--- | :--- | :--- |
| **olist_customers_dataset** | 99,441 | 5 | 고객 ID 및 주소지(우편번호, 도시, 주) 정보 |
| **olist_orders_dataset** | 99,441 | 8 | 주문 상태, 구매/배송/승인 시간 등 주문 핵심 정보 |
| **olist_order_items_dataset** | 112,650 | 7 | 주문된 상품 상세 정보 (가격, 배송비, 상품 ID 등) |
| **olist_order_payments_dataset** | 103,886 | 5 | 결제 방식(신용카드, 볼레토 등), 할부, 결제 금액 정보 |
| **olist_order_reviews_dataset** | 99,224 | 7 | 고객 리뷰 점수(1~5점) 및 리뷰 텍스트 |
| **olist_products_dataset** | 32,951 | 9 | 상품 카테고리, 무게, 길이, 부피 등 상품 상세 스펙 |
| **olist_sellers_dataset** | 3,095 | 4 | 판매자 ID 및 판매자 위치 정보 |
| **olist_geolocation_dataset** | 약 1,000,163 | 5 | 브라질 우편번호(Zip Code)에 따른 위도/경도 좌표 정보 |
| **product_category_name_translation** | 71 | 2 | 포르투갈어 카테고리명을 영어로 번역한 매핑 테이블 |
---

## 2. 기술 스택 (Tech Stack)

| Category | Stacks |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white) |
| **IDE & Env** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=Jupyter&logoColor=white) ![VSCode](https://img.shields.io/badge/VS_Code-007ACC?style=flat&logo=visual-studio-code&logoColor=white) |
| **Data Analysis** | `Pandas`, `Numpy` (전처리 및 통계 분석) |
| **Visualization** | `Matplotlib`, `Seaborn` (EDA 시각화) |
| **Machine Learning** | `Scikit-learn` (Logistic Regression), `CatBoost` |
| **NLP** | `HuggingFace`, `TfidfVectorizer` (리뷰 텍스트 분석) |

---

## 3. 아키텍처 (Architecture)
데이터 수집부터 전처리, 모델링, 인사이트 도출까지의 분석 파이프라인입니다.

1.  **Data Gathering:** Olist 데이터셋(Customers, Orders, Reviews 등 8개 테이블) 로드 [1]
2.  **Preprocessing:** 결측치/이상치 처리 (배송일 음수값 보정, 가격 0원 등 처리) [5]
3.  **EDA & Feature Eng:** 배송 지연일 산출, RFM 점수화
4.  **Modeling:** Logistic Regression(배송-리뷰 관계), RFM Segmentation, NLP Sentiment Analysis
5.  **Insight:** 타겟 마케팅 전략 및 운영 개선안 제안

---

## 4. 주요 기능 및 분석 내용 (Key Features)

### 4.1 EDA 및 데이터 전처리
*   **배송 프로세스 분석:** `구매 → 승인 → 택배사 전달 → 고객 수령` 단계별 타임라인을 분석하고, '승인 전 배송 시작' 등의 이상치를 정제했습니다.
*   **가격-배송비 상관관계:** 고가 상품일수록 배송비가 비례하여 증가하는지 분석한 결과, 상관계수 약 0.43으로 1:1 비례 관계가 아님을 확인했습니다.

### 4.2 RFM 고객 세분화 (Customer Segmentation)
*   **세분화 기준:** 이커머스 특성을 반영하여 **Recency(최근성)**에 높은 가중치를 부여했습니다.
    *   Recency 5점: 0-30일 이내 구매 (즉시 리텐션 대상)
    *   Frequency: 1회 구매 비중이 94%로 압도적으로 높아 재구매 유도가 핵심 과제로 식별됨
*   **고객 등급 정의:** VIP, 충성 고객, 잠재 고객, 이탈 위험, 휴면 고객 등으로 분류하여 맞춤형 전략 수립

### 4.3 배송-리뷰 상관관계 및 NLP 분석
*   **통계적 검증:** 배송 지연과 리뷰 점수 간의 **음의 상관관계(Pearson -0.30)**를 확인했으며, P-value 0.000으로 통계적 유의성을 입증했습니다.
*   **NLP 감성 분석:** `HuggingFace`와 `TfidfVectorizer`를 활용해 리뷰 텍스트를 분석하고, 부정 리뷰(배송 지연 등)와 긍정 리뷰의 키워드 패턴을 비교했습니다.

---

## 5. 결과 및 성과 (Results & Achievements)

### 📈 정량적 분석 결과
*   **배송 퍼포먼스:** 정시 배송률 **91.80%** 달성, 평균 배송일은 예정일보다 약 11.1일 빠름
*   **고객 이탈률:** 전체 고객의 약 **19.45% (17,897명)**가 이탈 고객으로 식별됨
*   **매출 기여도:** 신규 고객 비중(23.36%)은 높으나 매출 기여도(8.29%)는 낮음. 반면, **잠재 우수 고객(VIP 후보)이 전체 매출의 32.47%를 차지**

### 💡 문제 해결 및 비즈니스 인사이트
1.  **골든타임 마케팅 제안:** 신규 고객의 재구매율이 매우 낮으므로, 첫 구매 후 **30일 이내(Recency 5점 구간)**에 재구매를 유도하는 프로모션이 필수적입니다
2.  **VIP 집중 관리:** 매출의 30% 이상을 차지하는 잠재 우수 고객에게 프리미엄 혜택 및 멤버십을 제공하여 이탈을 방지해야 합니다.
3.  **배송 경험 개선:** 배송 지연이 리뷰 평점 하락의 주요 원인이므로, 물류 프로세스 최적화가 고객 만족도 관리에 핵심임을 데이터로 증명했습니다 
---

### 📂 폴더 구조 (Directory Structure)
```bash
project_olist_ecommerce/
├── project_dataset/       # 원본 데이터셋
├── RFM.ipynb              # 고객 세분화(RFM) 분석 코드
├── logistic.ipynb         # 물류 분석 코드
├── review.ipynb           # 리뷰 데이터 분석 코드
├── data-gathering.ipynb   # 데이터 수집 및 병합
└── README.md              # 프로젝트 문서
