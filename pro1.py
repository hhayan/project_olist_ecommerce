# %matplotlib inline
# %pip install statsmodels

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os
from scipy.stats import zscore
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from scipy.stats import kruskal

import warnings

warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# %pip install pandas-summary

# 파일들이 있는 폴더 경로
folder_path = 'C:/Users/mumu1/Desktop/project_movie_data/project_dataset'

# 파일 이름을 변수에 할당
o_df_customers = pd.read_csv(os.path.join(folder_path, 'olist_customers_dataset.csv'), encoding='ISO-8859-1')
o_df_geolocation = pd.read_csv(os.path.join(folder_path, 'olist_geolocation_dataset.csv'), encoding='ISO-8859-1')
o_df_order_items = pd.read_csv(os.path.join(folder_path, 'olist_order_items_dataset.csv'), encoding='ISO-8859-1')
o_df_order_payments = pd.read_csv(os.path.join(folder_path, 'olist_order_payments_dataset.csv'), encoding='ISO-8859-1')
o_df_order_reviews = pd.read_csv(os.path.join(folder_path, 'olist_order_reviews_dataset.csv'), encoding='ISO-8859-1')
o_df_products = pd.read_csv(os.path.join(folder_path, 'olist_products_dataset.csv'), encoding='ISO-8859-1')
o_df_sellers = pd.read_csv(os.path.join(folder_path, 'olist_sellers_dataset.csv'), encoding='ISO-8859-1')
o_df_product_category_name_translation = pd.read_csv(os.path.join(folder_path, 'product_category_name_translation.csv'), encoding='utf-8-sig')


print("✅ 모든 파일이 개별적으로 메모리에 로드되었습니다.")

# 카피본 생성
df_customers = o_df_customers.copy()
df_geolocation = o_df_geolocation.copy()
df_order_items = o_df_order_items.copy()
df_order_payments = o_df_order_payments.copy()
df_order_reviews = o_df_order_reviews.copy()
df_products = o_df_products.copy()
df_sellers = o_df_sellers.copy()
df_product_category_name_translation = o_df_product_category_name_translation.copy()
# 8개 데이터프레임의 결측값 분석
def check_missing(dfs, df_names):
    for df, name in zip(dfs, df_names):
        print(f"\n📊 {name} 데이터프레임 결측값 분석")
        
        missing_info = df.isnull().sum()
        m_pct = (missing_info / len(df)) * 100
        
        if missing_info.sum() == 0:
            print("✅ 결측값 없음. 완전")
        else:
            print("⚠️ 결측치 존재")
            missing_sum = pd.DataFrame({
                '결측수': missing_info,
                '결측율(%)': m_pct,
            }).round(2)
            missing_sum = missing_sum[missing_sum['결측수'] > 0]
            display(missing_sum)

# 사용 예시
original_dfs = [ 
    o_df_customers, o_df_geolocation, o_df_order_items,
    o_df_order_payments, o_df_order_reviews, o_df_products,
    o_df_sellers, o_df_product_category_name_translation,
]

df_names = [
    "customers", "geolocation", "order_items",
    "order_payments", "order_reviews", "products",
    "sellers", "product_category_name_translation",
]

check_missing(original_dfs, df_names)
# 전처리
# df_products

# 결측치
# 텍스트 리뷰 활용: 제목/메시지는 결측을 그대로 두고 "No Comment" 처리
# 이상치

# df_order_items 이상치 탐지
from sklearn.ensemble import IsolationForest

# 모델 초기화 (contamination은 이상치 비율을 가정)
model = IsolationForest(contamination=0.01) # 1%의 이상치가 있다고 가정

# 모델 학습 및 이상치 예측 (-1은 이상치, 1은 정상 데이터)
df_order_items['outlier_flag'] = model.fit_predict(df_order_items[['price', 'freight_value']])

# 이상치 개수 계산
outlier_count = df_order_items[df_order_items['outlier_flag'] == -1].shape[0]

# f-string을 사용해 결과 출력
print(f"전체 데이터 행 수: {len(df_order_items):,}")
print(f"IsolationForest 모델이 탐지한 이상치 개수: {outlier_count:,}")
print(f"이상치 비율: {(outlier_count / len(df_order_items)):.2%}")

# 이상치 시각화: 산점도 그리기
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='price', 
    y='freight_value', 
    data=df_order_items, 
    hue='outlier_flag', 
    palette=['red', 'blue'], 
    s=20,
    alpha=0.7
)

plt.title('가격과 운송료에 대한 이상치 시각화', fontsize=16)
plt.xlabel('가격 (Price)', fontsize=12)
plt.ylabel('운송료 (Freight Value)', fontsize=12)
plt.legend(title='이상치 여부', labels=['이상치', '정상'])
plt.grid(True)
plt.show()

# 실제 비즈니스적 맥락에서의 해석:
# 가격과 운송료가 모두 극단적으로 낮은 상품은 샘플, 이벤트용 상품, 또는 데이터 입력 오류일 가능성이 있습니다.
# 이상치는 값 대체해서 처리

# 이상치 대체 함수
def winsorize_outliers(df, column_name):
    """
    1%와 99% 백분위수 값을 기준으로 이상치를 대체합니다.
    """
    lower_bound = df[column_name].quantile(0.01)
    upper_bound = df[column_name].quantile(0.99)
    
    # 1%보다 작은 값을 1% 값으로, 99%보다 큰 값을 99% 값으로 대체
    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
    return df

# 'price'와 'freight_value' 컬럼의 이상치 대체
df_order_items = winsorize_outliers(df_order_items.copy(), 'price')
df_order_items = winsorize_outliers(df_order_items.copy(), 'freight_value')

# 이상치 처리 후의 데이터 통계량 확인
print("=== 'price' 컬럼 이상치 처리 후 통계량 ===")
print(df_order_items['price'].describe())
print("\n=== 'freight_value' 컬럼 이상치 처리 후 통계량 ===")
print(df_order_items['freight_value'].describe())

'''
df_review 

1. 데이터 상황 요약
review_comment_title: 결측치 88% → 고객 대부분이 제목은 아예 작성하지 않음.
review_comment_message: 결측치 59% → 절반 이상이 코멘트를 남기지 않음.
review_score (별점): 결측치 없음 (모든 리뷰는 점수 필수).
'''

# --- 리뷰 메시지 작성 여부 플래그 생성 ---
df_order_reviews["has_comment"] = df_order_reviews["review_comment_message"].notnull().astype(int)

# --- 리뷰 제목/메시지 결측치 "No Comment"로 치환 ---
df_order_reviews["review_comment_title"] = df_order_reviews["review_comment_title"].fillna("No Comment")
df_order_reviews["review_comment_message"] = df_order_reviews["review_comment_message"].fillna("No Comment")

# 확인
print(df_order_reviews[["review_score", "has_comment", "review_comment_title", "review_comment_message"]].head())
print("\n📌 has_comment 분포")
print(df_order_reviews["has_comment"].value_counts(normalize=True).round(3) * 100)

# orders 파일 읽어오기
file_path_absolute ='C:/Users/mumu1/Desktop/project_movie_data/project_dataset/olist_orders_dataset.csv'
o_df_order = pd.read_csv(file_path_absolute, encoding='ISO-8859-1')

df_order = o_df_order.copy()

# orders 데이터 탐색 : 누락, 중복, 이상 확인, 결측치 확인
display(o_df_order.head())
o_df_order.describe()
o_df_order.info()
o_df_order.isnull().sum()
# df_order 결측치 처리

# 1. 결측치를 확인할 컬럼 리스트 정의
missing_value_cols = ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']

# 2. 각 컬럼별로 결측치 여부를 확인하는 불리언 마스크(Boolean Mask) 생성
# 'isnull()' 함수는 결측치(NaN)일 때 True를 반환합니다.
approved_at_na = df_order['order_approved_at'].isnull()
carrier_date_na = df_order['order_delivered_carrier_date'].isnull()
customer_date_na = df_order['order_delivered_customer_date'].isnull()

# 3. 세 가지 마스크를 '|' (or) 연산자로 결합
# 이 조건들 중 하나라도 True인 행을 선택합니다.
na_mask = approved_at_na | carrier_date_na | customer_date_na

# 4. 마스크를 사용하여 결측값이 있는 행만 필터링
df_na = df_order[na_mask]

# 5. 필터링된 데이터의 일부를 확인
print("결측값이 있는 행의 데이터 샘플:")
print(df_na.head())

# 6. 결측값이 있는 행의 개수 확인
print(f"\n결측값이 있는 총 행의 수: {len(df_na)}")

# o_df_order 결측률
print(df_order.isnull().sum() / len(df_order))

# 5%이하 3개 컬럼의 결측치 제거
df_order.dropna(subset=['order_approved_at'], inplace=True)
df_order.dropna(subset=['order_delivered_carrier_date'], inplace=True)
df_order.dropna(subset=['order_delivered_customer_date'], inplace=True)

# 변경사항 확인
print("결측치 제거 후 df_order의 정보:")
print(df_order.info())

# 이상치 탐지: 계산한 배송시간 차이가 크거나 작은 경우
print('\n=== 이상값 확인 ===')

# 데이터 타입 변환
df_order["order_approved_at"] = pd.to_datetime(df_order["order_approved_at"])
df_order["order_purchase_timestamp"] = pd.to_datetime(df_order["order_purchase_timestamp"])
df_order["order_delivered_carrier_date"] = pd.to_datetime(df_order["order_delivered_carrier_date"])
df_order["order_delivered_customer_date"] = pd.to_datetime(df_order["order_delivered_customer_date"])
df_order["order_purchase_timestamp"] = pd.to_datetime(df_order["order_purchase_timestamp"])
df_order["order_estimated_delivery_date"] = pd.to_datetime(df_order["order_estimated_delivery_date"])

# 시간 차이 계산 (일 단위)
# 결제까지 걸린 시간: 주문승인일 - 결제일
df_order["purchase_to_approved"] = (df_order["order_approved_at"] - df_order["order_purchase_timestamp"]).dt.total_seconds()/86400
# 주문-배송 걸린 시간: 배송완료일 - 주문승인일
df_order["approved_to_carrier"] = (df_order["order_delivered_carrier_date"] - df_order["order_approved_at"]).dt.total_seconds()/86400
# 택배사-배송 걸린 시간: 배송완료일 - 택배사 전달일
df_order["carrier_to_customer"] = (df_order["order_delivered_customer_date"] - df_order["order_delivered_carrier_date"]).dt.total_seconds()/86400
# 계산-배송 걸린 시간: - 배송완료일 - 주문계산일
df_order["purchase_to_customer"] = (df_order["order_delivered_customer_date"] - df_order["order_purchase_timestamp"]).dt.total_seconds()/86400

# 모든 시간 계산 컬럼에서 음수 값만 찾기
# 시간 계산 컬럼 리스트
time_cols = ["purchase_to_approved","approved_to_carrier","carrier_to_customer","purchase_to_customer"]

# 각 컬럼별 음수 개수 계산
neg_counts = {col: (df_order[col] < 0).sum() for col in time_cols}

# 전체 음수 개수 (한 행이라도 음수인 경우)
total_neg = df_order[(df_order[time_cols] < 0).any(axis=1)].shape[0]

# 결과 출력
print("컬럼별 음수 개수:", neg_counts)
print("전체 음수 개수 (한 행이라도 음수):", total_neg)
# 1) 데이터 무결성 & 계산값 검증

# 이미 계산해둔 4개 지표가 맞는지 **허용 오차(±1초=1/86400일)**로 교차검증합니다.
# 또한 음수/이상치, delivered 이외 상태 존재 여부를 점검합니다.
# 1-1. datetime dtype 보장
datetime_cols = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
for c in datetime_cols:
    assert pd.api.types.is_datetime64_any_dtype(df_order[c]), f"{c}는 datetime 타입이어야 합니다."

# 1-2. 로직 재계산(검증용 임시 컬럼)
EPS = 1/86400  # 1초
calc = pd.DataFrame(index=df_order.index)
calc["purchase_to_approved_chk"] = (df_order["order_approved_at"] - df_order["order_purchase_timestamp"]).dt.total_seconds()/86400
calc["approved_to_carrier_chk"]  = (df_order["order_delivered_carrier_date"] - df_order["order_approved_at"]).dt.total_seconds()/86400
calc["carrier_to_customer_chk"]  = (df_order["order_delivered_customer_date"] - df_order["order_delivered_carrier_date"]).dt.total_seconds()/86400
calc["purchase_to_customer_chk"] = (df_order["order_delivered_customer_date"] - df_order["order_purchase_timestamp"]).dt.total_seconds()/86400

# 1-3. 기존 값과 일치 여부 확인
diffs = {
    "purchase_to_approved": (df_order["purchase_to_approved"] - calc["purchase_to_approved_chk"]).abs().max(),
    "approved_to_carrier": (df_order["approved_to_carrier"] - calc["approved_to_carrier_chk"]).abs().max(),
    "carrier_to_customer": (df_order["carrier_to_customer"] - calc["carrier_to_customer_chk"]).abs().max(),
    "purchase_to_customer": (df_order["purchase_to_customer"] - calc["purchase_to_customer_chk"]).abs().max(),
}
print("[검증] 최대 절대 오차(일):", diffs)
for k,v in diffs.items():
    assert v <= EPS, f"{k} 계산값이 사전 계산과 불일치 (max abs diff={v}일)"

# 1-4. 상태값 점검
status_counts = df_order["order_status"].value_counts(dropna=False)
print("\n[상태 분포]\n", status_counts)

# 1-5. delivered 필터 (상황에 따라 전체 vs delivered 별도 분석 가능)
df_deliv = df_order[df_order["order_status"]=="delivered"].copy()
print(f"\n[delivered 개수] {len(df_deliv):,} / 전체 {len(df_order):,}")

# 1-6. 음수/이상치 점검 (시간은 음수가 아니어야 정상)
duration_cols = ["purchase_to_approved","approved_to_carrier","carrier_to_customer","purchase_to_customer"]
neg_mask = (df_deliv[duration_cols] < -EPS).any(axis=1)
if neg_mask.any():
    print("\n[경고] 음수 시간 발견 (샘플 5개):\n", df_deliv.loc[neg_mask, ["order_id"]+duration_cols].head())
    # 필요 시 제외
    # df_deliv = df_deliv.loc[~neg_mask].copy()

# 1-7. 비현실적 장기 값(상위 0.5% 이상치) 확인
hi_thresh = df_deliv["purchase_to_customer"].quantile(0.995)
print(f"\n[알림] purchase_to_customer 상위 0.5% 컷오프: {hi_thresh:.2f}일 (참고용)")

# 'purchase_to_customer' 값이 26.73일보다 큰 이상치 데이터 필터링 후 제거
df_deliv_cleaned = df_deliv[df_deliv['purchase_to_customer'] <= 26.73]

# 이상치 제거 후 데이터 크기 확인
print(f"이상치 제거 전 행 수: {len(df_deliv):,}")
print(f"이상치 제거 후 행 수: {len(df_deliv_cleaned):,}")

# df_order 이상치 탐지 시각화

# 1️⃣ 히스토그램 시각화
df_order[time_cols].hist(bins=50, figsize=(12,6))
plt.suptitle("배송 시간 차이 히스토그램")
plt.show()

# 2️⃣ 극단치 비율 계산
print("=== Z-score 기준 이상치 비율 (|Z|>3) ===")
for col in time_cols:
    z = zscore(df_order[col].dropna())
    outlier_ratio = (abs(z) > 3).mean() * 100
    print(f"{col}: {outlier_ratio:.2f}%")

# 3️⃣ IQR 기반 이상치 비율 계산
print("\n=== IQR 기준 이상치 비율 ===")
for col in time_cols:
    data = df_order[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    iqr_outlier_ratio = ((data < lower_bound) | (data > upper_bound)).mean() * 100
    print(f"{col}: {iqr_outlier_ratio:.2f}%")
# --- 이상치 플래그 추가 ---
df_order["is_outlier"] = (df_order[time_cols] < 0).any(axis=1)

# 이상치 개수 확인
print("이상치 건수:", df_order["is_outlier"].sum())

# 이상치 제외한 데이터프레임 생성
df_order_clean = df_order[~df_order["is_outlier"]].copy()

print("정제 후 데이터 크기:", df_order_clean.shape)

# df_order_clean
# 이상치 제거 후 배송 시간 분포 확인
df_order_clean[time_cols].hist(bins=50, figsize=(12,6))
plt.suptitle("이상치 제거 후 배송 시간 분포")
plt.show()

# 이상치 확인 후 도메인 규칙 기반 제거
df_order_clean = df_order_clean[df_order_clean['approved_to_carrier'] >= 0]
df_order_clean = df_order_clean[df_order_clean['carrier_to_customer'] >= 0]

# 기초 통계 확인
df_order_clean[time_cols].describe()

df_order_clean.describe()
# df_order_clean = df_deliverd_clean

# df_order_payments: 결측X, 이상치 탐지

# 1. payment_type 분포 확인
plt.figure(figsize=(6,4))
sns.countplot(data=o_df_order_payments, x='payment_type', order=o_df_order_payments['payment_type'].value_counts().index)
plt.title("결제 수단 분포")
plt.xticks(rotation=30)
plt.show()

print("\n[결제 수단 비율]")
print(o_df_order_payments['payment_type'].value_counts(normalize=True).round(3))

# 2. 할부 개월 수 분포
plt.figure(figsize=(8,4))
sns.histplot(o_df_order_payments['payment_installments'], bins=30, kde=False)
plt.title("할부 개월 수 분포")
plt.xlabel("할부 개월 수")
plt.ylabel("빈도수")
plt.show()

print("\n[할부 개월 수 통계]")
print(o_df_order_payments['payment_installments'].describe())

# 3. 결제 금액 분포 (payment_value)
plt.figure(figsize=(8,4))
sns.boxplot(x=o_df_order_payments['payment_value'])
plt.title("결제 금액(Boxplot)")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(o_df_order_payments['payment_value'], bins=100, kde=True)
plt.title("결제 금액 분포 (히스토그램)")
plt.xlim(0, 1000)  # 고액 결제는 따로 확인하기 위해 일단 1000 이하만 시각화
plt.show()

print("\n[결제 금액 통계]")
print(o_df_order_payments['payment_value'].describe())

# 4. 이상치 건수 확인 (IQR 방식)
Q1 = o_df_order_payments['payment_value'].quantile(0.25)
Q3 = o_df_order_payments['payment_value'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (o_df_order_payments['payment_value'] < Q1 - 1.5*IQR) | (o_df_order_payments['payment_value'] > Q3 + 1.5*IQR)

print(f"\n[결제 금액 이상치 개수] {outlier_mask.sum()} / {len(o_df_order_payments)} ({outlier_mask.mean()*100:.2f}%)")

''' 이상치 간주
payment_type: 0, 음수, not_defined 제거
payment_installments (할부 개월 수): 0(일시불)/ 음수, 24개월 초과 제거
payment_value: 0(일시불), / 음수, Q1, Q3 기반 IQR로 극단치 검출 -> Winsorization (상한 절단)

평균(Mean) = 154.1 → 극단값(고액 결제)의 영향으로 평균이 중앙값보다 큼
최댓값 = 13,664.08 → 단 1건 정도의 초고액 결제 (전체 분포와 매우 동떨어짐)
IQR 이상치 비율 ≈ 7.7% (7,981건) → 전체 결제의 약 8%가 극단값
'''

df_order_payments = o_df_order_payments
df_order_payments.head()
'''
payments 이상치 라벨링 처리 -> 라벨링 df 생성: df_label_payment
installments = 0 → "일시불" 카테고리로 변환.
payment_value = 0 → "0원 결제" (ex. 쿠폰, 무료배송, 취소된 거래 등)으로 별도 라벨링.
'''
# 결제데이터 복사
df_lavel_payments = df_order_payments.copy()

# 일시불 라벨링
df_lavel_payments['installment_label'] = df_lavel_payments['payment_installments'].apply(
    lambda x: '일시불' if x == 0 else '할부'
)

# 결제금액 라벨링
df_lavel_payments['payment_label'] = df_lavel_payments['payment_value'].apply(
    lambda x: '0원결제' if x == 0 else '유료결제'
)

# 분포 확인
print(df_lavel_payments['installment_label'].value_counts())
print(df_lavel_payments['payment_label'].value_counts())

# 이상치로 보이는 데이터 일부 확인
print(df_lavel_payments[df_lavel_payments['payment_value'] == 0].head(10))
df_customers.head()
# MERGE
# join_order_c = df_order_clean + df_customers + df_payments + df_order_items
'''
MERGE
customer 데이터 탐색: 이상치 처리 안함, 데이터 손실 최소화
customer states 컬럼: SP(상파울루 주), RJ (리우데자네이루 주)
'''
# df_order.info() #77694, 컬럼 12개
# df_customers.info() #99441 컬럼 5개

# 1. 주문 + 고객 정보 데이터 조인 (order_id 기준)
join_order_c= df_order_clean.merge(
    df_customers,
    on='customer_id',
    how='left'   # 주문은 반드시 유지, 고객 정보가 없으면 NaN
)

print(f"Merge 후 레코드 수: {len(join_order_c)}")
print(f"원본 df_order 레코드 수: {len(df_order)}")
print("고유 order_id 개수:", join_order_c['order_id'].nunique())
print("전체 order_id 대비 중복 비율:", 1 - join_order_c['order_id'].nunique() / len(join_order_c))

print("customer_city 결측치 개수:", join_order_c['customer_city'].isnull().sum())
print("customer_city 결측치 비율:", join_order_c['customer_city'].isnull().mean())

print(join_order_c.dtypes)

join_order_c['purchase_to_approved'].head()
print(join_order_c.isnull().sum())
# df_join_order_cp

# 3. 1번 df + df_order_payments_sum 병합
df_join_order_cp= join_order_c.merge(
    df_order_payments,
    on='order_id',
    how='left'
)

# df_join_order_cp.info()
# print(f"Merge 후 레코드 수: {len(df_join_order_cp)}")
# print(f"원본 df_order 레코드 수: {len(df_join_order_cp)}")
# print(df_join_order_cp.isnull().sum())

# 'payment_type' 컬럼에 결측치가 있는 행만 필터링
nan_rows = df_join_order_cp[df_join_order_cp['payment_type'].isnull()]

print("=== 결측치가 있는 행 ===")
print(nan_rows)

# 결측치 개수 다시 확인
print("\n=== 결측치 개수 ===")
print(nan_rows.isnull().sum())

# 'payment_type' 컬럼에 결측치가 있는 행 제거
df_join_order_cp_cleaned = df_join_order_cp.dropna(subset=['payment_type'])

# 삭제 후 데이터프레임의 행 수 확인
print(f"결측치 제거 전 행 수: {len(df_join_order_cp):,}")
print(f"결측치 제거 후 행 수: {len(df_join_order_cp_cleaned):,}")

# 결측치가 제거되었는지 다시 확인
print("\n=== 결측치 제거 후 확인 ===")
print(df_join_order_cp_cleaned.isnull().sum())
# MERGE 3단계: 2번 결과에 아이템 정보 병합
df_join_ocpi = df_join_order_cp.merge(
    df_order_items, 
    on='order_id', 
    how='left'
)

df_join_ocpi.info()
print(f"Merge 후 레코드 수: {len(df_join_ocpi)}")
print(df_join_ocpi.isnull().sum())
'''
과제 1: 고객 세분화 및 RFM 분석
브라질 지역별 고객들의 구매 패턴을 분석하여 RFM(Recency, Frequency, Monetary) 모델을 구축하고,
고객을 세분화하여 각 세그먼트의 특성과 비즈니스 전략을 제시
'''
# ===============================
# 1. RFM 분석용 데이터 준비
# ===============================
# 분석 기준일 (데이터에서 가장 마지막 주문일 + 1일)
analysis_date = df_join_order_cp['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

# 고객별 RFM 집계
rfm = df_join_order_cp.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,  # Recency
    'order_id': 'nunique',                                                # Frequency (고객별 주문 횟수)
    'payment_value': 'sum'                                               # Monetary (총 결제 금액)
}).reset_index()

rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

# ===============================
# 2. RFM 점수화 (1~5등급)
# ===============================
# Recency: 최근일수 낮을수록 좋은 고객 → 낮으면 높은 점수
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])

# Frequency, Monetary: 값이 높을수록 좋은 고객 → 높으면 높은 점수
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

# RFM 조합 점수
rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

# ===============================
# 3. 고객 세그먼트 분류 (예시)
# ===============================
def segment_customer(row):
    if row['R_score'] in ['4','5'] and row['F_score'] in ['4','5']:
        return '우수 고객 (VIP)'
    elif row['R_score'] in ['3','4','5'] and row['F_score'] in ['1','2']:
        return '잠재 충성 고객'
    elif row['R_score'] in ['1','2'] and row['F_score'] in ['4','5']:
        return '이탈 위험 고객'
    elif row['R_score'] in ['1','2'] and row['F_score'] in ['1','2']:
        return '이탈 고객'
    else:
        return '일반 고객'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# ===============================
# 4. 지역별 RFM 분석 확장
# ===============================

customer_region = df_join_order_cp[['customer_id', 'customer_city']].drop_duplicates(subset=['customer_id'])

rfm_region = rfm.merge(customer_region, on='customer_id', how='left')

# 지역별 평균 RFM 값
region_summary = rfm_region.groupby('customer_city')[['Recency','Frequency','Monetary']].mean().round(1)

print("=== 지역별 평균 RFM ===")
print(region_summary.head())

# 1. customer_state 컬럼을 포함하여 조인
customer_region = df_join_order_cp[['customer_id', 'customer_city', 'customer_state']].drop_duplicates(subset=['customer_id'])

# 2. RFM 데이터에 지역 정보 병합
rfm_region = rfm.merge(customer_region, on='customer_id', how='left')

# 3. 주(state)별 평균 RFM 값
state_summary = rfm_region.groupby('customer_state')[['Recency','Frequency','Monetary']].mean().round(1)
print("=== 주(State)별 평균 RFM ===")
print(state_summary.head())

# 1. customer_state 컬럼을 포함하여 조인
customer_region = df_join_order_cp[['customer_id', 'customer_city', 'customer_state']].drop_duplicates(subset=['customer_id'])

# 2. RFM 데이터에 지역 정보 병합
rfm_region = rfm.merge(customer_region, on='customer_id', how='left')

# 3. 주(state)별 평균 RFM 값
state_summary = rfm_region.groupby('customer_state')[['Recency','Frequency','Monetary']].mean().round(1)
print("=== 주(State)별 평균 RFM ===")
print(state_summary.head())

# 4. 도시(city)별 평균 RFM 값
city_summary = rfm_region.groupby('customer_city')[['Recency','Frequency','Monetary']].mean().round(1)
print("\n=== 도시(City)별 평균 RFM ===")
print(city_summary.head())
'''
지역별(state, city별) 고객을 세분화하여 세그먼트별 특징
고객등급 별 구매금액
customer_stats['평균장바구니크기'] = customer_stats['총구매량'] / customer_stats['Frequency']
customer_stats['거래당상품종류'] = customer_stats['상품종류수'] / customer_stats['Frequency']
구매 기간 및 주기 계산 (추가하기)
'''
