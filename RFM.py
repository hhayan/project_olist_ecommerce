import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import warnings

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
o_df_orders = pd.read_csv(os.path.join(folder_path, 'olist_orders_dataset.csv'), encoding='ISO-8859-1')

# 카피본 생성
df_customers = o_df_customers.copy()
df_geolocation = o_df_geolocation.copy()
df_order_items = o_df_order_items.copy()
df_order_payments = o_df_order_payments.copy()
df_order_reviews = o_df_order_reviews.copy()
df_products = o_df_products.copy()
df_sellers = o_df_sellers.copy()
df_product_category_name_translation = o_df_product_category_name_translation.copy()
df_order = o_df_orders.copy()

# EDA  df_order

# 1. 날짜/시간 관련 컬럼들을 datetime 타입으로 변환
# errors='coerce'는 변환 중 오류 발생 시 해당 값을 NaT(Not a Time)으로 처리
time_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in time_cols:
    df_order[col] = pd.to_datetime(df_order[col], errors='coerce')

# 날짜 차이 계산 (일 단위)
df_order['purchase_to_approved'] = (df_order['order_approved_at'] - df_order['order_purchase_timestamp']).dt.total_seconds() / 86400
df_order['approved_to_carrier'] = (df_order['order_delivered_carrier_date'] - df_order['order_approved_at']).dt.total_seconds() / 86400
df_order['carrier_to_customer'] = (df_order['order_delivered_customer_date'] - df_order['order_delivered_carrier_date']).dt.total_seconds() / 86400
df_order['purchase_to_estimated'] = (df_order['order_estimated_delivery_date'] - df_order['order_purchase_timestamp']).dt.total_seconds() / 86400

# df_orders 이상치 확인 및 삭제 처리, 컬럼 추가

# 1. approved → carrier 음수
df_order['time_approved_to_carrier'] = (
    pd.to_datetime(df_order['order_delivered_carrier_date']) - 
    pd.to_datetime(df_order['order_approved_at'])
).dt.total_seconds() / 86400

negative_2 = (df_order['time_approved_to_carrier'] < 0).sum()

# 2. carrier → customer 음수
df_order['time_carrier_to_customer'] = (
    pd.to_datetime(df_order['order_delivered_customer_date']) - 
    pd.to_datetime(df_order['order_delivered_carrier_date'])
).dt.total_seconds() / 86400

negative_3 = (df_order['time_carrier_to_customer'] < 0).sum()

# 3. 이상치 제거
df_order = df_order[
    (df_order['time_approved_to_carrier'] >= 0) &
    (df_order['time_carrier_to_customer'] >= 0)
]

'''
df_order_reviews 전처리
'''
# 목적: 배송 지연 ↔ 리뷰 점수, 텍스트 작성 여부 분석

# 1. 플래그 추가
df_order_reviews['has_title'] = df_order_reviews['review_comment_title'].notna()
df_order_reviews['has_comment'] = df_order_reviews['review_comment_message'].notna()

#### merge
# 배송 완료된 주문만 먼저 필터링
df_order_delivered = df_order[df_order['order_status'] == 'delivered']

# 1) customer + order (구매 고객만)
merge_co = df_customers.merge(
    df_order_delivered,
    on="customer_id",
    how="inner",
    validate="1:m"
)

# 2) + order_items
merge_coi = merge_co.merge(
    df_order_items,
    on="order_id",
    how="left",
    validate="1:m"
)

# 3) + geolocation
merge_coig = merge_coi.merge(
    df_geolocation.drop_duplicates(
        subset="geolocation_zip_code_prefix",
        keep='first'
    ),
    left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left",
    validate="m:1"
)

# 4) + payments
df_payments = df_order_payments.groupby('order_id')['payment_value'].sum().reset_index()

merge_coigp = merge_coig.merge(
    df_payments,
    on="order_id",
    how="left",
    validate="m:1"  
)



