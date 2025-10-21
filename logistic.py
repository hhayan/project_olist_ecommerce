import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os
import folium
from folium import Map
from folium.plugins import HeatMap
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import warnings

from RFM import df_sellers, df_order_reviews, merge_coigp, df_product_category_name_translation, df_products, df_sellers

# 전체 지연구간 확인
# 전체 구간 (구매 ~ 고객 수령) 계산
merge_coigp["purchase_to_customer"] = (
    merge_coigp["order_delivered_customer_date"] - merge_coigp["order_purchase_timestamp"]
).dt.total_seconds() / 86400   # 일 단위

# 단계별 평균 다시 계산
step_cols = ["purchase_to_approved", "approved_to_carrier", "carrier_to_customer", "purchase_to_customer"]
step_mean = merge_coigp[step_cols].mean().round(2)
print("📌 단계별 평균 배송시간(일 단위)")
print(step_mean)

# --- 3) ETA 대비 지연일 계산 ---
# ETA는 날짜 단위 비교 권장 → floor("d") 처리
merge_coigp["delay_days"] = (
    merge_coigp["order_delivered_customer_date"].dt.floor("d")
    - merge_coigp["order_estimated_delivery_date"].dt.floor("d")
).dt.days

# 정시/지연 여부
merge_coigp["on_time"] = merge_coigp["delay_days"] <= 0

# KPI 요약
total = len(merge_coigp)
on_time = merge_coigp["on_time"].sum()
late = total - on_time
avg_delay = merge_coigp.loc[merge_coigp["delay_days"] > 0, "delay_days"].mean()

kpi = {
    "총 배송건수": total,
    "정시배송율": round(on_time / total * 100, 2),
    "지연배송율": round(late / total * 100, 2),
    "평균 지연일(지연건만)": round(avg_delay, 2)
}

# --- 4) 지연 구간 분포 ---
bins = [-np.inf, 0, 3, 7, 14, 30, np.inf]
labels = ["정시/조기", "1-3일 지연", "3-7일 지연", "7-14일 지연", "14-30일 지연", "30일 초과"]

delay_dist = pd.cut(merge_coigp["delay_days"], bins=bins, labels=labels).value_counts(normalize=True).sort_index()
delay_dist = (delay_dist * 100).round(2)

# 지역별 배송 성과
# 전체 구간 (구매 ~ 고객 수령) 계산
merge_coigp["purchase_to_customer"] = (
    merge_coigp["order_delivered_customer_date"] - merge_coigp["order_purchase_timestamp"]
).dt.total_seconds() / 86400   # 일 단위

# ETA 대비 지연일 계산
merge_coigp["delay_days"] = (
    merge_coigp["order_delivered_customer_date"].dt.floor("d")
    - merge_coigp["order_estimated_delivery_date"].dt.floor("d")
).dt.days

# 정시배송 여부
merge_coigp["on_time"] = merge_coigp["delay_days"] <= 0

# 주(state)별 집계
state_perf = (
    merge_coigp.groupby("customer_state")
      .agg(
          주문수=("order_id", "count"),
          평균_전체배송시간=("purchase_to_customer", "mean"),
          정시배송율=("on_time", "mean"),
          평균_지연일=("delay_days", lambda x: x[x > 0].mean())
      )
      .round(2)
)

# 정시배송율을 %로 변환
state_perf["정시배송율"] = (state_perf["정시배송율"] * 100).round(2)
print("📌 주(state)별 배송 성과")
print(state_perf)

# top 5주, 하위 5주
# 상위 5개 주 (정시배송율 내림차순, 평균 전체배송시간 오름차순)
top5_states = state_perf.sort_values(
    by=["정시배송율", "평균_전체배송시간"],
    ascending=[False, True]
).head(5)

# 하위 5개 주 (정시배송율 오름차순, 평균 전체배송시간 내림차순)
bottom5_states = state_perf.sort_values(
    by=["정시배송율", "평균_전체배송시간"],
    ascending=[True, False]
).head(5)

# df_order_reives 전처리: 최신 리뷰만 (주문당 1개), 리뷰 없는 주문: 819(약 0.7% 수준) 남김
# merge_coigp + df_order_reviews => merge_coigpr

df_order_reviews['review_creation_date'] = pd.to_datetime(df_order_reviews['review_creation_date'])
df_order_reviews = (
    df_order_reviews
        .sort_values(['order_id', 'review_creation_date'])
        .drop_duplicates(subset=['order_id'], keep='last')
)
df_coigpr = merge_coigp.merge(df_order_reviews, on='order_id', how='left')

#   - 리뷰 있는 주문: 107,762개 / 리뷰 없는 주문: 819개 / 리뷰 커버율: 99.25%
print(f"\n리뷰 점수 분포:")
print(df_coigpr['review_score'].value_counts().sort_index())

# merge_coigpr + + df_products = merge_coigprp
df_coigprp = df_coigpr.merge(
    df_products,
    on='product_id',
    how='left'
)
# - 제품 정보 있음: 107,055개 / 제품 정보 없음: 1,526개 / 매칭률: 98.59%

# df_coigprp + df_product_category_name_translation = df_logistic
df_logistic = df_coigprp.merge(
    df_product_category_name_translation,
    on='product_category_name',
    how='left'
)
#  - 영문 카테고리 있음: 107,033개 / 영문 카테고리 없음: 1,548개

# df_logistic + df_sellers 
df_logistic_final = df_logistic.merge(
    df_sellers,
    on='seller_id',
    how='left'
)





