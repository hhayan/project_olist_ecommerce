## project_ecommers_data
Sesac LLM DATA  1차 프로젝트 

## EDA
# 이상치
배송료(freight_value)가 0인 비율이 0.34%: 무료배송일 수 있으니 보존 -> 확인 필
order_item['price'],['fight_value]: 시각화
payment_sequential는 단일 주문에 대한 결제가 여러 번 이루어졌을 때 순서
평균값(Mean: 1.09)과 75% 지점(75%: 1)이 1이라는 것은 **대부분의 주문이 한 번의 결제(단일 결제)**로 이루어졌다는 것을 보여줍니다. 극단값 29 -> 바우처 결제, 정상

payment_value(거래금액): 최소값은 0이고 최대값은 약 13,664에 달하며, 대부분의 거래가 낮은 금액대에 분포
review_score: 높은 편

# 데이터 관계 분석
- 가격과 배송비의 관계: 가격대가 높은 상품이 배송비도 높은 경향이 있는지 분석
스피어만: 0.434  |   피어슨: 0.414   -->  중간정도
가격과 배송료는 어느 정도 연결되어 있지만, 1:1로 비례하지는 않음. 무게, 크기, 출고지/도착지 거리 같은 추가 요인들도 영향을 줍니다.
df_order_items (원본) → 이상치 포함 전체 데이터
df_order_items[df_order_items["is_outlier"] == 0] → 이상치 제거 데이터
# 추후 “전체 평균 배송비” vs “이상치 제외 평균 배송비” 분석 예정

olist_orders: order_status - shipped 상품이 판매자나 물류센터에서 발송되어 고객에게 전달되기 위한 준비가 완료
Order Items 데이터셋 분석: 각 주문(order_id) 내에서 구매된 상품(아이템)에 대한 정보

총 주문 가치 계산:
상품 가치: price 컬럼은 개별 상품의 가격을 나타냅니다. 총 상품 가치는 개별 가격($21.33)에 상품 수량(3)을 곱하여 계산합니다(21.33∗3=63.99).

운송료(freight): freight_value 컬럼은 해당 아이템에 할당된 운송료입니다.

중요: 주문에 상품이 여러 개 있는 경우, 총 운송료가 각 아이템에 분할되어 할당됩니다.

따라서 주문의 총 운송료를 구하려면, 각 아이템의 freight_value를 합산해야 합니다(15.10∗3=45.30).

총 주문 가치: 총 상품 가치와 총 운송료를 합산하여 계산합니다(63.99+45.30=109.29).

payment_value = sum(price) + sum(freight_value)
한 주문의 총 결제 금액(payment_value)은 해당 주문에 포함된 모든 상품들의 price와 freight_value를 각각 합산한 값과 일치해야 한다는 것을 의미
payments data: freight_value: 개별 아이템에 할당된 운송료
----------------------------------------------------------------------
# 데이터 전처리: 결측치, 이상치(0, 음수, IQR) (고유값, 중복 데이터, 상관관계X)
# df_order_reviews
리뷰 없는 결측치 50% 이상, 'no comment' 값 채움

# df_produts
결측치 비율 낮음 1.85% 삭제 처리
이상치: 음수 없음, 0 비율 낮음 삭제 처리

# df_order
결측치 비율 낮음 삭제 처리

# order_items
0 삭제, 이상치 데이터 분리
1) 데이터 탐색
product_id
seller_id
shipping_limit_date = 배송마감시간
price
freight_value = 운송비

# merge
join_order_customer
join_order_payments
df_join_ocpi: o_df_customers, o_df_order_items, o_df_order_payments, o_df_products
merge_full: merge_product_cate + df_join_ocpi

(과제2 배송지연 확인용) inner_join
1. df_order + df_order_items = join_order_items
2. join_order_items + df_customers = jj_order_items_cu
3. (seller_id) jj_order_items_cu + df_sellers = join_ois

# 특이사항
예상 배송일이랑 실제 배송일을 가지고 계산, 예상일보다 빨리 도착한 경우 음수가 나오는 걸로 설정


