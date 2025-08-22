# project_ecommers_data
Sesac LLM DA  1차 프로젝트 

# 데이터 탐색
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
----------------------------------------------------------------------
payments data: freight_value: 개별 아이템에 할당된 운송료

# 전처리
# df_produts
결측치 비율 낮음 삭제 처리
이상치: 음수 없음, 0 비율 낮음 삭제 처리

# df_order
결측치 비율 낮음 삭제 처리

# order_items
1) 데이터 탐색
product_id
seller_id
shipping_limit_date = 배송마감시간
price
freight_value = 운송비

2) 결측치, 이상치 탐지 및 처리
결측 없음, 이상치 탐지 후 모두 결측률이 낮아 삭제: 
가격(price) 및 운송료(freight_value)가 0이거나 음수인 경우, 너무 높은 값 확인
price = 0: 이상치 가능성 높음 → 제거
freight_value = 0: 일부 무료배송일 수 있음 → 제거 전 비율(결측률) 확인 -> 결측률 낮음 삭제

IsolationForest로 이상치 탐지 후 처리
고가 이상치 처리: Winsorization 1%와 99% 백분위수 값을 기준으로 이상치를 대체합니다.
price 컬럼의 경우, 6735.00과 같은 높은 가격은 매우 비싼 고가의 상품일 가능성이 있습니다. 
freight_value 컬럼의 경우, 409.68과 같은 높은 운송료는 매우 무거운 상품이나 국제 배송의 결과일 수 있습니다.

# merge
join_order_customer
join_order_payments
df_join_ocpi: o_df_customers, o_df_order_items, o_df_order_payments, o_df_products
merge_full: merge_product_cate + df_join_ocpi

(과제2 배송지연 확인용) inner_join
1. df_order + df_order_items = join_order_items
2. join_order_items + df_customers = jj_order_items_cu

# 특이사항
예상 배송일이랑 실제 배송일을 가지고 계산, 예상일보다 빨리 도착한 경우 음수가 나오는 걸로 설정


