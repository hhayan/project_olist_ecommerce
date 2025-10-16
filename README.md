## project_ecommers_data

분석 목표: 매출 증대 및 효과적인 투자

사용 데이터:
olist_customer
olist_geolocation
olist_orders
olist_order_items
olist_order_payments
olist_order_reviews
olist_products
olist_sellers

## EDA
1. 컬럼 이해
-product_items
order_id: 고객 1명이 주문한 ID
price: 상품 1개의 가격
freight_value: 상품 1개에 할당된 배송비
order_item_id: 주문 내 상품의 순서

2. 전처리 특이사항
- 결측치
olist_orders:

order_reviews:
(전처리 전)
	결측수	결측율(%)
review_comment_title	87656	88.34
review_comment_message	58247	58.70
(전처리 후): 제목있음, 내용있음 컬럼 추가
전체 리뷰: 99,224개
제목 있음: 11,568개 (11.7%)
코멘트 있음: 40,977개 (41.3%)

df_order_items 컬럼 추가
'item_total' id별 
'freight_total': id별 
'order_total' : id별, 'item_total' + 'freight_total'
'order_total' : 전체 총 매출

- 이상치: 원본 보존

3. EDA
- df_order_items 가격과 배송료 상관관계:
피어슨 상관계수: 0.414
스피어맨 상관계수: 0.434
거래비중
전체의 90% 이상: 저가 상품 + 적당한 배송비
전체의 5% 미만 : 고가 상품: 3,000~7,000 R$, 고액 배송비: 200~400 R$ (대형/무거운 상품)
특이 패턴(가능성 염두)
가격 0원 근처: 무료 증정/프로모션
배송비 0원: 무료 배송 이벤트
고가인데 저배송비: 고가 경량 상품 (전자제품 등)
저가인데 고배송비: 저가 대형 상품 (가구 등)

- df_order_payments
결제방법: credit_card 0.739 가장 높음 
할부개월: 대부분 고객은 일시불(0개월) ~ 1개월로 결제하고, 일부만 장기 할부(최대 24개월)를 사용
12개월이 넘어가는 장기 할부는 빈도가 매우 낮아 거의 보이지 않음
- 결제금액 분포:
결제금액: 중앙값 217, 최대값 13664 (달러)
전체 결제 건의 최소 75%가 매우 작은 금액대에 몰려있다
가장 큰 이상치는 약 14,000에 가까운 값으로, 일반적인 결제 금액과 엄청난 차이를 보임 

결제금액 이상치 보존:
고가 -> 실제로 고가 상품을 여러 개 구매한 고객일 수도 있음
이상치 비율 7.68% → 고액 결제 고객군 (VIP) 가능성이 큼.

결제금액 0 비율 0.01% → 사실상 데이터 오류 또는 무료 거래 (쿠폰/프로모션)일 수 있음.
payment_type
voucher        6
not_defined    3 <- 이상치 처리: 해당 데이터만 삭제

1. 기본 데이터 품질 점검
-결측치, 이상치 확인
가격(price), 배송비(freight_value), 리뷰(review_score) 등
0값, 비정상적으로 큰 값 탐지
-데이터 타입/분포 확인
날짜형 → 시간 추세 분석 가능 여부 확인
범주형 → 카테고리 불균형 확인
-새로운 파생 변수 생성

-데이터 병합

2. 주문 패턴 분석
주문 추세: 월별/연도별 주문 건수, 거래액 변화
주문 채널: 결제 방식 비율 (credit card, boleto, voucher 등)
구매 단위: 고객 1인당 평균 주문 수, 주문당 평균 상품 수

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

order_items["total_order_value"]컬럼, 총매출 컬럼 추가

df_order
-EDA: 주문시간대 가장 높은 시간: 11시, 1시~4시
주문상태-배달완료가 가장 많음

-전처리
결측치 없음
[배송시간] 이상치 없음
purchase → approved는 절대 음수가 될 수 없음 → 데이터 오류이므로 이상치 처리 
approved → carrier도 음수는 말이 안 됨 → 이상치 처리.
carrier → customer 역시 음수는 불가능 → 이상치 처리.
purchase → estimated에서 음수는 “예상보다 빨리 도착” 상황 → 이상치 아님.

EDA

reiew 전처리
----------------------------------------------------------------------
# 데이터 전처리: 결측치, 이상치(0, 음수, IQR) (고유값, 중복 데이터, 상관관계X)
# df_order_reviews
리뷰 없는 결측치 50% 이상, 'no comment' 값 채움

# df_produts
결측치 비율 낮음 1.85% 삭제 처리
이상치: 음수 없음, 0 비율 낮음 삭제 처리

# df_order
결측치 없음

# order_items
0 삭제, 이상치 데이터 분리
1) 데이터 탐색
product_id
seller_id
shipping_limit_date = 배송마감시간
price
freight_value = 운송비

on_time 컬럼 추가
보통 delay_days = 실제 배송일 - 예상 배송일 로 계산합니다.
on_time = (delay_days <= 0) → 정시 배송 여부 (True/False)
즉,on_time=True → 예정일보다 같거나 빨리 도착 (정시/조기 배송)

배송-리뷰 상관관계
지연이 심해질수록 고객이 낮은 점수를 준다”


