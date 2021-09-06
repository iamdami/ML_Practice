# 표
표는 데이터들의 모임<br>
표 == 데이터 셋
<br>
<br>

### 행(row)
-개체(instance)<br>
-관측치(observed value)<br>
-기록(record)<br>
-사례(example)<br>
-경우(case)<br>

### 열(column)
-특성(feature)<br>
-속성(attribute)<br>
-변수(variable)<br>
-field<br>
<br>
<br>
### 변수(variable)
표에 대해서의 변수는 열을 의미!
<br>
<br>

# 독립변수와 종속변수
-독립변수 = 원인이 되는 열<br>
-종속변수 = 결과가 되는 열<br>
<br>
<br>
# 상관관계
한쪽 값이 바뀌었을 때 다른 쪽 값도 바뀜<br>
-> 두 개의 특성이 서로 관련있다고 추측 가능<br>
 이런 관계를 상관관계라고 함<br>
<img src="https://s3-ap-northeast-2.amazonaws.com/opentutorials-user-file/module/4916/12266.jpeg" alt="상관관계" width="680" height="470">
<br>
<br>
# 인과관계
각 열이 원인과 결과의 관계일 때 인과관계가 있다고 함
<br>
<br>
* 상관관계는 인과관계를 포함 <br>
<img src="https://s3-ap-northeast-2.amazonaws.com/opentutorials-user-file/module/4916/12267.jpeg" alt="인과관계" width="580" height="340">
<br>
-모든 인과관계는 상관관계지만 모든 상관관계가 인과관계는 아님<br>
<br>

# 양적 데이터와 범주형 데이터
- 양적 데이터(Quantitative data)<br>
-얼마나 큰지, 얼마나 많은지 어느 정도인지 의미하는 데이터<br>
-양적 데이터 == 숫자<br>

- 범주형 데이터(Categorical data)<br>
-이름이라는 표현 대신 범주<br>
<br>

# 머신러닝 분류
<img src="https://s3-ap-northeast-2.amazonaws.com/opentutorials-user-file/module/4916/12287.jpeg" alt="머신러닝 분류" width="580" height="340">
<br>

### 지도학습(supervised learning)
"기계를 가르친다"<br>
-데이터로 컴퓨터를 학습시켜 모델 만드는 방식<br>
<br>
지도학습을 하기 위해서는 우선 과거 데이터가 있어야 하고<br>
그 데이터를 독립변수(원인)과 종속변수(결과)로 분리해야 함<br>
<br>
독립변수와 종속변수의 관계를 컴퓨터에게 학습시키면<br>
컴퓨터는 그 관계를 설명할 수 있는 공식 만들어 냄<br>
<br>
이 공식을 머신러닝에서 "모델" 이라고 함<br>
좋은 모델이 되려면 데이터가 많을수록, 정확할수록 좋음<br>
<br>
<br>
지도학습은 크게 회귀와 분류로 나뉨
<br>
<br>

#### 회귀(Regression)
예측하고 싶은 종속변수가 '숫자'일 때 보통 회귀 방법 사용<br>
-종속변수가 양적 데이터라면 회귀 사용<br>
<br>

#### 분류(Classification)
추측하고 싶은 결과가 이름 혹은 '문자'일 때 분류 방법 사용<br>
-종속변수가 범주형 데이터라면 분류 사용<br>
<br>
<br>

### 비지도학습(unsupervised learning)
지도학습에 포함되지 않는 방법들<br>
-대체로 기계에게 데이터에 대한 통찰력 부여<br>
-데이터 성격 파악하거나 데이터를 잘 정리정돈 하는 것에 주로 사용됨<br>
<br>
<br>
비지도학습의 사례로 군집화와 연관규칙이 있음
<br>
<br>

#### 군집화(Clustering)
비슷한 것들을 찾아 그룹 만드는 것<br>
- 군집화 : 어떤 대상들을 구분해 그룹 만드는 것<br>
- 분류 : 어떤 대상이 어떤 그룹에 속하는지 판단하는 것<br>
<br>

#### 연관규칙학습 (Association rule learning)
서로 연관된 특징 찾아내는 것<br>
"장바구니 분석"<br>
<br>
<img src="https://s3-ap-northeast-2.amazonaws.com/opentutorials-user-file/module/4916/12344.jpeg" alt="군집화 연관규칙" width="580" height="340">
<br>
- 관측치(행)를 그룹핑 해주는 것 -> 군집화<br>
- 특성(열)을 그룹핑 해주는 것 -> 연관규칙<br>
<br>
<br>

# 지도학습과 비지도학습 차이점
- 비지도학습은 데이터들의 성격 파악이 목적 -> 데이터만 있으면 됨<br>
- 지도학습은 과거의 원인과 결과를 바탕으로 결과를 모르는 원인이 발생했을 때 그것이 어떤 결과를 초래할 것인지 추측하는 게 목적 -> 독립변수(원인), 종속변수(결과) 필요<br>
<br>
<br>

### 강화학습(reinforcement learning)
학습을 통해 능력 향상시킨다는 점에서 지도학습과 비슷하지만 <br>
강화학습은 어떻게 하는 것이 더 좋은 결과를 낼 수 있는지 더 좋은 답을 찾아가는 것<br>
-<b>더 좋은 보상을 받기 위해 수련하는 것</b><br>
"일단 해보는 것"<br>
지도학습이 배움을 통해 실력 키우는 것이라면,<br>
강화학습은 경험을 통해 실력 키워가는 것<br>
<br>
답을 찾아가는 과정을 많이 반복하면 더 좋은 답을 찾아낼 수 있다는 것이 기본 아이디어<br>
<br>
<br>
