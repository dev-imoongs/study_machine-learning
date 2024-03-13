#!/usr/bin/env python
# coding: utf-8

# ### 회귀(Regression)
# - 데이터 값이 평균과 같은 일정한 값으로 돌아가려는 경향을 이용한 통계학 기법이다.
# - 여러 개의 독립 변수(예측 변수)와 한 개의 종속 변수 간의 상관관계를 모델링하는 기법을 통칭한다.
# - feature와 target 데이터 기반으로 학습하여 최적의 회귀계수를 찾는 것이 회귀의 목적이다.
# <img src="./images/regression01.png" width="500" style="margin-top:20px; margin-left: 0">
# 
# - 회귀 계수가 선형일 경우 선형 회귀로 구분되고, 비선형일 경우 비선형 회귀로 구분된다.
# - 독립 변수(예측 변수)가 한 개인지, 여러 개인지에 따라 단일 회귀, 다중 회귀로 나뉜다.
# - 여러 회귀 중 선형 회귀가 가장 많이 사용되며, 선형 회귀는 잔차를 최소화하는 직선을 최적화하는 방식이다.
# - 전체 데이터의 잔차(오류 값) 합이 최소가 되는 모델을 만들어야 하며, 이는 오류 값 합이 최소가 될 수 있는 회귀 계수를 찾는다는 것이다.
# <img src="./images/regression02.png" width="400" style="margin-top:20px; margin-left: 0; margin-bottom: 20px"> 
# 
# - 분류와 회귀의 차이는, 분류는 카테고리와 같은 범위 값(이산값)이고, 회귀는 연속형 값(연속값)이다.
# ##### 📌 이산값과 연속값
# <img src="./images/regression03.png" width="600" style="margin-top:20px; margin-left: 0; margin-bottom: 20px">

# ### 단순 선형 회귀 모델(Simple linear regression model)
# ##### 주택 크기에 따른 주택 가격을 예측한다고 가정한다.
# - 100개의 주택 데이터를 통해 주택 크기에 대한 주택 가격의 관계를 알아보고자 한다.
# 1. 특정 기간의 주택 크기를 독립 변수(예측 변수) x로 설정
# 2. 특정 기간의 주택 가격을 종속 변수 y로 설정
# <img src="./images/regression04.png" width="100" style="margin:30px; margin-top:5px; margin-left: 0">
# - 주택 경향을 알아보기 위해 아래와 같은 1차 방정식 선언하면 다음과 같다.
# <img src="./images/regression05.png" width="250" style="margin:30px; margin-top:0; margin-left: -20px">
# - 특정 기간을 지난 미래의 주택가격은 증가하거나 감소할 수 있기 때문에 이를 반영하기 위한 항을 추가한다.
# - 이 때 ε항을 랜덤 오차(random error)라고 하며, 오차가 가장 작은 일차 함수식을 구하는 것이 회귀의 목표이다.
# <img src="./images/regression06.png" width="250" style="margin:30px; margin-top:-2px; margin-left: -15px">
# - 이렇게 얻은 일차 함수식을 회귀선(regression line) 또는 단순 선형 회귀 모델(simple linear regression model)이라고 한다.
# <img src="./images/regression07.png" width="450" style="margin:30px; margin-top:-2px; margin-left: -15px">
# 
# ##### 회귀 계수 구하기 (출처: 존이)
# 1. 회귀계수인 β0(절편)과 β1(기울기)를 구하기 위해서는 <strong>기대값과 공분산</strong>을 이용할 수 있다.
# - 오차가 0에 가까울 수록 회귀선이 주어진 데이터를 잘 표현한다고 말할 수 있기 때문에, 오차항의 합은 0이라 가정한다.
# - 데이터의 개수가 2개일 경우 두 점을 연결하면 끝이기 때문에, 3개 이상으로 가정한다.  
# > 📌기대값(expected value)이란, 어떤 확률 과정을 무한히 반복했을 때, 얻을 수 있는 값의 평균으로서 기대할 수 있는 값이다.
# > <img src="./images/expected_value.png" width="150" style="margin:30px; margin-top:-2px; margin-left: -5px">  
# > 예시) 카지노에서 이길 확률은 99%이고. 이기면 100원을 받고, 지면 100,000원을 잃는다.<br>
# 확률 변수 X는 게임에서 얻는 돈의 양으로 정의한다면, X의 기대값 E[X]는 무엇일까?<br>
# > <strong>E[X] = 100 * 0.99 - 100,000 * 0.01 = -901</strong><br>
# > 한 게임당 얻는 돈의 기대값은 -901원이며, 게임을 계속 진행하게 되면 결과적으로 돈을 잃게 된다.<br><br>
# > 📌기대값의 성질<br>
# > 1. 상수 c의 기대값은 상수가 된다.<br>
# > <strong>E[c] = c</strong><br>
# > 2. <strong>E[cX] = cE[X]</strong>가 된다.<br>
# > - 안쪽에 있는 상수 c를 바깥쪽으로 꺼낼 수 있다.<br>
# > 3. <strong>E[cX + d]에서 c와 d가 상수일 때, E[cX + d] = cE[X] + d</strong>가 된다.<br>
# > - (1), (2) 성질을 결합한 것이다.<br>
# 
# - 식 양변에 기대값을 적용하면 아래와 같다.
# <img src="./images/regression08.png" width="250" style="margin:30px; margin-top:-2px; margin-left: -15px">
# 
# - x(독립 변수, 예측 변수)와 y(종속 변수)의 공분산을 구한다.
# <img src="./images/regression09.png" width="450" style="margin:30px; margin-top:-2px; margin-left: -15px">
# 
# - 위의 식은 모집단에 대한 자료를 바탕으로 구한 회귀 계수이며, 실제 모집단에 대한 모든 자료를 계산하여 선형회귀식을 구하는 것은 불가능에 가깝다.
# - 따라서 n개의 표본을 추출하여, 추출한 n개의 표본에 대한 선형회귀식을 얻어야 한다.  
# 📌모집단이란, 통계적인 관찰의 대상이 되는 집단 전체를 의미한다.
# - 표본으로 회귀 계수를 구할 때에는 표본공분산, 표본분산을 사용하며, 새로운 식은 아래와 같다.
# <img src="./images/regression10.png" width="280" style="margin:30px; margin-top:-2px; margin-left: -15px">
# 
# - 표본을 통해 얻은 선형회귀선은 다음과 같다.
# <img src="./images/regression11.png" width="210" style="margin:30px; margin-top:-2px; margin-left: -15px">
# <img src="./images/regression12.png" width="260" style="margin:30px; margin-top:-2px; margin-left: -20px">  
# 
# - 이렇게 표본을 통해 얻은 회귀선을 적합 회귀 선[모델] (fitted regression line[model], estimated regression line[model])이라고 하며,  
# 전체 모집단의 정확한 경향성을 보여주는 참회귀선(true regression line)은 실제로 구할 수 없다.
# <img src="./images/regression13.png" width="330" style="margin:30px; margin-top:-2px; margin-left: -20px">
# 
# - 적합회귀선과 참회귀선은 거의 동일한 모습을 보이며, 적합회귀선에서 실제 자료까지의 오차 거리를 잔차(e)라고 하며,  
# 참회귀선에서 실제 자료까지의 오차 거리를 오차항(ε)이라고 한다.
# <img src="./images/regression14.png" width="330" style="margin:30px; margin-top:-2px; margin-left: -20px">
# 
# - 오차항과 잔차는 다음과 같다.
# <img src="./images/regression15.png" width="700" style="margin:30px; margin-top:-2px; margin-left: -20px">
# 
# 2. 회귀계수인 β0(절편)과 β1(기울기)를 구하기 위해서는 <strong>최소제곱법</strong>을 이용할 수 있다.
# - 모든 잔차의 합은 0이며, 이는 모든 편차의 합이 0인것과 같다. 
# - 분산을 구할 때 편차를 제곱하여 구하는 것과 최소제곱법도 잔차를 제곱하여 사용한다.  
# 📌최소제곱법이란, 어떤 방정식의 해를 근사적으로 구하는 방법으로서, 근사 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법이다.
# ##### 좌표평면에 다음과 같은 데이터가 있다고 가정한다. (2, 4), (2, 6), (3, 6), (3, 8)
# <img src="./images/regression16.png" width="300" style="margin:30px; margin-top:0px; margin-left: -20px">
# 
# - 이 때 4개의 점을 2개의 직선으로 나타내고자 한다면 아래와 같다.
# <div style="display: flex;">
#     <div style="margin-left: 0">
#         <img src="./images/regression17.png" width="340">
#     </div>
#     <div>
#         <img src="./images/regression18.png" width="350">
#     </div>
# </div>  
# 
# - 식이 한 개 나와야 하는데, 식이 2개가 나오면 이는 주어진 자료를 잘 표현하는 식이 될 수 없다. 따라서 잔차를 제곱하는 방식을 사용해야 한다.
# - 잔차를 제곱하면 아래와 같다.
# <img src="./images/regression19.png" width="120" style="margin:30px; margin-top:0px; margin-left: -20px">
# - 표본으로 추출한 n개의 잔차 제곱의 총 합(SSE, Sum of Sqaures of the Error, RSS, 비용 함수, 손실 함수)은 아래와 같다.
# <img src="./images/regression20.png" width="200" style="margin:30px; margin-top:0px; margin-left: -20px">
# - 표본을 통해 얻은 선형회귀선은 다음과 같다.
# <img src="./images/regression11.png" width="210" style="margin:30px; margin-top:-2px; margin-left: -15px">
# - SSE(RSS, Reidual Sum of Squares)에 대입하면 아래와 같다.
# <img src="./images/regression21.png" width="370" style="margin:30px; margin-top:-2px; margin-left: -15px">
# - 잔차 제곱의 합이 최소가 되도록 해야 하고, 미지수는 절편과 기울기이므로 각 절편과 기울기를 기준으로 편미분을 진행한다. 
# - 회귀 알고리즘은 데이터를 계속 학습하면서 비용 함수가 반환하는 값(즉, 오류 값)을 지속해서 감소시키고 최종적으로는 더 이상 감소하지 않는 최소의 오류 값을 구하는 것이다.
# <img src="./images/regression22.png" width="280" style="margin:30px; margin-top:5px; margin-left: -10px">
# <img src="./images/regression23.png" width="300" style="margin:30px; margin-top:-2px; margin-left: 0px">
# 
# > 📌편미분(partial derivative)이란, 다변수 함수의 특정 변수를 제외한 나머지 변수를 상수로 간주하여 미분하는 것을 의미한다.  
# > - 독립변수가 2개 이상인 함수의 미분시, 3차원 이상을 형성하게 되고, 이로 인해 변화를 한 번에 확인하기 쉽지 않기 때문에, 각 독립변수의 변화를 관찰할 때에는 다른 나머지 독립변수를 상수로 취급하여 미분한다.
# <img src="./images/partial_derivative.png" width="300" style="margin:30px; margin-top:20px; margin-left: -15px">
# 
# - 1식과 2식을 연립하여 절편과 기울기를 구한다.
# <img src="./images/regression24.png" width="250" style="margin:30px; margin-top:0px; margin-left: -20px">
# - 연립하여 구한 절편과 기울기를 일차식에 대입한다.
# <img src="./images/regression25.png" width="450" style="margin:30px; margin-top:0px; margin-left: -20px">
# - 좌표 평면에 그려진 4개의 데이터, (2, 4), (2, 6), (3, 6), (3, 8)을 대입하면 아래와 같이 적합회귀선이 그려진다.
# <img src="./images/regression26.png" width="450" style="margin:30px; margin-top:0px; margin-left: -20px">
# 
# ---
# 
# ##### 적합 회귀선
# <img src="./images/regression27.png" width="450" style="margin:30px; margin-top:20px; margin-left: -20px">

# ### 경사 하강법(Gradient descent)
# - 함수의 기울기(경사)를 구하고 경사의 반대 방향으로 계속 이동시켜 극값에 이를 때까지 반복시키는 방법이다.
# - 실제 우리가 마주치는 함수들은 간단한 함수가 아니라 복잡하고, 비선형적인 함수가 대부분이다.  
# 따라서 미분을 통하여 그 값을 계산하기 어려운 경우가 많다.
# - 즉, 손실함수의 최저점을 가지는 파라미터(weight)를 바로 찾기는 매우 어렵기 때문에  
# 손실 함수의 최저점으로 향하는 방향성을 반복해서 조금씩 찾아내는 것이 더 효율적이다. 
# - 함수 미분을 직접 구현하는 대신 경사 하강법을 이용하여 함수의 최소, 최댓값을 찾는다.
# - 현재 위치의 기울기가 음수라면 파라미터(weight)를 증가시키면 최솟값을 찾을 수 있고,   
# 기울기가 양수라면 파라미터(weight)를 감소시키면 최솟값을 찾을 수 있다.
# <img src="./images/gradient_descent.gif" style="margin:20px; margin-left: -40px">
# 
# - 해당 파라미터(weight)에서 학습률 * 기울기를 빼면 최솟값이 되는 장소를 찾을 수 있다.
# <img src="./images/gradient_descent01.png" width="450" style="margin:20px; margin-left: -20px">
# 
# - 학습률과 기울기 정보를 혼합하여 내가 나아갈 방향과 거리를 결정하기 때문에 학습률을 적절히 조정하는 것이 매우 중요하다.
# <img src="./images/gradient_descent02.png" width="750" style="margin:20px; margin-left: -20px">
