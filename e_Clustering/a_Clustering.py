#!/usr/bin/env python
# coding: utf-8

# ### Clustering (군집화)
# - 모집단 또는 범주에 대한 사전 정보가 없는 경우 주어진 관측값들 사이의 거리(distance) 또는 유사성 을 이용하여 전체를 몇 개의 집단으로 그룹화함으로써 각 집단의 성격을 파악하고 데이터에 대한 이해를 돕고자 하는 분석법이다.
# - 데이터들을 별개의 군집으로 그룹화하는 것을 의미하며, 유사성이 높은 데이터들을 동일한 그룹으로 분류한다.
# - 비지도 학습에 속하는 알고리즘으로서 비슷한 샘플을 하나의 클러스터로 모으는 것을 목표로한다.
# <img src="./images/clustering01.png" width="500" style="margin-left: 0">
# 
# ##### 비계층적 군집화(Non-Hierarchical Clustering), 분할적 군집화(Partitioning Clustering)
# - 사전에 군집의 수를 정해준 뒤 개체를 분류하며, n개의 개체를 g개의 군집으로 나눌 수 있는 모든 방법을 점검해 최적화한 군집을 형성한다.
# - 초기 군집수를 결정하기 어렵고, 가중치와 거리정의가 어렵지만 자료의 크기에 제약이 없기 때문에 대용량 데이터 처리에 능하다.
# - 어떤 두 개의 군집이 하나로 합쳐진 적이 있다고 가정하여, 이 정보를 사용하는 측정법이며, 계층적 거리 측정법에 비해 계산량이 적어 효율적이다.
# - 대표적으로 K-평균 군집 분석(K-means Clustering, 중심 기반) 방법과 DBSCAN(밀도 기반)이 있다.
# <img src="./images/partitioning01.gif" width="500" style="margin-left:0">
# 
# ##### 계층적 군집화(Hierarchical Clustering), 합체 군집화(Agglomerative clustering)
# - n개의 군집으로 시작해 점차 군집의 개수를 줄여나가는 방법이기 때문에 군집의 수를 사전에 결정할 필요가 없다.
# - 최초에는 데이터 개수만큼 군집이 존재하지만 군집을 합치면서 최종적으로 하나의 군집만 남게 된다.
# - 중심연결, 단일연결, 완전연결, 평균연결, 와드연결 등을 사용하여 개체를 분류하기 때문에 비계층적 거리측정법에 비해 계산량이 많은 단점이 있다.
# - 대표적으로 scipy의 Hierarchical Clustering과 sklearn의 Agglomerative Clustering, Mean-Shift 방법이 있다.
# <img src="./images/hierarch01.gif" width="500" style="margin-left:0">
