#IMPORT

import numpy as np

from sklearn.metrics import mean_squared_error

import pandas as pd

from matplotlib import pyplot as plt

import sklearn

from sklearn.preprocessing import *

import seaborn as sns

#데이터

data=pd.read_excel('/20201103_B3_TEST.xlsx')

data.shape


##클리닝

data2=data.drop(['_NAME_','ORIGIN_BIZPL_CD'],axis=1)

#Null 처리

data2=data2.fillna(0)

#len(data2.columns[data2.sum(axis=0) <1000])

#음수가 0인것 위치

# 여기서 non_zeros는 아래 함수에서 확인할 수 있다.

h,v =data2.shape

negitive = [ (i, j, data2.iloc[i, j]) for i in range(h) for j in range(v) if data2.iloc[i, j] < 0 ]

negitive_h = [negitive[0] for negitive in negitive]

negitive_v = [negitive[1] for negitive in negitive]

 

#음수가 0인것 0으로 

data2[data2<0]=0


#0값 체크후 데이터 크기 줄이기 ()
((data2[data2==0].count()/len(data2))>0.5).sum()
# .`sort_values(ascending=False)>0.9 
# .sort_values(by=0,ascending=False)


#null 50 이상 일단 삭제 
data3=data2.drop(data2.columns[(data2[data2==0].count()/len(data2))>0.5],axis=1)
# data3.shape




#item정규화후 uI b_i 보정
#정규화
min=data3[data3>0].min()
# min.sort_values(ascending=False)
denominator=data3.max()-min


normal=data3.copy()
for i in range(len(normal.columns)) :
    normal.iloc[:,i]=(normal.iloc[:,i]-min[i])/denominator[i]
normal




#정규화한것 u,b_i,b_u
user, item= normal.shape
u_i=[None]*item
u_u=[None]*user
u=normal[data3>0].sum().sum()/normal[data3>0].count().sum()
for i in range(item):
    temp=normal[data3.iloc[:,i]>0].iloc[:,i]
    u_i[i]=temp.mean()
for i in range(user):
    temp=normal.iloc[i,:][data3.iloc[i,:]>0]
    u_u[i]=temp.mean()
b_i=u-u_i
b_u=u-u_u





# 실제 R 행렬과 예측 행렬의 오차를 구하는 함수
def get_rmse(R, P, Q, non_zeros):
    error = 0

    full_pred_matrix = np.dot(P, Q.T)

    # 여기서 non_zeros는 아래 함수에서 확인할 수 있다.
    x_non_zero_ind = [non_zeros[0] for non_zeros in non_zeros]
    y_non_zero_ind = [non_zeros[1] for non_zeros in non_zeros]

    # 원 행렬 R에서 0이 아닌 값들만 추출한다.
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

    # 예측 행렬에서 원 행렬 R에서 0이 아닌 위치의 값들만 추출하여 저장한다.
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
 #   print("full_pred_matrix.shape",full_pred_matrix.shape)
  #  print("mse.shape:",mse.shape)
   # print("rmse.shape",rmse.shape)

    return rmse




def matrix_factorization1(R,R2, K, steps, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape

    np.random.seed(1)
    P = np.random.normal(scale=1.0/K, size=(num_users, K))
    Q = np.random.normal(scale=1.0/K, size=(num_items, K))
    # R>0인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장한다.
    non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R2[i, j] > 0 ]

    # SGD 기법으로 P, Q 매트릭스를 업데이트 함
    for step in range(steps):
        for i, j, r in non_zeros : 
            # 잔차 구함
            b_iu=u+b_u[i]+b_i[j]
            eij = r-b_iu-np.dot(P[i, :], Q[j, :].T)
            # Regulation을 반영한 SGD 업데이터 적용
            b_u[i]=b_u[i]+learning_rate*(eij-r_lambda*b_u[i])
            b_i[j]=b_i[j]+learning_rate*(eij-r_lambda*b_i[j])
            P[i, :] = P[i, :] + learning_rate*(eij * Q[j, :] - r_lambda*P[i, :])
            Q[j, :] = Q[j, :] + learning_rate*(eij * P[i, :] - r_lambda*Q[j, :])
        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0 :
            print("iter step",step, "rmse : " , rmse)

    return np.dot(P, Q.T)


# K=50, step=200
R_hat1=matrix_factorization1(np.array(normal),np.array(data3), 50, 500 )




#원데이터로

user2, item2= data3.shape
u_i2=[None]*item
u_u2=[None]*user
u2=data3[data3>0].sum().sum()/data3[data3>0].count().sum()
for i in range(item2):
    temp=data3[data3.iloc[:,i]>0].iloc[:,i]
    u_i2[i]=temp.mean()
for i in range(user2):
    temp=data3.iloc[i,:][data3.iloc[i,:]>0]
    u_u2[i]=temp.mean()
b_i2=u2-u_i2
b_u2=u2-u_u2





def matrix_factorization2(R,R2, K, steps, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape

    np.random.seed(1)
    P = np.random.normal(scale=1.0/K, size=(num_users, K))
    Q = np.random.normal(scale=1.0/K, size=(num_items, K))
    # R>0인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장한다.
    non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R2[i, j] > 0 ]

    # SGD 기법으로 P, Q 매트릭스를 업데이트 함
    for step in range(steps):
        for i, j, r in non_zeros : 
            # 잔차 구함
            b_iu2=u2+b_u2[i]+b_i2[j]
            eij = r-b_iu2-np.dot(P[i, :], Q[j, :].T)
            # Regulation을 반영한 SGD 업데이터 적용
            b_u2[i]=b_u2[i]+learning_rate*(eij-r_lambda*b_u2[i])
            b_i2[j]=b_i2[j]+learning_rate*(eij-r_lambda*b_i2[j])
            P[i, :] = P[i, :] + learning_rate*(eij * Q[j, :] - r_lambda*P[i, :])
            Q[j, :] = Q[j, :] + learning_rate*(eij * P[i, :] - r_lambda*Q[j, :])
        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0 :
            print("iter step",step, "rmse : " , rmse)

    return np.dot(P, Q.T)


# K=50, step=200
R_hat2=matrix_factorization2(np.array(data3),np.array(data3), 50, 500 )


#b_i로만 보정


def matrix_factorization2(R,R2, K, steps, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape

    np.random.seed(1)
    P = np.random.normal(scale=1.0/K, size=(num_users, K))
    Q = np.random.normal(scale=1.0/K, size=(num_items, K))
    # R>0인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장한다.
    non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R2[i, j] > 0 ]

    # SGD 기법으로 P, Q 매트릭스를 업데이트 함
    for step in range(steps):
        for i, j, r in non_zeros : 
            # 잔차 구함
            eij = r-np.dot(P[i, :], Q[j, :].T)-b_i[j]
            # Regulation을 반영한 SGD 업데이터 적용
            P[i, :] = P[i, :] + learning_rate*((eij-b_i[j]) * Q[j, :] - r_lambda*P[i, :])
            Q[j, :] = Q[j, :] + learning_rate*((eij-b_i[j]) * P[i, :] - r_lambda*Q[j, :])
        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0 :
            print("iter step",step, "rmse : " , rmse)

    return np.dot(P, Q.T)



#R_hat추정후 보정



def matrix_factorization3(R,R2, K, steps, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape

    np.random.seed(1)
    P = np.random.normal(scale=1.0/K, size=(num_users, K))
    Q = np.random.normal(scale=1.0/K, size=(num_items, K))
    # R>0인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장한다.
    non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R2[i, j] > 0 ]

    # SGD 기법으로 P, Q 매트릭스를 업데이트 함
    for step in range(steps):
        for i, j, r in non_zeros : 
            # 잔차 구함
            eij = r-np.dot(P[i, :], Q[j, :].T)
            # Regulation을 반영한 SGD 업데이터 적용
            P[i, :] = P[i, :] + learning_rate*(eij * Q[j, :] - r_lambda*P[i, :])
            Q[j, :] = Q[j, :] + learning_rate*(eij* P[i, :] - r_lambda*Q[j, :])
        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0 :
            print("iter step",step, "rmse : " , rmse)

    return np.dot(P, Q.T)


# K=50, step=200
R_hat3=matrix_factorization3(np.array(normal),np.array(data3), 50, 200 )
df_hat3=pd.DataFrame(R_hat3.copy())
df_hat4=pd.DataFrame(R_hat3.copy())
df_hat3.columns=data3.columns
df_hat4.columns=data3.columns
df_hat3.head()



for i in range(len(df_hat3.columns)):
    df_hat3.iloc[:,i]=(df_hat3.iloc[:,i])*denominator[i]+min[i]


for i in range(len(df_hat4.columns)):
    df_hat4.iloc[:,i]=(df_hat4.iloc[:,i]-b_i[i])*denominator[i]+min[i]

df_hat3.head()
data3.head()


df_hat3[data3>0].iloc[:,1].sort_values(ascending=False)

data3[data3>0].iloc[:,1].sort_values(ascending=False)
