# Deep-Learning-Course
#5-3作业
import numpy as np

x0=np.ones(10,)#生成全1数组
x1=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
x2=np.array([2,3,4,2,3,4,2,4,1,3])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
X=np.stack((x0,x1,x2),axis=1)#堆叠x0,x1,x2
# print(X)
Y=y.reshape(10,1)#变换y为10*1的二维数组
# print(Y)
X=np.mat(X)#把X转化为矩阵
Y=np.mat(Y)#把Y转化为矩阵
w=((X.T*X)**(-1))*(X.T*Y)#求w的值
print("w的形状：{}".format(w.shape))
print("X:{}".format(X))
print("Y:{}".format(Y))
print("W:{}".format(w))
