# Deep-Learning-Course
#5-2作业

import numpy as np

x=[64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03]
y=[62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84]
# x=[1,2,3,4,6]#实验数据
# y=[1,2,3,4,5]#实验数据
y_=np.var(y)#求方差
x_=np.var(x)#求方差
x_x_=x-x_#求xi-x的方差
x_2=x_x_**2#求xi-x的方差的平方
y_y_=y-y_#求yi-y的方差
fenm=x_x_*y_y_#求xi-x的方差与yi-y的方差的乘积
fsum=np.sum(fenm)#求公式中的分子
msum=np.sum(x_2)#求分母
w=fsum/msum
b=y_-w*x_

print("w的值为：{}  b的值为：{}".format(w,b))
# print(y_,x_)
