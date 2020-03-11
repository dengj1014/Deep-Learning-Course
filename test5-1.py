# Deep-Learning-Course
'''
作业5-1
生成一个[0,1)之间均匀分布的随机数数组，包含1000个元素， 
随机种子为612。接收用户输入一个1-100之间的数字。
打印随机数组中所有索引值可以被输入整数整除的数字，
并打印序号和索引值。序号从1开始，依次加1.  例如，用户输入50，
则打印数组中索引值为0,50,100...1000的随机数。
'''
import numpy as np

np.random.seed(612)
rando=np.random.rand(1,1000)
inputnumber=int(input("Plese input a number:"))
n=1
print("序号","索引值","随机数")
for i in range(1000):
    if i%inputnumber==0:
        print(n,i,rando[0][i])
        n=n+1
