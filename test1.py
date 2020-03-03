'''
	利用Python完成一元二次方程 的求解，
    要求程序输入任意 的值后，程序能判断输出有解或无解，
    有解的话，输出 的解为多少。
    '''
import math

a=int(input("Plese input a:"))  #定义a
b=int(input("Plese input b:"))  #定义b
c=int(input("Plese input c:"))  #定义c
s=(b*b)-(4*a*c)                 #一元二次根式判断是否有解的公司
if s<0:                         #s<0代表此方程没有解
    print("不好意思，次方程无解！！！")
elif s>0:                       #s>0代表此方程有两个不相同的解
    end_1=(-b+math.sqrt(s))/(2*a)
    end_2=(-b-math.sqrt(s))/(2*a)
    print("此方程有两个解，第一个解为:{},第二个解为:{}".format(end_1,end_2))
else:                           #s等于0代表此方程有两个相同的解，也就是一个解
    end_1=(-b)/(2*a)
    print("此方程只有一个解：{}".format(end_1))
   
