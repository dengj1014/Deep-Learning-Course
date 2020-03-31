import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

x=tf.placeholder("float",name="x")
y=tf.placeholder("float",name="y")
x_data=np.linspace(0,1,500)
y_data=3.1234*x_data+2.98+np.random.randn(*x_data.shape)*0.4
plt.figure(figsize=(8,8))
plt.scatter(x_data,y_data)
plt.plot(x_data,3.1234*x_data+2.98,"r")
def model(x,w,b):
    return tf.multiply(x,w)+b
w=tf.Variable(1.0,name="w0")
b=tf.Variable(0.0,name="b0")
pred=model(x,w,b)
train_epochs=10
learning_rate=0.01
loss_function=tf.reduce_mean(tf.square(y-pred))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
step=0
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):
        _,loss=sess.run([optimizer,loss_function],{x:xs,y:ys})
        step+=1
        if step % 20 == 0:
            print("Training Epoch:",'%d'%(epoch+1),"Step:%d"%(step),"loss=%f"%(loss))
    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    plt.plot(x_data,w0temp*x_data+b0temp)
   
    

print("预测x=5.79时，y的值：",sess.run(model(w,5.79,b)))
plt.show()
sess.close()
logdir='D:/log'
writer=tf.summary.FileWriter(logdir,tf.get_default_graph())
# writer=tf.summary.create_file_writer(logdir)
writer.close()