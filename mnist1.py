import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#导入数据
mnist=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
total_num=len(train_images)
#进行归一化处理
train_images=train_images/255.0
test_images=test_images/255.0
#独热编码
train_labels_ohe=tf.one_hot(train_labels,depth=10).numpy()
test_labels_ohe=tf.one_hot(test_labels,depth=10).numpy()
#建立Sequential线性堆叠模型
model=tf.keras.models.Sequential()
#添加平坦层
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#添加隐藏层
model.add(tf.keras.layers.Dense(units=64,kernel_initializer='normal',activation='relu'))
model.add(tf.keras.layers.Dense(units=32,kernel_initializer='normal',activation='relu'))
#添加输出层
model.add(tf.keras.layers.Dense(10,activation='softmax'))
# model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#设置优化器，定义损失函数，评估模型的方式为准确率
#设置训练参数
train_epochs=59
batch_size=30
train_history=model.fit(train_images,train_labels_ohe,validation_split=0.2,epochs=train_epochs,batch_size=batch_size,verbose=2)
#可视化
# def show_train_history(train_history,train_metric,val_metric):
#     plt.plot(train_history.history[train_metric])
#     plt.plot(train_history.history[val_metric])
#     plt.title('Train History')
#     plt.ylabel(train_metric)
#     plt.xlabel('Epoch')
#     plt.show()
# show_train_history(train_history,'loss','val_loss')