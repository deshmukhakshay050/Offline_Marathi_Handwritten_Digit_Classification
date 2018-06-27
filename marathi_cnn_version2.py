
# coding: utf-8

# In[4]:


import numpy as np



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import pandas as pd
import numpy as np

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation






# In[6]:


data0=pd.read_csv('20000.csv') 


# In[7]:


data0.shape


# In[2]:


data1 = np.load('marathi_numerals_DATA.npy')


# In[3]:


data2 = np.load('marathi_numerals_LABEL.npy')


# In[5]:


data1.shape,data2.shape


# In[6]:


x1=data1[0:2000,:,:,:]
x2=data1[2000:4000,:,:,:]
x3=data1[4000:6000,:,:,:]
x4=data1[6000:8000,:,:,:]
x5=data1[40000:42000,:,:,:]
x6=data1[50000:60000,:,:,:]
x7=data1[60000:70000,:,:,:]
xtest=data1[70000:80000,:,:,:]
x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,x6.shape,x7.shape,xtest.shape


# In[7]:


y1=data2[0:10000,:]
y2=data2[10000:20000,:]
y3=data2[20000:30000,:]
y4=data2[30000:40000,:]
y5=data2[40000:50000,:]
y6=data2[50000:60000,:]
y7=data2[60000:70000,:]
ytest=data2[70000:80000,:]
y1.shape,y2.shape,y3.shape,y4.shape,y5.shape,y6.shape,y7.shape,ytest.shape


# In[8]:


xtrain1=np.append(x1,x2,axis=0)
xtrain1=np.append(xtrain1,x3,axis=0)
xtrain1=np.append(xtrain1,x4,axis=0)
xtrain1=np.append(xtrain1,x5,axis=0)
xtrain1=np.append(xtrain1,x6,axis=0)


ytrain1=np.append(y1,y2,axis=0)
ytrain1=np.append(ytrain1,y3,axis=0)
ytrain1=np.append(ytrain1,y4,axis=0)
ytrain1=np.append(ytrain1,y5,axis=0)
ytrain1=np.append(ytrain1,y6,axis=0)



xtest1=x7
ytest1=y7

xtrain1.shape,ytrain1.shape




# In[9]:


xtrain2=np.append(x1,x2,axis=0)
xtrain2=np.append(xtrain2,x3,axis=0)
xtrain2=np.append(xtrain2,x4,axis=0)
xtrain2=np.append(xtrain2,x5,axis=0)
xtrain2=np.append(xtrain2,x7,axis=0)


ytrain2=np.append(y1,y2,axis=0)
ytrain2=np.append(ytrain2,y3,axis=0)
ytrain2=np.append(ytrain2,y4,axis=0)
ytrain2=np.append(ytrain2,y5,axis=0)
ytrain2=np.append(ytrain2,y7,axis=0)



xtest2=x6
ytest2=y6

xtrain2.shape,ytrain2.shape


# In[10]:


xtrain3=np.append(x1,x2,axis=0)
xtrain3=np.append(xtrain3,x3,axis=0)
xtrain3=np.append(xtrain3,x4,axis=0)
xtrain3=np.append(xtrain3,x6,axis=0)
xtrain3=np.append(xtrain3,x7,axis=0)


ytrain3=np.append(y1,y2,axis=0)
ytrain3=np.append(ytrain3,y3,axis=0)
ytrain3=np.append(ytrain3,y4,axis=0)
ytrain3=np.append(ytrain3,y6,axis=0)
ytrain3=np.append(ytrain3,y7,axis=0)



xtest3=x5
ytest3=y5

xtrain3.shape,ytrain3.shape


# In[12]:


xtrain4=np.append(x1,x2,axis=0)
xtrain4=np.append(xtrain4,x3,axis=0)
xtrain4=np.append(xtrain4,x5,axis=0)
xtrain4=np.append(xtrain4,x6,axis=0)
xtrain4=np.append(xtrain4,x7,axis=0)


ytrain4=np.append(y1,y2,axis=0)
ytrain4=np.append(ytrain4,y3,axis=0)
ytrain4=np.append(ytrain4,y5,axis=0)
ytrain4=np.append(ytrain4,y6,axis=0)
ytrain4=np.append(ytrain4,y7,axis=0)



xtest4=x4
ytest4=y4

xtrain4.shape,ytrain4.shape


# In[13]:


xtrain5=np.append(x1,x2,axis=0)
xtrain5=np.append(xtrain5,x4,axis=0)
xtrain5=np.append(xtrain5,x5,axis=0)
xtrain5=np.append(xtrain5,x6,axis=0)
xtrain5=np.append(xtrain5,x7,axis=0)


ytrain5=np.append(y1,y2,axis=0)
ytrain5=np.append(ytrain5,y4,axis=0)
ytrain5=np.append(ytrain5,y5,axis=0)
ytrain5=np.append(ytrain5,y6,axis=0)
ytrain5=np.append(ytrain5,y7,axis=0)



xtest5=x3
ytest5=y3

xtrain5.shape,ytrain5.shape


# In[14]:


xtrain6=np.append(x1,x3,axis=0)
xtrain6=np.append(xtrain6,x4,axis=0)
xtrain6=np.append(xtrain6,x5,axis=0)
xtrain6=np.append(xtrain6,x6,axis=0)
xtrain6=np.append(xtrain6,x7,axis=0)


ytrain6=np.append(y1,y3,axis=0)
ytrain6=np.append(ytrain6,y4,axis=0)
ytrain6=np.append(ytrain6,y5,axis=0)
ytrain6=np.append(ytrain6,y6,axis=0)
ytrain6=np.append(ytrain6,y7,axis=0)



xtest6=x2
ytest6=y2

xtrain6.shape,ytrain6.shape


# In[15]:


xtrain7=np.append(x2,x3,axis=0)
xtrain7=np.append(xtrain7,x4,axis=0)
xtrain7=np.append(xtrain7,x5,axis=0)
xtrain7=np.append(xtrain7,x6,axis=0)
xtrain7=np.append(xtrain7,x7,axis=0)


ytrain7=np.append(y2,y3,axis=0)
ytrain7=np.append(ytrain7,y4,axis=0)
ytrain7=np.append(ytrain7,y5,axis=0)
ytrain7=np.append(ytrain7,y6,axis=0)
ytrain7=np.append(ytrain7,y7,axis=0)



xtest7=x1
ytest7=y1

xtrain7.shape,ytrain7.shape


# In[17]:


model=Sequential()


model.add(Convolution2D(32, 3, 3, input_shape=(1,28,28))) 
p1= Activation('relu')
model.add(p1)


model.add(Convolution2D(32, 3, 3))
p2=Activation('relu')
model.add(p2)


model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(64, 3,3 ))
p3=Activation('relu')
model.add(p3)
model.add(MaxPooling2D(pool_size=(2, 2)))


          
model.add(Dropout(0.2))
p4=Flatten()
model.add(p4)


model.add(Dropout(0.2))


model.add(Dense(400, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(150, activation='relu'))
model.add(Dense(10, activation='softmax'))    



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# In[18]:


model.summary()


# In[19]:


for i in range(2):
    model.fit(xtrain1, ytrain1,validation_data=[xtest1,ytest1], epochs=1, batch_size=200)
    model.fit(xtrain2, ytrain2,validation_data=[xtest2,ytest2], epochs=1, batch_size=200)
    model.fit(xtrain3, ytrain3,validation_data=[xtest3,ytest3], epochs=1, batch_size=200)
    model.fit(xtrain4, ytrain4,validation_data=[xtest4,ytest4], epochs=1, batch_size=200)
    model.fit(xtrain5, ytrain5,validation_data=[xtest5,ytest5], epochs=1, batch_size=200)
    model.fit(xtrain6, ytrain6,validation_data=[xtest6,ytest6], epochs=1, batch_size=200)
    model.fit(xtrain7, ytrain7,validation_data=[xtest7,ytest7], epochs=1, batch_size=200)


scores = model.evaluate(xtest, ytest, verbose=0)
print('Test accuracy:', scores[1])


# In[14]:


model_json = model.to_json()



with open("model_marathi_cnn_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_marathi_cnn_2.h5")
print("Saved model to disk")


# In[1]:


from keras.models import model_from_json

json_file = open('model_marathi_cnn_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_marathi_cnn_2.h5")

model=loaded_model




# In[15]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain1, ytrain1,validation_data=[xtest1,ytest1], epochs=1, batch_size=200)
model.fit(xtrain2, ytrain2,validation_data=[xtest2,ytest2], epochs=1, batch_size=200)


# In[16]:


scores = model.evaluate(xtrain1, ytrain1, verbose=0)
print('Test accuracy:', scores[1])


# In[17]:


scores = model.evaluate(xtrain2, ytrain2, verbose=0)
print('Test accuracy:', scores[1])


# In[18]:


scores = model.evaluate(xtrain3, ytrain3, verbose=0)
print('Test accuracy:', scores[1])


# In[19]:


scores = model.evaluate(xtrain4, ytrain4, verbose=0)
print('Test accuracy:', scores[1])


# In[21]:



scores = model.evaluate(xtest, ytest, verbose=0)
print('Test accuracy:', scores[1])


# In[2]:


w1 = model.layers[0].get_weights()[0]

b1 = model.layers[0].get_weights()[1]


w2 = model.layers[2].get_weights()[0]
b2 = model.layers[2].get_weights()[1]


w3 = model.layers[5].get_weights()[0]
b3 = model.layers[5].get_weights()[1]


print(w2.shape,b2.shape)
print(w1.shape,b1.shape)
print(w3.shape,b3.shape)


# In[18]:


import matplotlib.pyplot as plt
import matplotlib.cm as cmplt
img=x4[4]

print(img.shape)
img=np.expand_dims(img,axis=0)

def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img)
    convolutions = np.squeeze(convolutions)
    print ('Shape of conv:', convolutions.shape)
    
    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))
    print(convolutions.shape)
    
    
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')
       
    return convolutions


# In[21]:


f1=layer_to_visualize(model.layers[1])#Visualization of conv layers on activation relu
f2=layer_to_visualize(model.layers[3])
f3=layer_to_visualize(model.layers[6])


# In[22]:


plt.show()


# In[19]:


f3=layer_to_visualize(model.layers[4])
f4=layer_to_visualize(model.layers[7])#Visualization of maxpool layers on 


# In[20]:


plt.show()


# In[3]:


from keras.models import Model 

model1=Model(inputs=model.input,outputs=model.layers[9].output) 


# In[19]:


xtrain1=np.append(xtrain1,x7,axis=0)
ytrain1=np.append(ytrain1,y7,axis=0)




# In[20]:


xtrain1.shape,ytrain1.shape


# In[44]:


import numpy as np

x0=np.array(np.zeros(shape=(7030,1,28,28)))
x1=np.array(np.zeros(shape=(6995,1,28,28)))
x2=np.array(np.zeros(shape=(7050,1,28,28)))
x3=np.array(np.zeros(shape=(7003,1,28,28)))
x4=np.array(np.zeros(shape=(6873,1,28,28)))
x5=np.array(np.zeros(shape=(6945,1,28,28)))
x6=np.array(np.zeros(shape=(6987,1,28,28)))
x7=np.array(np.zeros(shape=(7087,1,28,28)))
x8=np.array(np.zeros(shape=(7025,1,28,28)))
x9=np.array(np.zeros(shape=(7005,1,28,28)))


print(xtrain1.shape)

i0=i1=i2=i3=i4=i5=i6=i7=i8=i9=0


for i in range(len(ytrain1)):
    y=np.argmax(ytrain1[i])
    if(y==0):
        x0[i0,:,:,:]=xtrain1[i,:,:,:]
        i0=i0+1
    elif(y==1):
        x1[i1,:,:,:]=xtrain1[i,:,:,:]
        i1=i1+1
    elif(y==2):
        x2[i2,:,:,:]=xtrain1[i,:,:,:]
        i2=i2+1
    elif(y==3):
        x3[i3,:,:,:]=xtrain1[i,:,:,:]
        i3=i3+1
    elif(y==4):
        x4[i4,:,:]=xtrain1[i,:,:]
        i4=i4+1
    elif(y==5):
        x5[i5,:,:]=xtrain1[i,:,:]
        i5=i5+1
    elif(y==6):
        x6[i6,:,:]=xtrain1[i,:,:]
        i6=i6+1
    elif(y==7):
        x7[i7,:,:]=xtrain1[i,:,:]
        i7=i7+1
    elif(y==8):
        x8[i8,:,:]=xtrain1[i,:,:]
        i8=i8+1
    elif(y==9):
        x9[i9,:,:]=xtrain1[i,:,:]
        i9=i9+1
   



print(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9)



# In[45]:


print(i0+i1+i2+i3+i4+i5+i6+i7+i8+i9)


# In[46]:


temp=x4[7]
temp=temp.reshape(28,28)
plt.imshow(temp,cmap='gray')
plt.show(temp,1)


# In[47]:


f0= model1.predict(x0)
f1= model1.predict(x1)
f2= model1.predict(x2)
f3= model1.predict(x3)
f4= model1.predict(x4)
f5= model1.predict(x5)
f6= model1.predict(x6)
f7= model1.predict(x7)
f8= model1.predict(x8)
f9= model1.predict(x9)




# In[48]:


print(f0.shape,f1.shape,f2.shape,f3.shape,f4.shape,f5.shape,f6.shape,f7.shape,f8.shape,f9.shape)


# In[49]:


np.savetxt('f0_marathi_new.csv', f0, delimiter=",", fmt="%s") 


# In[50]:


np.savetxt('f1_marathi_new.csv', f1, delimiter=",", fmt="%s") 
np.savetxt('f2_marathi_new.csv', f2, delimiter=",", fmt="%s") 
np.savetxt('f3_marathi_new.csv', f3, delimiter=",", fmt="%s") 
np.savetxt('f4_marathi_new.csv', f4, delimiter=",", fmt="%s") 
np.savetxt('f5_marathi_new.csv', f5, delimiter=",", fmt="%s") 
np.savetxt('f6_marathi_new.csv', f6, delimiter=",", fmt="%s") 
np.savetxt('f7_marathi_new.csv', f7, delimiter=",", fmt="%s") 
np.savetxt('f8_marathi_new.csv', f8, delimiter=",", fmt="%s") 
np.savetxt('f9_marathi_new.csv', f9, delimiter=",", fmt="%s") 


# In[52]:


model.summary()


# In[24]:


print(f1.shape)

temp=f7[1]


temp=temp.reshape(64,5,5)

n = temp.shape[0]
n = int(np.ceil(np.sqrt(n)))
fig = plt.figure(figsize=(12,8))
for i in range(len(temp)):
    ax = fig.add_subplot(n,n,i+1)
    ax.imshow(temp[i], cmap='gray')


# In[55]:


plt.show()

