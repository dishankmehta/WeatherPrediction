from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np

seed = 7
np.random.seed(seed)


data = np.loadtxt("weather-data.txt")
X_train = data[:4646,:12]
Y_train = data[:4646,12:13]
X_test = data[4646:6936,:12]
Y_test = data[4646:6936,12:13]

len_y_train = len(Y_train)
len_y_test = len(Y_test)
print(len(Y_train))
print(len(Y_test))


for i in range(0,len_y_train):
    if(Y_train[i]==1000):
        Y_train[i] = 3
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]==100):
        Y_train[i] = 2
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]==10):
        Y_train[i] = 1
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]==1):
        Y_train[i] = 0
        Y_train[i] = int(Y_train[i])


for i in range(0,len_y_test):
    if(Y_test[i]==1000):
        Y_test[i] = 3
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]==100):
        Y_test[i] = 2
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]==10):
        Y_test[i] = 1
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]==1):
        Y_test[i] = 0
        Y_test[i] = int(Y_test[i])

print(Y_train)
print(Y_test)

Y_train = Y_train.astype('int32')
Y_train = np_utils.to_categorical(Y_train,4)

Y_test = Y_test.astype('int32')
Y_test = np_utils.to_categorical(Y_test,4)


model = Sequential()
model.add(Dense(100, input_dim=12, init='uniform', activation='relu'))
model.add(Dense(80, init='uniform', activation='relu'))
model.add(Dense(60, init='uniform', activation='relu'))
model.add(Dense(60, init='uniform', activation='relu'))
model.add(Dense(4))
model.add(Activation('softmax'))

model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=150, batch_size=10, verbose=2, validation_data=(X_test,Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print("\n")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5',overwrite=True)