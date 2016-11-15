from keras.models import model_from_json
import numpy as np

pre_data = np.loadtxt('predict.txt')

model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


pre_data = pre_data.astype('int32')
pre_data = pre_data.reshape(1,12)

print(pre_data)

ans = model.predict(x=pre_data)
print('\n',ans)


