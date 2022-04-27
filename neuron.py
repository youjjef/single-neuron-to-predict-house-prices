import tensorflow as tf
import numpy as np
from tensorflow import keras
def house_model():

    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0],dtype=float)
    ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype=float)

    model =tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(xs, ys, epochs=1000)

    return model
model = house_model()
new_y = 8.0
prediction = model.predict([new_y])
print(prediction)
