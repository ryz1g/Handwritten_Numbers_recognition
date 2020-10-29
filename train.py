import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import model_from_json

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

class myCall(tf.keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        if(logs.get('val_accuracy')>0.99) :
            self.model.stop_training=True

callbacks=myCall()

x_train=np.load("train_images_canny.npy")
x_test=np.load("test_images_canny.npy")
y_train=np.load("train_labels_canny.npy")
y_test=np.load("test_labels_canny.npy")
x_train=x_train.reshape(60000,28,28,1)
y_train=y_train.reshape(60000,1)
x_test=x_test.reshape(10000,28,28,1)
y_test=y_test.reshape(10000,1)

x_train=x_train.astype("float32")
y_train=tf.keras.utils.to_categorical(y_train,10)
x_test=x_test.astype("float32")
y_test=tf.keras.utils.to_categorical(y_test,10)

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5,5), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2,2))
#model.add(tf.keras.layers.Conv2D(128, (3,3), activation="relu"))
#model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#model.load_weights("model_w.h5")

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, epochs=30, batch_size=2048, callbacks=[callbacks])

model.summary()

plot_graphs(history, "accuracy")

ti=input("save weights?(Y/N)")
if ti=="Y" :
    model_json=model.to_json()
    with open("model_w.json", "w") as json_file :
        json_file.write(model_json)
    model.save_weights("model_w.h5")
