import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt

tf.device('/gpu:0')
# Dimensiones de nuestras imágenes.
img_width, img_height = 128, 128

# Directorios de los datos de entrenamiento y validación
train_dir = 'C:/Users/Omar/OneDrive - UNIVERSIDAD DE INGENIERIA Y TECNOLOGIA/Escritorio/VCICLO/MACHINE LEARNING/project4-machine/train'
test_dir = 'C:/Users/Omar/OneDrive - UNIVERSIDAD DE INGENIERIA Y TECNOLOGIA/Escritorio/VCICLO/MACHINE LEARNING/project4-machine/test'

batch_size = 32

# Ajustar la forma de entrada dependiendo de la configuración de los canales
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Creación del modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))  # Cambia el número de neuronas aquí para que coincida con el número de actores/clases
model.add(Activation('softmax'))  # Cambia 'sigmoid' por 'softmax' para clasificación multiclase
# Resumen del modelo
model.summary()

# Compilación del modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=10,  # Establece el número de épocas que desees
    validation_data=validation_generator)

# Graficar las curvas de pérdida y precisión
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Pérdidas de entrenamiento', 'Pérdidas de validación'], fontsize=18)
plt.xlabel('Épocas', fontsize=16)
plt.ylabel('Pérdida', fontsize=16)
plt.title('Curvas de pérdida', fontsize=16)
plt.show()

plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Precisión de entrenamiento', 'Precisión de validación'], fontsize=18)
plt.xlabel('Épocas', fontsize=16)
plt.ylabel('Precisión', fontsize=16)
plt.title('Curvas de precisión', fontsize=16)
plt.show()
