from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Definir la ruta de la base de datos
database_path = '../../database'

# Preprocesamiento de datos con ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)  # Especificar el split de entrenamiento y validación

# Generadores de datos para entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    database_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Indicar que es el subset de entrenamiento

validation_generator = train_datagen.flow_from_directory(
    database_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Indicar que es el subset de validación

# Inicializar el modelo
model = Sequential()

# Agregar capas convolucionales y de pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Agregar capas fully connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))  # Número de clases

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=10,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))
