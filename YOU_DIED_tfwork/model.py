import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import cv2

train_data = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\train\\'
valid_data = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\valid\\'

image_size = (180, 180)
batch_size = 16
epochs = 50

train_set = image_dataset_from_directory(
    train_data,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)
valid_set = image_dataset_from_directory(
    valid_data,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

labels = train_set.class_names
print(labels)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            'horizontal',
            input_shape=(180, 180, 3)
        ),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(224, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(224, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(112, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(112, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(56, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(56, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(28, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(28, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(14, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(14, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid'),
]) # Dropout :: 규제기법, 항상 사용하는것은 아님, input:0.2, 은닉층:0.5, 은닉층을 2배로 준다.

model.compile(
    optimizer='Adam',# tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10) # patience :: 참을성 인자, val_loss 값이 더이상 줄어들지 않으면 조기 종료
    # tf.keras.callbacks.ModelCheckpoint('save_at_{epoch}.h5'),
]
history = model.fit(
    train_set,
    validation_data=valid_set,
    callbacks=callbacks,
    epochs=epochs
)

model.save('./save_model_5')
model.save('./save_model_5.h5', save_format='h5')

early_epoch = history.epoch[-1] + 1
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

# 학습 종료시 epoch 기록
epochs_range = range(early_epoch)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 테스트 데이터 평가
test_data = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\test\\'
test_set = image_dataset_from_directory(
    test_data,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)
loss, accuracy = model.evaluate(test_set)
print('Test Loss : ', loss)
print('Test Accuracy : ', accuracy)
