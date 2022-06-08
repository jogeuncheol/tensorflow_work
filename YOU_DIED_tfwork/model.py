import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping

train_data = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\train\\'
valid_data = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\valid\\'
test_data = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\test\\'

image_size = (224, 224)
batch_size = 16
epochs = 100
patience = 8
# patience: 학습 조기종료 인자. val_loss가 10epoch 동안 낮아지지 않으면 학습 종료

# directory tree
# |- train
#     |- non_died
#     |- you_died
# |- valid
#     |- non_died
#     |- you_died
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
# ['non_died', 'you_died']

# 학습 데이터 섞기
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)

# 데이터 증강
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            'horizontal',
            input_shape=(224, 224, 3)
        ),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)

# 모델 구조 VGG-16 모방
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(224, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(224, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(224, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience) # patience :: 참을성 인자, val_loss 값이 더이상 줄어들지 않으면 조기 종료
    # tf.keras.callbacks.ModelCheckpoint('save_at_{epoch}.h5'), # 학습 중간저장 체크포인트
]
history = model.fit(
    train_set,
    validation_data=valid_set,
    callbacks=callbacks,
    epochs=epochs
)

model.save('./save_model')
model.save('./save_model.h5', save_format='h5')

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
plt.savefig('./model_19_dataset1_graph.png', facecolor='#FFFFFF')

# 테스트 데이터 평가
test_set = image_dataset_from_directory(
    test_data,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)
loss, accuracy = model.evaluate(test_set)
print('Test Loss : ', loss)
print('Test Accuracy : ', accuracy)

# 검증 데이터 평가
loss, accuracy = model.evaluate(valid_set)
print('Test Loss : ', loss)
print('Test Accuracy : ', accuracy)
