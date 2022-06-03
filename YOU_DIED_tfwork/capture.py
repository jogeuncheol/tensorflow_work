import tensorflow as tf
import cv2
import numpy as np
import mss

model = tf.keras.models.load_model('./save_model_6')
img_height = 224
img_width = 224
class_names = ['non_died', 'you_died']

with mss.mss() as screen:
    while True:
        original_image = np.array(screen.grab(screen.monitors[1]))
        image = cv2.resize(original_image, dsize=(img_height, img_width))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(1, img_height, img_width, 3).astype('float32')
        prediction = model.predict(image)
        prediction = tf.nn.softmax(prediction)
        class_name = class_names[np.argmax(prediction)]
        print("predict", class_name)

        cv2.imshow('capture', cv2.resize(original_image, dsize=(960, 480)))
        if cv2.waitKey(5) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
