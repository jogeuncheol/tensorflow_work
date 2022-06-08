import tensorflow as tf
import cv2
import numpy as np
from queue import Queue
import mss


def capture():
    model = tf.keras.models.load_model('./save_model_19_dataset1_class2')
    img_height = 224
    img_width = 224
    class_names = ['non_died', 'you_died']

    with mss.mss() as screen:
        q = Queue(10)
        count = 0
        while True:
            original_image = np.array(screen.grab(screen.monitors[1]))
            image = cv2.resize(original_image, dsize=(img_height, img_width))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.reshape(1, img_height, img_width, 3).astype('float32')
            prediction = model.predict(image)
            prediction = tf.nn.softmax(prediction)
            class_name = class_names[np.argmax(prediction)]
            print("predict", class_name)
            resize = cv2.resize(original_image, dsize=(960, 480))
            h, w, _ = resize.shape
            # 큐가 full 이 아니면 분류값 넣기
            if not q.full():
                q.put(np.argmax(prediction))
            # 큐가 full 이면 프레임의 분류값을 보고 count 증가
            if q.full():
                label = q.get()
                if label:
                    count += 1
                else:
                    count = 0
            # 5 프레임이상 분류가 you died --> putText
            if count > 5:
                cv2.putText(
                    resize,
                    'YOU DIED',
                    (w//3, h//4), 0, 2,
                    (255, 0, 255),
                    2
                )
            cv2.imshow('capture', resize)
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    capture()
