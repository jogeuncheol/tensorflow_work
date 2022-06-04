import cv2
import os

path_dir = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\train\\non_died\\'
save_dir = 'E:\\workspace\\ML_study\\AI_project\\dataset2\\non_died\\'


def image_crop():
    for idx, filename in enumerate(os.listdir(path_dir)):
        image_path = os.path.join(path_dir, filename)
        print(image_path)

        # image read
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        except:
            print('image read fail')
            break
        h, w, _ = image.shape

        # image crop
        # image = image[h//3:h//3*2, w//3:w//3*2]
        image = image[h // 4:h // 4 * 3, w // 8:w // 8 * 7]
        cv2.imshow("crop image", image)
        if cv2.waitKey(5) & 0xFF == 27:
            cv2.destroyAllWindows()

        # crop image save
        img_name = save_dir + str(idx) + '.jpg'
        cv2.imwrite(img_name, image)

    print(len(os.listdir(path_dir)))
    # total you_died train image: 884


def image_resize():
    for idx, filename in enumerate(os.listdir(path_dir)):
        image_path = os.path.join(path_dir, filename)
        print(image_path)

        # image read
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        except:
            print('image read fail')
            break
        if image.shape[0] == 1080:
            image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow('resize', image)
        if cv2.waitKey(5) & 0xFF == 27:
            cv2.destroyAllWindows()

        # resize image save
        img_name = save_dir + str(idx) + '.jpg'
        cv2.imwrite(img_name, image)


image_resize()