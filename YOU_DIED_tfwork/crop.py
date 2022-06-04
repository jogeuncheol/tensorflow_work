import cv2
import os

path_dir = 'E:\\workspace\\ML_study\\AI_project\\dataset1\\train\\you_died\\'
save_dir = 'E:\\workspace\\ML_study\\AI_project\\dataset2\\you_died\\'

for idx, filename in enumerate(os.listdir(path_dir)):
    image_path = os.path.join(path_dir, filename)
    print(image_path)

    # image read
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    # image crop
    image = image[h//3:h//3*2, w//3:w//3*2]
    cv2.imshow("crop image", image)
    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()

    # crop image save
    img_name = save_dir + str(idx + 731) + '.jpg'
    cv2.imwrite(img_name, image)

print(len(os.listdir(path_dir)))
# total you_died train image: 884