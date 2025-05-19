import os
import cv2
import warnings
warnings.filterwarnings("ignore")

def get_224(folder, dstdir):
    imgfilepaths = []
    for root, dirs, imgs in os.walk(folder):
        for this_img in imgs:
            this_img_path = os.path.join(root, this_img)
            imgfilepaths.append(this_img_path)
    for this_img_path in imgfilepaths:
        dir_name, filename = os.path.split(this_img_path)
        dir_name = dir_name.replace(folder, dstdir)
        new_file_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img = cv2.imread(this_img_path)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(new_file_path, img)
    print('Finish resizing'.format(folder=folder))


DATA_DIR_224 = r'D:\final\train_A_cicids2017/'
get_224(folder=r'D:\final\train_cicids2017', dstdir=DATA_DIR_224)

DATA_DIR2_224 = r'D:\final/test_A_cicids2017/'
get_224(folder=r'D:\final\test_cicids2017', dstdir=DATA_DIR2_224)
