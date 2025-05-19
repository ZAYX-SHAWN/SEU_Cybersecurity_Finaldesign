import os
import random
import shutil

# 1. 原始数据集路径
DATA_DIR = r'D:\final/image_cicids2017'  # 你的原始数据集路径
TRAIN_DIR = r'D:\final\train_cicids2017'
TEST_DIR = r'D:\final\test_cicids2017'

# 2. 创建训练和测试集文件夹，如果不存在就创建
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 3. 设置测试集比例
test_ratio = 0.2

# 4. 遍历每个攻击类型的子文件夹
for attack_type in os.listdir(DATA_DIR):
    attack_path = os.path.join(DATA_DIR, attack_type)
    if not os.path.isdir(attack_path):
        continue

    # 4.1 获取所有图片文件列表
    imgs = os.listdir(attack_path)

    # 4.2 打乱图片顺序
    random.shuffle(imgs)

    # 4.3 测试集大小
    test_size = int(len(imgs) * test_ratio)

    # 4.4 准备训练集和测试集的文件夹（保留攻击类型子目录结构）
    train_subdir = os.path.join(TRAIN_DIR, attack_type)
    test_subdir = os.path.join(TEST_DIR, attack_type)
    os.makedirs(train_subdir, exist_ok=True)
    os.makedirs(test_subdir, exist_ok=True)

    # 4.5 移动或复制图片
    for i, img_name in enumerate(imgs):
        src_path = os.path.join(attack_path, img_name)
        if i < test_size:
            dst_path = os.path.join(test_subdir, img_name)
        else:
            dst_path = os.path.join(train_subdir, img_name)

        # 移动文件，如果你想保留原图则用shutil.copy(src_path, dst_path)
        shutil.move(src_path, dst_path)

print("数据集划分完成！")