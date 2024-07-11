from predict import single_predict
import os
import shutil
from tqdm import tqdm


def batch_predict(img_dir):
    for img_name in tqdm(os.listdir(img_dir), desc='Processing images'):
        # 原图片路径
        img_path = os.path.join(img_dir, img_name)
        class_num = single_predict(img_path)
        # 测试结果存储目录
        destination_folder = os.path.join('testdata', class_num)
        if not os.path.exists(destination_folder):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(destination_folder)
        # 测试结果存储路径
        destination_path = os.path.join(destination_folder, img_name)
        shutil.copy(img_path, destination_path)  # 复制文件


if __name__ == '__main__':
    path = r"F:\dataset\病害数据图片\街北高速\20240327150614\images"
    batch_predict(path)
