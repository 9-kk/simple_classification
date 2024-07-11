import numpy as np
import os
import random
import shutil

# dataset_path = r'F:\BaiduSyncdisk\gitlab\dataset\intinsic'
dataset_path = r"F:/dataset/病害数据图片/分类"
fisheye_path = r'F:\gitlab\dataset\intinsic\fisheye.txt'
pinhole_path = r'F:\gitlab\dataset\intinsic\pinhole.txt'


# 读取txt文件获取文件名列表
# def make_list(path):
#     new_list = []
#     f = open(path, encoding='utf-8')
#     line = f.readline()
#     while line:
#         line = str(line.split('.')[0]) + '.jpg'
#         img_path = os.path.join(dataset_path, line)
#         new_list.append(img_path)
#         line = f.readline()
#     train_list, val_list = split_list(new_list)
#     copy_files(train_list, path, "train")
#     copy_files(val_list, path, "val")

def make_list(path):
    # 遍历文件夹，即分类的个数
    for root_dir in os.listdir(path):
        folder_path = os.path.join(path, root_dir)
        for root, dirs, files in os.walk(folder_path):
            # 将文件路径list打乱划分为train和val
            file_list = [os.path.join(root, item) for item in files]
            train_list, val_list = split_list(file_list)
            copy_files(train_list, root, "train")
            copy_files(val_list, root, "val")


# 随机打乱列表并分为训练集与测试集
def split_list(lst, ratio=0.8):
    # 打乱列表
    random.shuffle(lst)
    split_index = int(len(lst) * ratio)  # 计算分割索引
    return lst[:split_index], lst[split_index:]


# 将对应文件复制到指定文件夹
def copy_files(file_list, path_name, data_class):
    save_path = os.path.join(os.getcwd(), 'dataset')
    destination_name = path_name.split('\\')[-1].split('.')[0]
    destination_folder = os.path.join(save_path, data_class, destination_name)
    if not os.path.exists(destination_folder):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(destination_folder)
    for file_path in file_list:
        file_name = os.path.basename(file_path)  # 获取文件名
        destination_path = os.path.join(destination_folder, file_name)  # 构建目标路径
        shutil.copy(file_path, destination_path)  # 复制文件
    print(destination_folder, 'done!')


def main():
    make_list(dataset_path)


if __name__ == '__main__':
    main()
