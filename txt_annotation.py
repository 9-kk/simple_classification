import os


def traverse_directories(path):
    for root, dirs, files in os.walk(path):
        for single_dir in dirs:
            print("Found subdirectory: ", os.path.join(root, single_dir))
        # for file in files:
        #     print("Found file: {}".format(os.path.join(root, file)))


def main():
    # 调用函数，传入需要遍历的文件夹路径
    traverse_directories("F:/dataset/病害数据图片/分类")


if __name__ == '__main__':
    main()
