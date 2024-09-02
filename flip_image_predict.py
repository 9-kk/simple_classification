from model import resnet34
import os
import argparse
import torch
from torchvision import transforms
import json
from PIL import Image
import shutil
import numpy


def parse_args():
    parser = argparse.ArgumentParser()
    # 有关数据集的读取
    parser.add_argument('--img_dir', type=str, default=r'F:\dataset\病害数据图片\分类',
                        help='输入图片的目录')
    # parser.add_argument('--save_dir', type=str, default=os.getcwd(),
    #                     help='生成txt文件的保存路径')
    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(), "lanes_cls.txt"),
                        help='生成txt文件的保存路径')
    # parser.add_argument('--batch_size', type=int, default=1, help='每次预测时将多少张图片打包成一个batch')

    return parser.parse_args()


def generate_txt(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # read class_indict
    json_path = 'log/2024_09_02_11_42/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "log/2024_09_02_11_42/resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    # 存储车道位置的list
    lanes_list = []

    img_path_list = []
    for root, dirs, files in os.walk(args.img_dir):
        for img_name in files:
            img_path_list.append(os.path.join(root, img_name))
    with torch.no_grad():
        for img_path in img_path_list:
            img = Image.open(img_path)
            # 对图片进行水平翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            predict_img = data_transform(img).unsqueeze(0)

            # predict class
            output = model(predict_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            # # 将img转回PIL格式
            # print(img.shape)
            # # 维度大小为[1, 3, 256, 306](b, c, h, w)，去掉表示批次的第一维度
            # # (c, h, w) -> (h, w, c)
            # img_array = numpy.uint8(img[0].permute(1, 2, 0))
            # print(type(img_array))
            # print(img_array.shape)
            # a1 = transforms.ToPILImage()
            # img_PIL = a1(img_array)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                lanes_list.append(class_indict[str(cla.numpy())])
                # print(class_indict[str(cla.numpy())],  os.path.basename(img_path))

                # 测试结果存储目录
                destination_folder = os.path.join('testdata', class_indict[str(cla.numpy())])
                if not os.path.exists(destination_folder):
                    # 如果文件夹不存在，则创建文件夹
                    os.makedirs(destination_folder)
                # 测试结果存储路径
                destination_path = os.path.join(destination_folder, 'flip_' + os.path.basename(img_path))
                # 如果需要，保存翻转后的图片
                img.save(destination_path)


if __name__ == '__main__':
    args = parse_args()
    generate_txt(args)
