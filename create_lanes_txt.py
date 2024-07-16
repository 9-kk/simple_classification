from model import resnet34
import os
import argparse
import torch
from torchvision import transforms
import json
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    # 有关数据集的读取
    parser.add_argument('--img_dir', type=str, default='F:/dataset/病害数据图片/街北高速/20240327150614/images',
                        help='输入图片的目录')
    # parser.add_argument('--save_dir', type=str, default=os.getcwd(),
    #                     help='生成txt文件的保存路径')
    parser.add_argument('--save_txt', type=str, default=os.path.join(os.getcwd(), "lanes_cls.txt"),
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
    json_path = 'log/height_center/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "log/height_center/resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    # 存储车道位置的list
    lanes_list = []
    # 对读取的图片路径进行排序
    img_path_list = [os.path.join(args.img_dir, i) for i in os.listdir(args.img_dir) if i.endswith(".jpg")]
    img_path_list.sort()
    with torch.no_grad():
        for img_path in img_path_list:
            img = Image.open(img_path)
            img = data_transform(img).unsqueeze(0)

            # predict class
            output = model(img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                lanes_list.append(class_indict[str(cla.numpy())])
    # 写入txt文件
    # filename = os.path.join(args.save_dir, 'lanes_cls.txt')
    # 写入整个列表到文件
    with open(args.save_txt, 'w') as file:
        file.write(str(lanes_list))


if __name__ == '__main__':
    args = parse_args()
    generate_txt(args)
