import os
import sys
import json

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
import random
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image
from utils import loss_draw
import time


# 训练超参设置
def parse_args():
    parser = argparse.ArgumentParser()
    # 有关数据集的读取
    parser.add_argument('--image_path', type=str, default='dataset',
                        help='训练集和验证集的txt文件路径，底下包含train和val文件夹，内部通过不同文件夹来分类')
    parser.add_argument('--save_path', type=str, default='resNet34.pth',
                        help='保存模型路径')

    # 模型训练参数
    parser.add_argument('--input_shape', type=int, default=256,  # [1152, 2048]  # [864, 1536]  # [576, 1024]
                        help='网络的输入分辨率大小[h, w],一定要为32的倍数')
    parser.add_argument('--crop_size', type=int, default=[1216, 1824],
                        help='网络的输入分辨率大小[h, w],一定要为32的倍数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='bs')

    # 预训练模型配置
    parser.add_argument('--model_weight_path', type=str,
                        default='./resnet34-pre.pth',
                        help='预训练权重路径 download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth')

    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    # parser.add_argument('--seed', type=int, default=20230404,
    #                     help='训练时的随机种子，如果设为-1则随机')

    # 学习率相关
    parser.add_argument('--lr', type=float, default=0.001,
                        help='模型初始化学习率')
    # parser.add_argument('--optimizer_type', type=str, default='sgd',
    #                     help='当使用adam优化器时建议设置  Init_lr=1e-3，当使用SGD优化器时建议设置Init_lr=1e-2')
    # parser.add_argument('--momentum', type=float, default=0.937,
    #                     help='优化器内部使用到的momentum参数')
    # parser.add_argument('--weight_decay', type=float, default=5e-4,
    #                     help='权值衰减，可防止过拟合，adam会导致weight_decay错误，使用adam时建议设置为0,sgd的default=5e-4')
    # parser.add_argument('--lr_decay_type', type=str, default='cos',
    #                     help='使用到的学习率下降方式，可选的有step、cos')

    # 有关数据增强
    # parser.add_argument('--flip', type=bool, default=False,
    #                     help='使用水平翻转数据增强')

    return parser.parse_args()


def hsv_to_rgb(hsv_image):
    # 将 HSV 图像转换回 RGB 颜色空间
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return rgb_image


# 随机hsv数据增强
def random_hsv_transform(image, h_gain=10, s_gain=0.5, v_gain=0.5):
    # 将 PIL 图像转换为 NumPy 数组
    image_np = np.array(image)
    # 确保图像是 RGB 格式
    if image_np.shape[2] == 4:  # 如果有 alpha 通道，将其去除
        image_np = image_np[:, :, :3]
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    # 将 RGB 转换为 HSV并分离 HSV 通道
    hue, sat, val = cv2.split(cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV))
    dtype = image_np.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    # 合并hsv通道
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    # 将 HSV 转换回 RGB
    rgb_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # 将 NumPy 数组转换回 PIL 图像
    final_image = Image.fromarray(rgb_image.astype(np.uint8))
    return final_image


def fixed_height_random_width_crop(image, height_scale=(0.5, 1.0), scale_range=(0.65, 1.0)):
    """
       对图像进行固定高度、宽度按比例随机裁剪，同时保持高度居中。

       参数:
       - image: PIL图像对象。
       - original_height: 原始图像的高度。
       - scale_range: 一个元组，表示宽度比例的范围 (最小值, 最大值)。

       返回:
       - 裁剪后的PIL图像。
       """
    original_width, original_height = image.size
    # 计算目标宽度的比例
    target_width = int(original_width * random.uniform(*scale_range))
    # 计算目标高度的比例
    target_height = int(original_height * random.uniform(*height_scale))
    # 计算左右的边界，随机获取
    left = random.randint(0, original_width - target_width)
    right = left + target_width
    # 计算上下边界，保持高度居中
    top = (original_height - target_height) // 2
    bottom = top + target_height
    image_crop = image.crop((left, top, right, bottom))

    return image_crop.resize((256, 256))


# 高斯噪声
def add_gaussian_noise(image, mean=0, std=10):
    # 将 PIL 图像转换为 NumPy 数组
    image_np = np.array(image)
    assert isinstance(image_np, np.ndarray), "image must be a numpy array"
    noisy_image = image_np + np.random.normal(mean, std, image_np.shape)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # 确保值在0-255内
    PIL_image = Image.fromarray(noisy_image)
    return PIL_image


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # fixed_height_random_width_crop,
            transforms.RandomResizedCrop(args.input_shape, scale=(0.5, 1.0), ratio=(1, 1.19)),
            # transforms.CenterCrop(args.crop_size),
            # transforms.Resize(args.input_shape),
            random_hsv_transform,
            add_gaussian_noise,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(args.input_shape),
            # transforms.CenterCrop(args.input_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join("dataset", "flower_data")  # flower data set path

    assert os.path.exists(args.image_path), "{} path does not exist.".format(args.image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(args.image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    # 相关文件存储路径
    save_dir = os.path.join('log', time.strftime('%Y_%m_%d_%H_%M', time.localtime()))
    if not os.path.exists(save_dir):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(save_dir)
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(save_dir, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(args.image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    assert os.path.exists(args.model_weight_path), "file {} does not exist.".format(args.model_weight_path)
    net.load_state_dict(torch.load(args.model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # 类别数量设置
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    best_acc = 0.0
    train_steps = len(train_loader)
    # 实例化loss曲线绘制的类
    draw_loss = loss_draw(save_dir)
    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        draw_loss.drawing_loss(running_loss / train_steps, val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(save_dir, args.save_path))

    print('Finished Training')


if __name__ == '__main__':
    args = parse_args()
    main(args)
