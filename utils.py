import os
from matplotlib import pyplot as plt
import scipy.signal


class loss_draw:
    def __init__(self, log_dir):
        self.log_dir = log_dir  # 用于保存loss相关文件的路径
        self.train_loss = []  # 用于储存训练集的loss
        self.val_loss = []  # 用于储存验证集的loss

    def drawing_loss(self, train_loss, val_loss):
        self.train_loss.append(train_loss)  # 将训练集的loss存在列表中
        self.val_loss.append(val_loss)  # 将验证集的loss存在列表中

        iters = range(1, len(self.train_loss) + 1)

        # 创建画布
        plt.figure()
        # 绘制loss和val_loss
        plt.plot(iters, self.train_loss, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val_accuracy')
        try:
            if len(self.train_loss) < 25:
                num = 5
            else:
                num = 15
            # 绘制平滑后的loss和val_loss
            # plt.plot(iters, scipy.signal.savgol_filter(self.train_loss, num, 3), 'green', linestyle='--', linewidth=2,
            #          label='smooth train loss')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
            #          label='smooth val loss')
        except:
            pass

        # 绘制其他的一些细节
        plt.grid(True)  # 是否带背景网格
        plt.xlabel('Epoch')  # x轴变量名称
        plt.ylabel('Loss')  # y轴变量名称
        plt.legend(loc="upper right")  # 在右上角绘制图例标签

        # 保存图片到所在路径
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变
        plt.close("all")  # 关闭 window，如果没有指定，则指当前 window。