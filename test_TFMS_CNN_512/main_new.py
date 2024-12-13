import os
import sys
import gc
import msvcrt
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from resnet_vector_group import resnet18_vector_group, resnet10_vector_group
from TFMS_CNN import TFMS_CNN_Case3

from confusionMatrix import ConfusionMatrix

import argparse
import select
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

# from spikingjelly.activation_based import ann2snn

# from dataset_vector import data_set_split
# from dataset import data_set_split
from dataset_T_F import data_set_split

from torchsummary import summary
from sklearn.manifold import TSNE


def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # /root/miniconda3/bin/python /root/autodl-tmp/lzk/test/main.py --model_path 'ANN22_model.pth' --Train True --Test True --name ANN22
    parser = argparse.ArgumentParser(description="Train a localization model")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # 娣诲姞鍏朵粬鍙傛暟
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the model')
    # parser.add_argument('--runs', type=str, default='runs_Resnet10_model', help='Loss runing graph')
    parser.add_argument('--Train', type=bool, default=True, help='Switch wheater to train')
    parser.add_argument('--Test', type=bool, default=True, help='Test the test_loader use trained model')
    parser.add_argument('--TSNE', type=bool, default=False, help='The t-SNE Visualisation')
    # parser.add_argument('--spikingjelly_ann2snn', type=bool, default=False, help='Converter ann2snn')

    parser.add_argument('--class_num', type=int, default=10, help='class number')
    parser.add_argument('--model', type=str, default='TFMS_CNN_512', help='switch model')
    parser.add_argument('--image_path', type=str, default='F:\\D00\\data_set_1024_1024', help='dataset path')
    # parser.add_argument('--name', type=str, default='D01_1024_1024', help='project name')
    parser.add_argument('--T', type=int, default=50, help='sim steps')

    args = parser.parse_args()
    print(args)

    release_gpu_memory()# 释放GPU内存

    if sys.platform.startswith('linux'):
        system=0
    elif sys.platform.startswith('win'):
        system=1

    torch.manual_seed(42)

    # if you have GPU, the device will be cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # pattern = r"\\(D\d+)\\.*_(\d+_\d+)"
    # match = re.search(pattern, args.image_path)
    # name = f"{match.group(1)}_{match.group(2)}"
    folder_name = os.path.basename(os.path.dirname(args.image_path))  # 提取D00
    file_name = os.path.basename(args.image_path).replace("data_set_", "")  # 提取1024_1024
    name = f"{folder_name}_{file_name}"

    # folder_path = './' + args.name + '_bs' + str(args.batch_size) + '_lr' + str(args.lr)
    # the folder name is the train args, and some information about this train will be saved in this folder
    folder_path = './' + args.model + '_' + name + '_bs' + str(args.batch_size) + '_lr' + str(args.lr)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print('The folder is {}'.format(folder_path))
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print("The number of works is {}.".format(nw))

    # image_path = 'F:\\D01\\data_set_1024_1024'
    image_path = args.image_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_loader, validate_loader, test_loader, train_num, val_num, test_num, labels__ = data_set_split(image_path, batch_size, nw)
    print(labels__)


    # Update the model to fit your input size and number of classes
    if args.model == 'resnet18_pre':
        model = models.resnet18(weights=True)
        model.fc = nn.Linear(model.fc.in_features, args.class_num)
        # model = resnet18(num_classes=args.class_num)
    elif args.model == '1d_lw_resnet18':
        model = resnet18_vector_group(num_classes=args.class_num)
    elif args.model == '1d_lw_resnet10':
        model = resnet10_vector_group(num_classes=args.class_num)
    elif args.model == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, args.class_num)
    elif args.model == 'googlenet':
        model = models.googlenet(weights=True)
        model.fc = nn.Linear(model.fc.in_features, args.class_num)
    elif args.model == 'TFMS_CNN_512':
        model = TFMS_CNN_Case3(num_classes=args.class_num, dropout_rate=0.25)
    model.to(device)

    # This script is used to show the model structure
    # summary(model,input_size=(3,224))
    # exit()
    # # set first layer weight to ones
    # model.conv1.weight = torch.nn.Parameter(torch.ones_like(model.conv1.weight))
    # # print(model.state_dict()['conv1.weight'])

    # for param in model.conv1.parameters():
    #     param.requires_grad = False
    ##############################################

    if args.Train:
        print(f"Training with {args.epochs} epochs, batch size {batch_size}, learning rate {args.lr}")
        writer = SummaryWriter(folder_path + '/runs')

        # Define loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        epochs = args.epochs
        best_acc = 0.0
        save_path = folder_path + '/' + args.model_path
        train_steps = len(train_loader)
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                freq_seq, labels = data
                # img = torch.cat((time_seq, freq_seq), dim=-1).to(device)
                optimizer.zero_grad()
                outputs = model(freq_seq.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                train_bar.desc = "Train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            # Validation phase
            model.eval()
            acc = 0.0  # Accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    freq_seq, val_labels = val_data
                    # img = torch.cat((time_seq, freq_seq), dim=-1).to(device)
                    outputs = model(freq_seq.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    val_bar.desc = "Valid epoch[{}/{}]".format(epoch + 1, epochs)

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, val_accurate))
            
            scheduler.step(running_loss / train_steps)

            if val_accurate > best_acc:
                best_acc = val_accurate
                best_model_state = model.state_dict()

            writer.add_scalar('Loss/train', running_loss / train_steps, epoch)
            writer.add_scalar('Acc/val', val_accurate, epoch)

            if system == 1 and msvcrt.kbhit():  # 检查是否有键盘输入
                user_input = msvcrt.getch().decode('utf-8').strip()
                if user_input == 'q':
                    print("Stopping training after this epoch.")
                    break
            elif system == 0 and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:  # Check if 'q' was pressed
                user_input = sys.stdin.readline().strip()
                if user_input == 'q':
                    print("Stopping training after this epoch.")
                    break

        print('Finished Training')
        writer.close()
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_path)

    if args.Test:
        print('Testing')
        model_path = folder_path + '/' + args.model_path
        assert os.path.exists(model_path), "cannot find {} file".format(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        confusion = ConfusionMatrix(num_classes=args.class_num, labels=labels__)
        model.eval()
        with torch.no_grad():
            for val_data in tqdm(test_loader):#validate_loader test_loader
                freq_seq, val_labels = val_data
                # img = torch.cat((time_seq, freq_seq), dim=-1).to(device)
                outputs = model(freq_seq.to(device))
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
        confusion.plot(folder_path)
        confusion.summary()



    if args.TSNE:
        print('Testing')
        model_path = folder_path + '/' + args.model_path
        assert os.path.exists(model_path), "cannot find {} file".format(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        model.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for val_data in tqdm(test_loader):  # validate_loader test_loader
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # 假设模型的倒数第二层是特征提取层
                features = model.get_features(val_images.to(device), layer=['layer1', 'layer2', 'layer3', 'layer4'])  # 需要您根据模型实际修改
                features = features.cpu().numpy()
                all_features.append(features)
                all_labels.extend(val_labels.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        # Reshape all_features to have 2 dimensions
        all_features = all_features.reshape(all_features.shape[0], -1)
        all_labels = np.array(all_labels)

        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(all_features)

        cmaps = 'viridis'
        # 绘制t-SNE图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=all_labels, cmap=cmaps, alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Model Performance')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')

        # 获取viridis colormap
        cmap = cm.get_cmap(cmaps, len(labels__))

        # 创建自定义图例
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=cmap(i), markersize=10) for i, label in enumerate(labels__)]
        plt.legend(handles=legend_elements, ncol=1)  # 调整图例列数

        plt.savefig(os.path.join(folder_path, 'tsne_plot.png'))  # 保存图像
        plt.show()

        # writer = SummaryWriter(folder_path + '/graph')
        # writer.add_graph(model,val_images.to(device))
        # writer.close()

if __name__ == '__main__':
    main()

