import os
import sys
import gc
# import msvcrt
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from resnet_vector_group import resnet18_vector_group, resnet10_vector_group
from TFMS_CNN import TFMS_CNN_Case3
from TFMS_SNN import TFMS_SNN_Case3

from confusionMatrix import ConfusionMatrix

import argparse
import select
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from spikingjelly.activation_based import ann2snn
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from torch.cuda import amp

# from dataset_vector import data_set_split
# from dataset import data_set_split
from dataset_T_F import data_set_split

from torchsummary import summary
from sklearn.manifold import TSNE

import time

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
    return correct / total if T is None else corrects / total

def main():
    # /root/miniconda3/bin/python /root/autodl-tmp/lzk/test/main.py --model_path 'ANN22_model.pth' --Train True --Test True --name ANN22
    parser = argparse.ArgumentParser(description="Train a localization model")
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # 娣诲姞鍏朵粬鍙傛暟
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the model')
    # parser.add_argument('--runs', type=str, default='runs_Resnet10_model', help='Loss runing graph')
    parser.add_argument('--Train', type=bool, default=False, help='Switch wheater to train')
    parser.add_argument('--Test', type=bool, default=True, help='Test the test_loader use trained model')
    parser.add_argument('--TSNE', type=bool, default=False, help='The t-SNE Visualisation')
    parser.add_argument('--spikingjelly_ann2snn', type=bool, default=False, help='Converter ann2snn')

    parser.add_argument('--class_num', type=int, default=10, help='class number')
    parser.add_argument('--model', type=str, default='TFMS_CNN_inception_resnet_s7', help='switch model')
    parser.add_argument('--image_path', type=str, default='/media/server125/SSD1/D00/data_set_1024_1024', help='dataset path')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    elif args.model == 'TFMS_CNN_inception_resnet':
        model = TFMS_CNN_Case3(num_classes=args.class_num, dropout_rate=0.25)
    elif args.model == 'TFMS_CNN_inception_resnet_s7':
        model = TFMS_SNN_Case3(num_classes=args.class_num, dropout_rate=0.25)
    model = nn.DataParallel(model)# two GPUs
    model.to(device)

    encoder = encoding.PoissonEncoder()

    # This script is used to show the model structure
    # summary(model,input_size=(3,224))
    # exit()
    # # set first layer weight to ones
    # model.conv1.weight = torch.nn.Parameter(torch.ones_like(model.conv1.weight))
    # # print(model.state_dict()['conv1.weight'])

    # for param in model.conv1.parameters():
    #     param.requires_grad = False
    ##############################################
    scaler = amp.GradScaler()
    if args.Train:
        print(f"Training with {args.epochs} epochs, batch size {batch_size}, learning rate {args.lr}")
        writer = SummaryWriter(folder_path + '/runs')
        # in_ = torch.rand(1, 1, 1024).to(device)
        # writer.add_graph(model, encoder(in_))

        # Define loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        epochs = args.epochs
        best_acc = 0.0
        save_path = folder_path + '/' + args.model_path
        train_steps = len(train_loader)
        # Training SNN model
        for epoch in range(epochs):
            # start_time = time.time()
            # Training phase
            model.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0

            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                freq_seq, labels = data
                # img = torch.cat((time_seq, freq_seq), dim=-1).to(device)
                optimizer.zero_grad()
                freq_seq = freq_seq.to(device)
                labels = labels.to(device)
                label_onehot = F.one_hot(labels, args.class_num).float()

                outputs = 0.

                # 计算每个批次的最小值和最大值
                min_vals = freq_seq.min(dim=2, keepdim=True)[0]
                max_vals = freq_seq.max(dim=2, keepdim=True)[0]

                # 进行归一化
                freq_seq = (freq_seq - min_vals) / (max_vals - min_vals)

                for t in range(args.T):
                    encoded_freq_seq = encoder(freq_seq)
                    outputs += model(encoded_freq_seq)
                outputs /= args.T
                loss = loss_function(outputs, label_onehot)

                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 观察梯度
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                #     writer.add_histogram(name + '_grad', param.grad.cpu().data.numpy(), epoch)

                train_samples += labels.numel()
                train_loss += loss.item() * labels.numel()
                train_acc += (outputs.argmax(1) == labels).float().sum().item()

                functional.reset_net(model)

                # Print statistics
                # running_loss += loss.item()
                train_bar.desc = "Train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, train_loss / train_samples)
            
            train_loss /= train_samples # add to tensorboard
            train_acc /= train_samples

            # Validation phase
            model.eval()
            # test_loss = 0
            test_acc = 0
            test_samples = 0
            # acc = 0.0  # Accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    freq_seq, val_labels = val_data
                    freq_seq = freq_seq.to(device)
                    val_labels = val_labels.to(device)
                    label_onehot = F.one_hot(val_labels, args.class_num).float()
                    outputs = 0.

                    # 计算每个批次的最小值和最大值
                    min_vals = freq_seq.min(dim=2, keepdim=True)[0]
                    max_vals = freq_seq.max(dim=2, keepdim=True)[0]

                    # 进行归一化
                    freq_seq = (freq_seq - min_vals) / (max_vals - min_vals)

                    for t in range(args.T):
                        encoded_freq_seq = encoder(freq_seq)
                        outputs += model(encoded_freq_seq)
                    outputs = outputs / args.T
                    # loss = loss_function(outputs, label_onehot)

                    test_samples += val_labels.numel()
                    # test_loss += loss.item() * val_labels.numel()
                    test_acc += (outputs.argmax(1) == val_labels).float().sum().item()

                    val_bar.desc = "Valid epoch[{}/{}] acc:{:.3f}".format(epoch + 1, epochs, test_acc / test_samples)
                    functional.reset_net(model)

            test_acc = test_acc / test_samples
            # test_loss = test_loss / test_samples

            print('[epoch %d] train_loss: %.3f  train_acc: %.3f  test_acc: %.3f' % 
                  (epoch + 1, train_loss, train_acc, test_acc) )
            
            scheduler.step(train_loss)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_state = model.state_dict()

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)
            # writer.add_scalar('Loss/val', test_loss, epoch)
            writer.add_scalar('Acc/val', test_acc, epoch)

            # if epoch % 10 == 0:
            #     checkpoint = {
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'epoch': epoch,
            #         'loss': loss.item()
            #     }
            #     torch.save(checkpoint, folder_path + '/pth/' + f'checkpoint_epoch_{epoch}.pth')


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

        # train_time = time.time()
        # train_speed = train_samples / (train_time - start_time)
        # print(f'train speed ={train_speed: .4f} images/s')

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
        test_samples = 0
        test_acc = 0
        start_time = time.time()
        with torch.no_grad():
            for test_data in tqdm(test_loader):#validate_loader test_loader
                freq_seq, test_labels = test_data
                freq_seq = freq_seq.to(device)
                test_labels = test_labels.to(device)
                label_onehot = F.one_hot(test_labels, args.class_num).float()
                outputs = 0.

                # 计算每个批次的最小值和最大值
                min_vals = freq_seq.min(dim=2, keepdim=True)[0]
                max_vals = freq_seq.max(dim=2, keepdim=True)[0]

                # 进行归一化
                freq_seq = (freq_seq - min_vals) / (max_vals - min_vals)

                for t in range(args.T):
                    encoded_freq_seq = encoder(freq_seq)
                    outputs += model(encoded_freq_seq)
                outputs = outputs / args.T

                test_samples += test_labels.numel()
                test_acc += (outputs.argmax(1) == test_labels).float().sum().item()
                functional.reset_net(model)

                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())

        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_acc /= test_samples
        print(f'test speed ={test_speed: .4f} images/s')
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

    if args.spikingjelly_ann2snn:
        T = args.T
        model = model.module
        
        print('---------------------------------------------')
        print('Converting using MaxNorm')
        model_converter = ann2snn.Converter(mode='max', dataloader=train_loader)
        snn_model = model_converter(model)
        print('Simulating...')
        mode_max_accs = val(snn_model, device, test_loader, T=T)
        print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_max_accs[-1]))

        print('---------------------------------------------')
        print('Converting using RobustNorm')
        model_converter = ann2snn.Converter(mode='99.9%', dataloader=train_loader)
        snn_model = model_converter(model)
        print('Simulating...')
        mode_robust_accs = val(snn_model, device, test_loader, T=T)
        print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_robust_accs[-1]))

        print('---------------------------------------------')
        print('Converting using 1/2 max(activation) as scales...')
        model_converter = ann2snn.Converter(mode=1.0 / 2, dataloader=train_loader)
        snn_model = model_converter(model)
        print('Simulating...')
        mode_two_accs = val(snn_model, device, test_loader, T=T)
        print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_two_accs[-1]))

        print('---------------------------------------------')
        print('Converting using 1/3 max(activation) as scales')
        model_converter = ann2snn.Converter(mode=1.0 / 3, dataloader=train_loader)
        snn_model = model_converter(model)
        print('Simulating...')
        mode_three_accs = val(snn_model, device, test_loader, T=T)
        print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_three_accs[-1]))

        print('---------------------------------------------')
        print('Converting using 1/4 max(activation) as scales')
        model_converter = ann2snn.Converter(mode=1.0 / 4, dataloader=train_loader)
        snn_model = model_converter(model)
        print('Simulating...')
        mode_four_accs = val(snn_model, device, test_loader, T=T)
        print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_four_accs[-1]))

        print('---------------------------------------------')
        print('Converting using 1/5 max(activation) as scales')
        model_converter = ann2snn.Converter(mode=1.0 / 5, dataloader=train_loader)
        snn_model = model_converter(model)
        print('Simulating...')
        mode_five_accs = val(snn_model, device, test_loader, T=T)
        print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_five_accs[-1]))

if __name__ == '__main__':
    main()

