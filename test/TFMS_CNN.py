import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, dropout_rate=0.25, **kwargs):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=7, stride=stride, padding=3, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=7, stride=1, padding=3, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        out = self.dropout(out)
        return out


class TFMS_CNN_Case3(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.25, groups=16):
        super(TFMS_CNN_Case3, self).__init__()
        
        # 卷积块超参数
        kernel_size = 7
        num_blocks = 3
        initial_filters = 16 # in paper is 16

        self.groups = groups
        
        # 时域分支
        self.time_branch = self._build_branch(BasicBlock1D, num_blocks, kernel_size, initial_filters, dropout_rate)
        
        # 频域分支
        self.freq_branch = self._build_branch(BasicBlock1D, num_blocks, kernel_size, initial_filters, dropout_rate)
        
        # 全连接层
        self.fc_layer = nn.Sequential(
            nn.Linear(initial_filters * (2**4), 128),  # 每个分支的最终输出特征连接后输入到全连接层
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类层
        self.classifier = nn.Linear(128, num_classes)
    
    def _build_branch(self, block, num_blocks, kernel_size, initial_filters, dropout_rate):
        layers = []
        in_channels = 1  # 假设输入信号是单通道
        # for i in range(num_blocks):
        #     out_channels = initial_filters * (2**i)
        #     layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=2))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.MaxPool1d(2))
        #     layers.append(nn.Dropout(dropout_rate))
        #     in_channels = out_channels
        # return nn.Sequential(*layers)

        layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, initial_filters, kernel_size, padding=3, stride=2),
                nn.BatchNorm1d(initial_filters),
                nn.ReLU()
            )
        )
        for i in range(num_blocks):
            # initial_filters = initial_filters * (2**i)
            downsample = None
            
            downsample = nn.Sequential(
                nn.Conv1d(initial_filters, initial_filters * 2, kernel_size=kernel_size, padding=3, stride=2, bias=False, groups=self.groups),
                nn.BatchNorm1d(initial_filters * 2)
            )
            layers.append(block(initial_filters, initial_filters * 2, stride=2, downsample=downsample, groups=self.groups, dropout_rate=dropout_rate))
            layers.append(block(in_channel=initial_filters * 2, out_channel=initial_filters * 2, groups=self.groups))
            initial_filters *= 2

            

        # downsample = nn.Sequential(
        #     nn.Conv1d(32, 64, kernel_size=1, stride=2, bias=False, groups=self.groups),
        #     nn.BatchNorm1d(64)
        # )
        # layers.append(block(32, 64, stride=2, downsample=downsample, groups=self.groups))
        
        # layers.append(block(in_channel=64, out_channel=64, groups=self.groups))
        # # layers.append(block(in_channel=64, out_channel=64, groups=self.groups))

        # downsample = nn.Sequential(
        #     nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False, groups=self.groups),
        #     nn.BatchNorm1d(128)
        # )
        # layers.append(block(64, 128, stride=2, downsample=downsample, groups=self.groups))
        
        # layers.append(block(in_channel=128, out_channel=128, groups=self.groups))
        # # layers.append(block(in_channel=128, out_channel=128, groups=self.groups))

        # downsample = nn.Sequential(
        #     nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False, groups=self.groups),
        #     nn.BatchNorm1d(256)
        # )
        # layers.append(block(128, 256, stride=2, downsample=downsample, groups=self.groups))
        
        # layers.append(block(in_channel=256, out_channel=256, groups=self.groups))
        # # layers.append(block(in_channel=256, out_channel=256, groups=self.groups))

        return nn.Sequential(*layers)


    def forward(self, input):
        # 时域和频域分支
        time_input = input[:,:,0:1024]
        freq_input = input[:,:,1024:2048]
        time_features = self.time_branch(time_input)
        freq_features = self.freq_branch(freq_input)
        
        # 全局最大池化层
        time_features = nn.functional.adaptive_max_pool1d(time_features, 1).squeeze(-1)
        freq_features = nn.functional.adaptive_max_pool1d(freq_features, 1).squeeze(-1)
        
        # 特征融合
        combined_features = torch.cat((time_features, freq_features), dim=1)
        
        # 全连接层和分类
        out = self.fc_layer(combined_features)
        out = self.classifier(out)
        return out


# from torchsummary import summary
# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = TFMS_CNN_Case3()
# model.to(device)

# # # summary(model, input_size=(1, 2048))
# inputs = torch.randn(1, 1, 2048).to(device)
# flops, params = profile(model, (inputs,))
# print('flops: ', flops, 'params: ', params)

# # 示例输入
# time_input = torch.randn(16, 1, 1024)  # Batch size=16, 单通道，长度=5000
# freq_input = torch.randn(16, 1, 1024)

# # 初始化网络
# model = TFMS_CNN_Case3()

# input = torch.cat((time_input, freq_input), dim=-1)
# # 前向传播
# output = model(input)
# print("输出形状:", output.shape)  # 输出形状应为 [16, 10]
