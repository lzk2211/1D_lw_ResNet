import torch
import torch.nn as nn


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sizes=[5, 9], stride=1, groups=1, dropout_rate=0.25):
        super(InceptionBlock1D, self).__init__()
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                )
            )
        self.conv1x1 = nn.Conv1d(len(kernel_sizes) * out_channel, out_channel, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 残差连接
        self.residual = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel)
            )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.conv1x1(out)
        out = self.bn1x1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 添加残差连接
        out += self.residual(x)
        out = self.relu(out)
        
        return out


class TFMS_CNN_Case3(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.25, groups=8):
        super(TFMS_CNN_Case3, self).__init__()
        
        # 卷积块超参数
        kernel_size = 7
        num_blocks = 3
        initial_filters = 8

        self.groups = groups
        
        # 频域分支
        self.freq_branch = self._build_branch(InceptionBlock1D, num_blocks, initial_filters, kernel_size, dropout_rate)
        
        # 全连接层
        self.fc_layer = nn.Sequential(
            nn.Linear(initial_filters * (2**3), 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类层
        self.classifier = nn.Linear(64, num_classes)
    
    def _build_branch(self, block, num_blocks, initial_filters, kernel_size, dropout_rate):
        layers = []
        in_channels = 1  # 假设输入信号是单通道

        layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, initial_filters, kernel_size=kernel_size, padding=3, stride=2),
                nn.BatchNorm1d(initial_filters),
                nn.ReLU()
            )
        )
        for i in range(num_blocks):
            downsample = nn.Sequential(
                nn.Conv1d(initial_filters, initial_filters * 2, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(initial_filters * 2)
            )
            layers.append(block(initial_filters, initial_filters * 2, stride=2, groups=self.groups, dropout_rate=dropout_rate))
            initial_filters *= 2

        return nn.Sequential(*layers)

    def forward(self, freq_input):
        freq_features = self.freq_branch(freq_input)
        
        # 全局最大池化层
        freq_features = nn.functional.adaptive_max_pool1d(freq_features, 1).squeeze(-1)
                
        # 全连接层和分类
        out = self.fc_layer(freq_features)
        out = self.classifier(out)
        return out


from torchsummary import summary
from thop import profile
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TFMS_CNN_Case3()
model.to(device)

summary(model, input_size=(1, 1024))
inputs = torch.randn(1, 1, 1024).to(device)
flops, params = profile(model, (inputs,))
print('flops: ', flops, 'params: ', params)
    