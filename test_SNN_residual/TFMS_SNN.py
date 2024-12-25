import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

class ResidualLIFNode(neuron.LIFNode):
    def __init__(self, tau=2.0, surrogate_function=surrogate.ATan(), v_threshold=1.0):
        super(ResidualLIFNode, self).__init__(tau=tau, surrogate_function=surrogate_function, v_threshold=v_threshold)

    def forward(self, x):
        lif_output = super(ResidualLIFNode, self).forward(x)
        return x + lif_output  # 添加残差分支

class InceptionBlock1D_SNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sizes=[7], stride=1, groups=1, tau=0.15, v_threshold=2.0, dropout_rate=0.25):
        super(InceptionBlock1D_SNN, self).__init__()
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
                    nn.BatchNorm1d(out_channel),
                    ResidualLIFNode(tau=tau, surrogate_function=surrogate.ATan(), v_threshold=v_threshold)
                )
            )
        self.conv1x1 = nn.Conv1d(len(kernel_sizes) * out_channel, out_channel, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm1d(out_channel)
        self.lif1x1 = ResidualLIFNode(tau=tau, surrogate_function=surrogate.ATan(), v_threshold=v_threshold)

        self.residual = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel)
            )
        self.residual_lif = ResidualLIFNode(tau=tau, surrogate_function=surrogate.ATan(), v_threshold=v_threshold)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out_t = torch.cat(branch_outputs, dim=1)
        out_t = self.conv1x1(out_t)
        out_t = self.bn1x1(out_t)
        out_t = self.lif1x1(out_t)
        out_t += self.residual(x)
        out = self.residual_lif(out_t)
        return out

class TFMS_SNN_Case3(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.25, groups=8):
        super(TFMS_SNN_Case3, self).__init__()
        kernel_size = 7
        num_blocks = 3
        initial_filters = 8
        tau = 4.0
        v_threshold = 0.7

        self.groups = groups

        in_channels = 1
        self.conv1 = nn.Conv1d(in_channels, initial_filters, kernel_size=kernel_size, padding=3, stride=2)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.neuron1 = ResidualLIFNode(tau=tau, surrogate_function=surrogate.ATan(), v_threshold=v_threshold)

        self.freq_branch = self._build_branch(InceptionBlock1D_SNN, num_blocks, initial_filters, kernel_size, tau, v_threshold, dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(64, 32)
        self.neuron2 = ResidualLIFNode(tau=5.0, surrogate_function=surrogate.ATan(), v_threshold=0.1)

        self.fc3 = nn.Linear(32, num_classes)
        self.neuron3 = ResidualLIFNode(tau=5.0, surrogate_function=surrogate.ATan(), v_threshold=0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-1, 1)
                if m.bias is not None:
                    m.bias.data.uniform_(-1, 1)
            elif isinstance(m, nn.Conv1d):
                m.weight.data.uniform_(-1, 1)
                if m.bias is not None:
                    m.bias.data.uniform_(-1, 1)

    def _build_branch(self, block, num_blocks, initial_filters, kernel_size, tau, v_threshold, dropout_rate):
        layers = []
        for i in range(num_blocks):
            layers.append(block(initial_filters, initial_filters * 2, stride=2, groups=self.groups, tau=tau, v_threshold=v_threshold, dropout_rate=dropout_rate))
            initial_filters *= 2

        return nn.Sequential(*layers)

    def forward(self, freq_input):
        x = self.conv1(freq_input)
        x = self.bn1(x)
        x = self.neuron1(x)

        freq_features = self.freq_branch(x)
        out = self.avgpool(freq_features).squeeze(-1)
        out = self.fc2(out)
        out = self.neuron2(out)
        out = self.fc3(out)
        out = self.neuron3(out)

        return out

# from torchsummary import summary
# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = TFMS_SNN_Case3()
# model.to(device)

# summary(model, input_size=(1, 1024))
# inputs = torch.randn(1, 1, 1024).to(device)
# flops, params = profile(model, (inputs,))
# print('flops: ', flops, 'params: ', params)


