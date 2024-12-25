import torch
import torch.nn as nn

# 定义一个简单的神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 40)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 创建一个模型实例
model = Net()

# 确保模型处于训练模式
model.train()

# 定义一个用于检查梯度的回调函数
def gradient_hook(module, grad_input, grad_output):
    print('输出梯度值：', grad_output)

# 遍历每个层，并注册回调函数
for name, module in model.named_modules():
    module.register_backward_hook(gradient_hook)

# 创建一个输入张量
input = torch.randn(1, 10)

# 前向传播
output = model(input)

# 反向传播
output.backward()
