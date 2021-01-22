# @author : Panda
import matplotlib.pyplot as plt
import numpy as np
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
lr = 0.01


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 构造对象，能够自动反向传播torch.nn.Linear(input_features输入一个样本的维度,output_f,bias=True)
        self.linear = torch.nn.Linear(1, 1)

    # 对forward的overwrite
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# MSE(size_average=True,reduce=True)
criterion = torch.nn.MSELoss(size_average=False)

# 对模型权重进行学习，对linear的两个权重，w and b.   model.p是万能写法
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# 以下为损失优化函数的多个测试
#optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("--------------------------------")
print("w:", model.linear.weight.item())
print("b:", model.linear.bias.item())

x = np.linspace(0, 10, 200)
x_t = torch.tensor(x).view(200, 1)
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c="r")
plt.grid()
plt.show()
