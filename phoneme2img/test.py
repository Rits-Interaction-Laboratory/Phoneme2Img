import torch
from torch.autograd import Variable
import torch.optim as optim
import re

def linear_model(x, W, b):
    return torch.matmul(x, W) + b

def format_params(timing, W, b):
    """見やすいように出力をフォーマット"""
    torch.set_printoptions(precision=4)
    print(f"--- {timing}")
    print(re.sub("\n|\t|       |, requires_grad=True|", "", f"    W = {W}, W_grad = {W.grad}"))
    print(re.sub("\n|\t|       |, requires_grad=True|", "", f"    b = {b}, b_grad = {b.grad}"))
    print("")

def x_format_params(timing, X, a):
    """見やすいように出力をフォーマット"""
    torch.set_printoptions(precision=4)
    print(f"--- {timing}")
    print(re.sub("\n|\t|       |, requires_grad=True|", "", f"    X = {X}, W_grad = {X.grad}"))
    print(re.sub("\n|\t|       |, requires_grad=True|", "", f"    a = {a}, b_grad = {a.grad}"))
    print("")

data = torch.Tensor([[1, 1], [3, 3]])
targets = torch.Tensor([10, 100])

W = Variable(torch.tensor([[0.5143], [-0.8160]]), requires_grad=True)
b = Variable(torch.tensor([-0.8217]), requires_grad=True)

X = Variable(torch.tensor([[-0.5490], [-0.0385]]), requires_grad=True)
a = Variable(torch.tensor([-0.2651]), requires_grad=True)


criterion = torch.nn.MSELoss()
optimizer = optim.Adam([W, b])
x_optimizer = optim.Adam([X, a])

for i, (sample, target) in enumerate(zip(data, targets)):
    print(f"loop: {i + 1}")
    format_params("init", W, b)
    x_format_params("x_init", X, a)

    # optimizer.zero_grad()
    # x_optimizer.zero_grad()
    # format_params("optimizer.zero_grad()", W, b)
    # x_format_params("x_optimizer.zero_grad()", X, a)

    output = linear_model(sample, W, b)
    x_output = linear_model(sample, X, a)
    loss = criterion(output.squeeze(), target)
    x_loss = criterion(x_output.squeeze(), target)
    total_loss = loss + x_loss
    total_loss.backward()
    format_params("loss.backward()", W, b)
    x_format_params("x_loss.backward()", X, a)

    optimizer.step()
    x_optimizer.step()
    format_params("optimizer.step()", W, b)
    x_format_params("x_optimizer.step()", X, a)

