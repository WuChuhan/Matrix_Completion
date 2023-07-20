import torch
import torch.nn as nn
import torch.optim as optim

"""
input_tensor = torch.randn(4, 3, 4)
input_tensor_zero = torch.zeros(4, 3, 2)
input_tensor_total = torch.cat((input_tensor, input_tensor_zero), dim=-1)
input_tensor_2D = input_tensor_total.view(-1, input_tensor_total.shape[-1])
c = input_tensor_2D.shape
print(c)

a = torch.randn(c[-1], 2)
b = torch.randn(c[:-1] + (2,))
print(a)
print(a.shape)
print(b)
print(b.shape)
"""
input_tensor = torch.randn(4, 3, 4)
a, b, c = input_tensor.shape
print(a)
print(b)
print(c)