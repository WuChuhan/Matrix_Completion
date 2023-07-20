import torch
import torch.nn as nn
import torch.optim as optim


# 构建补全模型
class MatrixCompletionModel(nn.Module):
    def __init__(self, input_shape, rank):
        super(MatrixCompletionModel, self).__init__()
        self.rank = rank
        self.embedding = nn.Parameter(torch.rand(input_shape[-1], rank))
        self.weight = nn.Parameter(torch.rand(input_shape[:-1] + (rank,)))

    def forward(self):
        return torch.matmul(self.weight, self.embedding.t())


# 定义补全函数
def matrix_completion(input_tensor, rank, lr=0.01, num_iterations=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor_2D = input_tensor.view(-1, input_tensor.shape[-1]).to(device)

    model = MatrixCompletionModel(input_tensor_2D.shape, rank)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 转换为2D矩阵形式，以便进行补全
    # input_tensor_2 = input_tensor.view(-1, input_tensor.shape[-1])

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        completed_tensor = model()
        loss = loss_fn(completed_tensor, input_tensor_2D)
        loss.backward()

        optimizer.step()

        if (iteration + 1) % 10 == 0:
            print(f"Iteration [{iteration+1}/{num_iterations}], Loss: {loss.item()}")

    # 补全后的矩阵
    # completed_tensor_new = completed_tensor.view(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4], model.embedding.shape[0])
    input_tensor_new = input_tensor_2D[..., :4]
    completed_tensor_new = torch.cat((input_tensor_new, completed_tensor[..., -2:]), dim=-1)

    return completed_tensor_new


# 示例使用
input_tensor = torch.rand(4, 3, 80, 80, 4)
input_tensor_zero = torch.zeros(4, 3, 80, 80, 2)
input_tensor_total = torch.cat((input_tensor, input_tensor_zero), dim=-1)

completed_tensor = matrix_completion(input_tensor_total, rank=2, lr=0.01, num_iterations=10000)
print("Completed Tensor Shape:", completed_tensor.shape)
print(input_tensor)
print(completed_tensor)
split_tensor = torch.split(completed_tensor, 4)
reshaped_tensor = torch.stack(split_tensor).reshape(4, 3, 80, 80, 6)
print(reshaped_tensor)
print(completed_tensor.shape)
print(reshaped_tensor.shape)