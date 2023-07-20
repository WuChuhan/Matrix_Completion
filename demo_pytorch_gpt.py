import torch
import torch.nn as nn
import torch.optim as optim


# 构建补全模型
class MatrixCompletionModel(nn.Module):
    def __init__(self, input_shape, rank):
        super(MatrixCompletionModel, self).__init__()
        self.rank = rank
        self.embedding = nn.Parameter(torch.randn(input_shape[-1], rank))
        self.weight = nn.Parameter(torch.randn(input_shape[:-1] + (rank,)))

    def forward(self):
        return torch.matmul(self.weight, self.embedding.t())


# 定义补全函数
def matrix_completion(input_tensor, rank, lr=0.01, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MatrixCompletionModel(input_tensor.shape, rank)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 转换为2D矩阵形式，以便进行补全
    input_tensor = input_tensor.view(-1, input_tensor.shape[-1])

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        completed_tensor = model()
        loss = loss_fn(completed_tensor, input_tensor)
        loss.backward()

        optimizer.step()

        if (iteration + 1) % 10 == 0:
            print(f"Iteration [{iteration+1}/{num_iterations}], Loss: {loss.item()}")

    # 补全后的矩阵
    completed_tensor = completed_tensor.view(input_tensor.shape[:-1] + (model.embedding.shape[-1],))
    return completed_tensor


# 示例使用
input_tensor = torch.randn(4, 3, 80, 80, 4)
completed_tensor = matrix_completion(input_tensor, rank=2, lr=0.01, num_iterations=100)
print("Completed Tensor Shape:", completed_tensor.shape)
