import torch
import torch.nn as nn
import torch.cuda
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 更新模型结构，适配更大张量
class Conv3DTestModel(nn.Module):
    def __init__(self):
        super(Conv3DTestModel, self).__init__()
        self.conv = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((16, 16, 16))  # 输出形状固定为16³
        self.fc = nn.Linear(16 * 16 * 16 * 16, 1)  # 对应flatten维度

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 主执行函数
def main():
    model = Conv3DTestModel().to(device)
    outputs = []

    batch_size = 4
    input_shape = (batch_size, 8, 128, 128, 128)  # 更大输入张量

    print(f"Generating input of shape {input_shape}")

    for i in range(10):  # 循环10次，保留计算图
        x = torch.randn(input_shape, device=device)
        output = model(x)
        outputs.append(output)  # 不调用 .detach()，显存累积
        print(f"[{i+1}/10] Output shape: {output.shape}")
        time.sleep(0.2)

    # 显存信息
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    print(f"\n[CUDA Memory] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

if __name__ == "__main__":
    main()
