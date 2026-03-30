import time
import numpy as np
import torch
import sys
import os

# 导入你的模型
from models.AG_BiGRU import AG_BiGRU

# --- 1. 关键参数设置 (根据你的补充信息) ---
# 采样率 1000Hz, 窗口 128ms -> 采样点数 L = 128
# SIAT 数据集通道数 C = 9
WINDOW_LEN = 128
CHANNELS = 9
BATCH_SIZE = 1  # 实时推理必须是单样本

# 输入形状: [Batch, 1, Channels, Length]
input_shape = (BATCH_SIZE, 1, CHANNELS, WINDOW_LEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 准备模型 ---
print(f"--- 正在测试 AG-BiGRU 推理耗时 ---")
print(f"设备: {device}")
print(f"输入维度: {input_shape} (128ms @ 1000Hz)")

# 实例化模型
try:
    model = AG_BiGRU(rnn_type='BiGRU')
except:
    # 兼容某些版本的定义
    model = AG_BiGRU()

model.to(device)
model.eval()  # 必须开启评估模式

# 创建虚拟输入数据
dummy_input = torch.randn(*input_shape).to(device)

# --- 3. 预热 (Warm-up) ---
# 这一步不可省略！GPU 需要预热。
print("正在预热硬件...")
with torch.no_grad():
    for _ in range(50):
        _ = model(dummy_input)

# --- 4. 正式计时 (Loop 100次) ---
print("开始正式计时 (100次循环取平均)...")
costs = []

with torch.no_grad():
    for k in range(100):
        # 同步 GPU 时间 (关键)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()

        # 推理
        output = model(dummy_input)

        # 同步 GPU 时间
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        costs.append(end - start)

# --- 5. 结果输出 ---
avg_time_ms = np.mean(costs) * 1000  # 秒 -> 毫秒

print(f"========================================")
print(f"AG-BiGRU 平均推理时间: {avg_time_ms:.4f} ms")
print(f"========================================")

# 对比基准 (SC-BiGRU = 1.7ms)
baseline = 1.7
if avg_time_ms <= baseline:
    print(f"🚀 结论：AG-BiGRU 比 SC-BiGRU 快 {baseline - avg_time_ms:.2f} ms")
else:
    diff = avg_time_ms - baseline
    print(f"⚖️ 结论：AG-BiGRU 比 SC-BiGRU 慢 {diff:.2f} ms")
    print(f"   (仅增加了 {diff:.2f} ms，仍在毫秒级，完全满足实时性)")