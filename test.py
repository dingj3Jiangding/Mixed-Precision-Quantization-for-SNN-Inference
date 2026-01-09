import torch
from spikingjelly.activation_based import neuron

# 1. 设置：创建一个最简单的 IF (Integrate-and-Fire) 神经元
# v_threshold=1.0 表示电压超过 1.0 就会发放脉冲
net = neuron.IFNode(v_threshold=1.0)

# 2. 输入：模拟一个电压输入
# 这里的 1.5 大于阈值 1.0，理论上应该会激发脉冲
x = torch.tensor([1.5]) 

# 3. 运行：把输入喂给神经元
y = net(x)

# 4. 打印结果
print(f"输入电压: {x.item()}")
print(f"神经元输出 (脉冲): {y.item()}")
print(f"当前膜电位 (V): {net.v.item()}")

if y.item() == 1.0:
    print("\n✅ 测试成功！神经元成功发放了一个脉冲！")
else:
    print("\n❌ 测试异常，请检查代码。")