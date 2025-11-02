import matplotlib.pyplot as plt
import torch
from spherenet import SphereConv2d
import numpy as np

def visualize_sampling_grid(grid):
    """
    可视化采样网格。

    参数：
     grid: 形状为 (1, H*Kh, W*Kw, 2) 的张量
    """
    # 将 grid 转换为 numpy 数组
    grid_np = grid.squeeze(0).cpu().numpy()  # (H*Kh, W*Kw, 2)
    
    # 提取 x 和 y 坐标
    x_coords = grid_np[:, :, 0]
    y_coords = grid_np[:, :, 1]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=0.5, c='blue')
    plt.title('Sampling Grid Visualization')
    plt.xlabel('Longitude (normalized)')
    plt.ylabel('Latitude (normalized)')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()
    
def visualize_sampling_grid_at_position(grid, H, W, Kh, Kw, h_idx, w_idx, title='Sampling Grid at Position'):
    """
    可视化输入图像中特定位置 (h_idx, w_idx) 的采样网格。

    参数：
    - grid: 形状为 (1, H*Kh, W*Kw, 2) 的张量
    - H, W: 输入图像的高度和宽度
    - Kh, Kw: 卷积核的高度和宽度
    - h_idx, w_idx: 要可视化的输入图像位置索引
    - title: 图像标题
    """
    # 将 grid 重新调整形状为 (H, Kh, W, Kw, 2)
    grid_np = grid.squeeze(0).cpu().numpy()  # (H*Kh, W*Kw, 2)
    grid_np = grid_np.reshape(H, Kh, W, Kw, 2)  # (H, Kh, W, Kw, 2)
    grid_np = grid_np.transpose(0, 2, 1, 3, 4)  # (H, W, Kh, Kw, 2)

    # 检查 h_idx 和 w_idx 是否在有效范围内
    assert 0 <= h_idx < H, f"h_idx {h_idx} 超出范围 (0, {H-1})"
    assert 0 <= w_idx < W, f"w_idx {w_idx} 超出范围 (0, {W-1})"

    # 提取指定位置的采样点
    sample_points = grid_np[h_idx, w_idx]  # (Kh, Kw, 2)

    x_coords = sample_points[:, :, 0]
    y_coords = sample_points[:, :, 1]

    plt.figure(figsize=(6, 6))
    plt.pcolormesh(x_coords, y_coords, np.zeros_like(x_coords), shading='auto', edgecolors='k')
    plt.title(f'{title} at (h={h_idx}, w={w_idx})')
    plt.xlabel('Longitude (normalized)')
    plt.ylabel('Latitude (normalized)')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()
    
def visualize_sampling_grid_mesh(grid, H, W, Kh, Kw):
    """
    可视化采样网格的变形网格。

    参数：
    - grid: 形状为 (1, H*Kh, W*Kw, 2) 的张量
    - H, W: 输入图像的高度和宽度
    - Kh, Kw: 卷积核的高度和宽度
    """
    grid_np = grid.squeeze(0).cpu().numpy()
    grid_np = grid_np.reshape(H, Kh, W, Kw, 2)
    grid_np = grid_np.transpose(0, 2, 1, 3, 4)  # (H, W, Kh, Kw, 2)
    
    # 仅可视化中心位置的采样点
    center_h = H // 2
    center_w = W // 2
    sample_points = grid_np[center_h, center_w]  # (Kh, Kw, 2)
    
    x_coords = sample_points[:, :, 0]
    y_coords = sample_points[:, :, 1]
    
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(x_coords, y_coords, np.zeros_like(x_coords), shading='auto', edgecolors='k')
    plt.title('Deformed Sampling Grid at Center Position')
    plt.xlabel('Longitude (normalized)')
    plt.ylabel('Latitude (normalized)')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()
    
def visualize_sampling_grid_detailed(grid, H, W, Kh, Kw, h_idx, w_idx, title='Sampling Grid at Position'):
    """
    可视化输入图像中特定位置 (h_idx, w_idx) 的采样网格，清晰展示采样点并勾勒采样框。

    参数：
    - grid: 形状为 (1, H*Kh, W*Kw, 2) 的张量
    - H, W: 输入图像的高度和宽度
    - Kh, Kw: 卷积核的高度和宽度
    - h_idx, w_idx: 要可视化的输入图像位置索引
    - title: 图像标题
    """
    # 将 grid 重新调整形状为 (H, Kh, W, Kw, 2)
    grid_np = grid.squeeze(0).cpu().numpy()  # (H*Kh, W*Kw, 2)
    grid_np = grid_np.reshape(H, Kh, W, Kw, 2)  # (H, Kh, W, Kw, 2)
    grid_np = grid_np.transpose(0, 2, 1, 3, 4)  # (H, W, Kh, Kw, 2)

    # 检查 h_idx 和 w_idx 是否在有效范围内
    assert 0 <= h_idx < H, f"h_idx {h_idx} 超出范围 (0, {H-1})"
    assert 0 <= w_idx < W, f"w_idx {w_idx} 超出范围 (0, {W-1})"

    # 提取指定位置的采样点
    sample_points = grid_np[h_idx, w_idx]  # (Kh, Kw, 2)

    # 获取采样点的 x 和 y 坐标
    x_coords = sample_points[:, :, 0]
    y_coords = sample_points[:, :, 1]

    # 创建图形
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # 绘制采样点
    for i in range(Kh):
        for j in range(Kw):
            x = x_coords[i, j]
            y = y_coords[i, j]
            # 绘制采样点，使用不同的标记
            plt.plot(x, y, 'ro')
            # 标注采样点索引
            plt.text(x, y, f'({i},{j})', fontsize=12, ha='right')

    # 绘制采样框的轮廓
    # 获取采样点的顺序，按照卷积核的顺时针顺序连接
    # 这里假设卷积核为 3x3
    if Kh == 3 and Kw == 3:
        indices = [
            (0, 0), (0, 1), (0, 2),
            (1, 2), (2, 2), (2, 1),
            (2, 0), (1, 0), (0, 0)  # 回到起点
        ]
    else:
        # 如果卷积核尺寸不同，需要调整索引的顺序
        indices = []
        for i in range(Kh):
            indices.append((0, i))
        for i in range(1, Kw):
            indices.append((i, Kh - 1))
        for i in range(Kh - 2, -1, -1):
            indices.append((Kw - 1, i))
        for i in range(Kw - 2, 0, -1):
            indices.append((i, 0))
        indices.append((0, 0))  # 回到起点
    # 获取当前像素位置的归一化坐标
    current_x = (w_idx / (W - 1)) * 2 - 1
    current_y = (h_idx / (H - 1)) * 2 - 1

    # 获取卷积核中心位置的索引
    center_i = Kh // 2
    center_j = Kw // 2

    # 获取卷积核中心采样点的坐标
    center_x = x_coords[center_i, center_j]
    center_y = y_coords[center_i, center_j]

    # 打印坐标值
    print(f'当前像素位置 (归一化坐标): x={current_x}, y={current_y}')
    print(f'卷积核中心采样点 (归一化坐标): x={center_x}, y={center_y}')


    
        # 获取当前像素位置的归一化坐标
    current_x = (w_idx / (W - 1)) * 2 - 1
    current_y = (h_idx / (H - 1)) * 2 - 1

    # 绘制当前像素位置
    plt.plot(current_x, current_y, 'gs', markersize=10, label='当前像素位置')
    plt.legend()

    # 翻转 y 轴，使得 y = -1 在顶部，y = 1 在底部
    plt.gca().invert_yaxis()

    # 设置图形标题和坐标轴
    plt.title(f'{title} at (h={h_idx}, w={w_idx})')
    plt.xlabel('Longitude (normalized)')
    plt.ylabel('Latitude (normalized)')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()
"""
# 假设输入图像尺寸为 H x W
H, W = 64, 128  # 您可以根据需要修改

# 创建 SphereConv2d 实例
sphere_conv = SphereConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1)

# 生成采样网格
sampling_grid = sphere_conv.get_sampling_grid(H, W)

# 可视化采样网格
visualize_sampling_grid(sampling_grid)
"""
# 假设输入图像尺寸为 H x W
H, W = 64, 128  # 根据您的实际情况调整


# 创建 SphereConv2d 实例
sphere_conv = SphereConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1)
"""
# 生成采样网格
sampling_grid = sphere_conv.get_sampling_grid(H, W)


Kh, Kw = sphere_conv.kernel_size

# 生成采样网格
sampling_grid = sphere_conv.get_sampling_grid(H, W)

# 可视化中心位置的采样模式
center_h = H // 2
center_w = W // 2
visualize_sampling_grid_at_position(sampling_grid, H, W, Kh, Kw, center_h, center_w, title='中心位置的采样网格')

# 可视化上端位置的采样模式
top_h = 0  # 第一行
visualize_sampling_grid_at_position(sampling_grid, H, W, Kh, Kw, top_h, center_w, title='上端位置的采样网格')

# 可视化下端位置的采样模式
bottom_h = H - 1  # 最后一行
visualize_sampling_grid_at_position(sampling_grid, H, W, Kh, Kw, bottom_h, center_w, title='下端位置的采样网格')

visualize_sampling_grid_mesh(sampling_grid, H, W, Kh, Kw)
"""

Kh, Kw = sphere_conv.kernel_size

# 生成采样网格
sampling_grid = sphere_conv.get_sampling_grid(H, W)

# 定义要可视化的位置列表，包括中间部分的点
positions = [
    (0, W // 2, '上端中心'),
    (1*H // 32, W // 2, '上四分之一处'),
    (1*H // 16, W // 2, '上四分之一处'),
    (H // 8, W // 2, '上四分之一处'),
    (H // 4, W // 2, '上四分之一处'),
    (H // 2, W // 2, '中心'),
    (3 * H // 4, W // 2, '下四分之一处'),
    (7*H // 8, W // 2, '上四分之一处'),
    (15*H // 16, W // 2, '上四分之一处'),
    (31*H // 32, W // 2, '上四分之一处'),
    (H - 1, W // 2, '下端中心'),
    (H // 2, W // 2, '中心'),
]

for h_idx, w_idx, pos_name in positions:
    visualize_sampling_grid_detailed(
        sampling_grid, H, W, Kh, Kw, h_idx, w_idx, title=f'{pos_name}位置的采样网格'
    )