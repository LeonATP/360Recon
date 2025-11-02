import numpy as np
import struct
import torch
import os
import matplotlib.pyplot as plt
INT_BYTES = 4
USHORT_BYTES = 2

# 设置文件路径
file_path = '/home/yzm/dataset/Matterport3D/1LXtFkjw3qL/frame_000.dpt'

'''
def load_hinter_depth(path):
    """
    Loads depth image from the '.dpt' format used by Hinterstoisser.
    """
    f = open(path, 'rb')
    _= struct.unpack('i', f.read(INT_BYTES))[0]
    w = struct.unpack('i', f.read(INT_BYTES))[0]
    h = struct.unpack('i', f.read(INT_BYTES))[0]
    depth_map = []
    # 计算文件中应包含的深度数据的字节数
    #expected_data_size = h * w * INT_BYTES

    # 获取文件的实际大小
    #actual_file_size = os.path.getsize(path)
    
    for i in range(1024):
        depth_map.append(struct.unpack(w*'i', f.read(w * INT_BYTES)))
        print(i)
    # return np.array(depth_map, np.uint16)
    """
    if(i==1023):
       n=struct.unpack('i', f.read(INT_BYTES))[0]
    """
    m = np.array(depth_map, np.float32) / 4000
    #visualize_depth_map(m)
    return m
'''

def load_hinter_depth(path):
    '''
    Loads depth image from the '.dpt' format used by Hinterstoisser.
    '''
    INT_BYTES = 4

    with open(path, 'rb') as f:
        _ = struct.unpack('i', f.read(INT_BYTES))[0]  # Skipping the first integer
        w = struct.unpack('i', f.read(INT_BYTES))[0]  # Width of the image
        h = struct.unpack('i', f.read(INT_BYTES))[0]  # Height of the image

        # Read the entire depth map in one go
        depth_map = np.fromfile(f, dtype=np.float32, count=w * h).reshape((h, w))
        depth_map_tensor = torch.from_numpy(depth_map)
    m = depth_map_tensor
    print(m.max())
    print(m.min())
    print(m.mean())
    visualize_depth_map(m)  # Uncomment this if you need to visualize the depth map
    return m


def visualize_depth_map(depth_map):
    plt.imshow(depth_map, cmap='viridis')  # 使用 'viridis' 颜色映射，可以换成其他颜色映射
    plt.colorbar(label='Depth')  # 添加颜色条表示深度值
    plt.title('Depth Map Visualization')
    plt.show()

load_hinter_depth(file_path)