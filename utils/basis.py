import numpy as np
from matplotlib import pyplot as plt


def basis_plot():
    fig = plt.figure(figsize=(2.5, 2))

    ax = fig.add_subplot(111, projection='3d')

    # 定义向量
    v1 = np.array([2, 1, 0])  # 共线向量之一
    v2 = np.array([4, 2, 0])  # 共线向量之二 (v2 = 2 * v1)
    v3 = np.array([0, 2, 2])  # 共面向量之一
    v4 = np.array([0, 4, 2])  # 共面向量之二
    v5 = np.array([3, 1, 3])  # 共面向量之二

    # 原点
    origin = np.array([0, 0, 0])

    # 绘制向量
    plot_vector(ax, origin, v1, 'r', 'v1')
    plot_vector(ax, origin, v2, 'r', 'v2')
    plot_vector(ax, origin, v3, 'b', 'v3')
    plot_vector(ax, origin, v4, 'b', 'v4')
    plot_vector(ax, origin, v5, 'g', 'v5')

    # 绘制共面区域 (扩展到整个 Y-Z 平面)
    y = np.linspace(0, 5, 10)
    z = np.linspace(0, 5, 10)
    Y, Z = np.meshgrid(y, z)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.2)

    # 坐标轴设置
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Vectors with Collinear and Coplanar Properties")
    plt.savefig("static/basis/plot.png", dpi=120, bbox_inches='tight')
    plt.close()

def plot_vector(ax, origin, vector, color, label):
    ax.quiver(*origin, *vector, color=color, arrow_length_ratio=0.1)
    ax.text(*(np.array(origin) + np.array(vector)), label, color=color, fontsize=10)