import os
import uuid
import time
import numpy as np
import imageio
import plotly.graph_objs as go
from flask import Flask
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from functools import lru_cache



# 配置中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False



# 确保必要的目录存在
os.makedirs("static/animations", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)
os.makedirs("temp_frames", exist_ok=True)


def ease_in_out(t):
    """缓动函数，使动画更平滑"""
    return 0.5 * (1 + np.sin(np.pi * (t - 0.5)))


def clean_old_files(directory, max_age_hours=1):
    """清理超过指定时间的旧文件"""
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)
            if file_age > max_age_hours * 3600:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")


def rotation_matrix(angle, axis1, axis2):
    """生成旋转矩阵"""
    norm = np.linalg.norm
    axis = np.cross(axis1, axis2)
    axis = axis / norm(axis)

    cos = np.cos(angle)
    sin = np.sin(angle)

    return np.array([
        [cos + axis[0] ** 2 * (1 - cos),
         axis[0] * axis[1] * (1 - cos) - axis[2] * sin,
         axis[0] * axis[2] * (1 - cos) + axis[1] * sin],
        [axis[1] * axis[0] * (1 - cos) + axis[2] * sin,
         cos + axis[1] ** 2 * (1 - cos),
         axis[1] * axis[2] * (1 - cos) - axis[0] * sin],
        [axis[2] * axis[0] * (1 - cos) - axis[1] * sin,
         axis[2] * axis[1] * (1 - cos) + axis[0] * sin,
         cos + axis[2] ** 2 * (1 - cos)]
    ])

def generate_cross_product_animation(v1, v2, result, resolution):
    """专用叉乘动画生成"""
    num_frames = 60
    frames = []
    filename = f"static/animations/cross_{uuid.uuid4().hex}.gif"

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置固定坐标系
    max_val = max(np.abs(np.concatenate([v1, v2, result]))) * 1.5
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # 初始箭头
    arrow_v1 = ax.quiver(0, 0, 0, *v1, color='red', label='v₁')
    arrow_v2 = ax.quiver(0, 0, 0, *v2, color='blue', label='v₂')
    arrow_result = ax.quiver(0, 0, 0, *result, color='green', label='v₁×v₂')

    # 添加右手法则示意图
    right_hand = ax.plot([], [], [], 'k-', lw=2)[0]
    angles = np.linspace(0, 2 * np.pi, num_frames)

    def update(frame):
        # 旋转向量展示方向关系
        theta = angles[frame]
        rot_matrix = rotation_matrix(theta, v1, v2)

        # 更新箭头
        new_v2 = rot_matrix @ v2
        arrow_v2.remove()
        new_arrow = ax.quiver(0, 0, 0, *new_v2, color='blue')

        # 更新右手法则曲线
        t = np.linspace(0, 1, 50)
        curve = 0.5 * (v1 + new_v2) + 0.3 * result * np.sin(2 * np.pi * t)
        right_hand.set_data(curve[:2])
        right_hand.set_3d_properties(curve[2])

        return new_arrow, right_hand

    anim = FuncAnimation(fig, update, frames=num_frames, blit=True)
    anim.save(filename, writer='pillow', fps=20)
    plt.close()
    return filename


def generate_gif(v1, v2, result, operation, dimension='2d', resolution=400, quality=8):
    """
    生成向量运算的GIF动画
    :param v1: 向量1
    :param v2: 向量2
    :param result: 运算结果
    :param operation: 运算类型 ('add', 'subtract', 'cross')
    :param dimension: 维度 ('2d' 或 '3d')
    :param resolution: 输出GIF的分辨率
    :param quality: GIF质量 (1-10)
    :return: GIF文件路径
    """
    try:
        # 根据维度处理向量
        v1 = np.array(v1[:3] if dimension == '3d' else v1[:2])
        v2 = np.array(v2[:3] if dimension == '3d' else v2[:2])
        result = np.array(result[:3] if dimension == '3d' else result[:2])

        num_frames = 30
        frames = []
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        # 生成每一帧
        for i in range(num_frames):
            alpha = ease_in_out(i / (num_frames - 1))

            fig = plt.figure(figsize=(6, 6))
            if dimension == '2d':
                ax = fig.add_subplot(111)
                axis_limit = np.max(np.abs([v1, v2, result])) + 2
                ax.set_xlim([-axis_limit, axis_limit])
                ax.set_ylim([-axis_limit, axis_limit])
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.axvline(0, color='gray', linewidth=0.5)

                # 绘制向量
                ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
                          color='red', width=0.015, label='v1')

                move_vec = v2 if operation == 'add' else -v2
                tail = v1 * alpha
                ax.quiver(tail[0], tail[1], move_vec[0], move_vec[1], angles='xy',
                          scale_units='xy', scale=1, color='green', width=0.015,
                          label='v2' if operation == 'add' else '-v2')

                ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy',
                          scale=1, color='blue', width=0.015, label='result')

                ax.legend(loc='upper right')
                ax.set_aspect('equal')
            elif operation == 'cross' and dimension == '3d':
                return generate_cross_product_animation(v1, v2, result, resolution)
            else:
                ax = fig.add_subplot(111, projection='3d')
                axis_limit = np.max(np.abs([v1, v2, result])) + 2
                ax.set_xlim([-axis_limit, axis_limit])
                ax.set_ylim([-axis_limit, axis_limit])
                ax.set_zlim([-axis_limit, axis_limit])

                # 绘制3D向量
                ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='red', label='v1')

                move_vec = v2 if operation == 'add' else -v2
                tail = v1 * alpha
                ax.quiver(tail[0], tail[1], tail[2], move_vec[0], move_vec[1], move_vec[2],
                          color='green', label='v2' if operation == 'add' else '-v2')

                ax.quiver(0, 0, 0, result[0], result[1], result[2], color='blue', label='result')

                ax.legend()
                ax.set_box_aspect([1, 1, 1])

            # 保存帧
            frame_path = os.path.join(temp_dir, f"frame_{i}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            frames.append(frame_path)

        # 生成GIF
        gif_filename = f"static/animations/vector_{operation}_{uuid.uuid4().hex}.gif"
        with imageio.get_writer(gif_filename, mode='I', duration=0.1, quality=quality) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                image = Image.fromarray(image).resize((resolution, resolution))
                writer.append_data(image)

        # 清理临时文件
        for frame in frames:
            try:
                os.remove(frame)
            except:
                pass

        return gif_filename

    except Exception as e:
        print(f"Error generating GIF: {str(e)}")
        raise


@lru_cache(maxsize=32)
def plot_vectors(v1, v2, operation, dimension):
    """
    缓存静态图生成结果
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    if operation == 'add':
        result = v1 + v2
    elif operation == 'subtract':
        result = v1 - v2
    elif operation == 'cross' and dimension == '3d':
        result = np.cross(v1, v2)
    else:
        raise ValueError("不支持的运算类型")

    if dimension == '2d':
        return plot_2d(v1, v2, result, operation)
    else:
        return plot_3d(v1, v2, result, operation)


def plot_2d(v1, v2, result, operation):
    """生成2D向量图"""
    fig = go.Figure()

    # 添加向量箭头
    for vec, color, name in [
        (v1, 'red', 'v₁'),
        (v2, 'green', 'v₂' if operation == 'add' else '-v₂'),
        (result, 'blue', 'result'),
    ]:
        fig.add_trace(go.Scatter(
            x=[0, vec[0]], y=[0, vec[1]],
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=10),
            name=name
        ))

    # 设置布局
    axis_limit = np.max(np.abs(np.concatenate([v1, v2, result]))) * 1.2
    fig.update_layout(
        title=f"2D Vector{'Addition' if operation == 'add' else 'Subtraction'}",
        xaxis=dict(range=[-axis_limit, axis_limit], zeroline=True),
        yaxis=dict(range=[-axis_limit, axis_limit], zeroline=True),
        showlegend=True,
        width=500,
        height=500
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def plot_3d(v1, v2, result, operation):
    """生成3D向量图"""
    fig = go.Figure()

    # 添加3D向量
    for vec, color, name in [
        (v1, 'red', 'v₁'),
        (v2, 'green', 'v₂' if operation == 'add' else '-v₂'),
        (result, 'blue', 'result')
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines',
            line=dict(color=color, width=6),
            name=name
        ))

    # 设置3D场景
    axis_limit = np.max(np.abs(np.concatenate([v1, v2, result]))) * 1.2
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-axis_limit, axis_limit], title='X'),
            yaxis=dict(range=[-axis_limit, axis_limit], title='Y'),
            zaxis=dict(range=[-axis_limit, axis_limit], title='Z'),
            aspectmode='cube'
        ),
        title=f"3D Vector{'Addition' if operation == 'add' else 'Subtraction'}",
        width=600,
        height=600
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def process_vectors(v1, v2, operation, dimension):
    """
    处理向量运算和可视化
    :return: (plot_html, gif_url, result)
    """
    try:
        # 验证输入
        if len(v1) != len(v2):
            raise ValueError("向量维度不一致")

        if dimension == '2d' and len(v1) < 2:
            raise ValueError("2D运算需要至少2维向量")

        if dimension == '3d' and len(v1) < 3:
            raise ValueError("3D运算需要至少3维向量")

        if operation == 'cross' and dimension != '3d':
            raise ValueError("叉积仅支持3D向量")

        # 计算结果
        if operation == 'add':
            result = np.add(v1, v2)
        elif operation == 'subtract':
            result = np.subtract(v1, v2)
        elif operation == 'cross':
            result = np.cross(v1, v2)
        else:
            raise ValueError("不支持的运算类型")

        # 生成可视化
        plot_html = plot_vectors(tuple(v1), tuple(v2), operation, dimension)
        gif_url = generate_gif(v1, v2, result, operation, dimension)

        return plot_html, gif_url, result

    except Exception as e:
        print(f"Error processing vectors: {str(e)}")
        raise