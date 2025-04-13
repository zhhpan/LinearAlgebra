import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def plot_planes(A, b):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")

    # 网格分辨率更高，范围更大，避免曲面过度畸形
    x = y = np.linspace(-30, 30, 20)
    X, Y = np.meshgrid(x, y)

    try:
        solution = np.linalg.solve(A, b)
        has_solution = True
    except np.linalg.LinAlgError:
        has_solution = False

    plane_params = {
        'alpha': 0.3,
        'rstride': 2,
        'cstride': 2,
        'edgecolor': 'k',
        'linewidth': 0.3,
        'antialiased': False
    }

    colors = ['#FF4755', '#2ED573', '#5352ED']
    labels = ['Plant 1', 'Plant 2', 'Plant 3']

    # 绘制平面
    for i in range(3):
        a, b_coeff, c = A[i]

        # 避免 c=0 除零错误
        d = c if abs(c) > 1e-6 else 1e-6

        Z = (b[i] - a * X - b_coeff * Y) / d
        surf = ax.plot_surface(X, Y, Z, color=colors[i], **plane_params)

        # 强制让图例不会因为plot_surface无法识别label而丢失
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

    # 绘制解的点
    if has_solution:
        x_sol, y_sol, z_sol = solution
        ax.scatter(
            x_sol, y_sol, z_sol,
            s=150,
            color='#FFD700',
            edgecolors='black',
            linewidth=1.2,
            label='解的坐标',
            zorder=100
        )

        # 辅助线：x/y/z轴方向参考
        ax.plot([x_sol, x_sol], [y_sol, y_sol], [z_sol - 5, z_sol + 5], color='#FF0000', linestyle='-', linewidth=1.2)
        ax.plot([x_sol, x_sol], [y_sol - 5, y_sol + 5], [z_sol, z_sol], color='#00FF00', linestyle='-', linewidth=1.2)
        ax.plot([x_sol - 5, x_sol + 5], [y_sol, y_sol], [z_sol, z_sol], color='#0000FF', linestyle='-', linewidth=1.2)

    # 坐标轴设置
    ax.set_xlabel('X Axis', fontsize=9, labelpad=8)
    ax.set_ylabel('Y Axis', fontsize=9, labelpad=8)
    ax.set_zlabel('Z Axis', fontsize=9, labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # 用 proxy artist 补全图例
    proxies = [Patch(facecolor=colors[i], label=labels[i], alpha=0.3) for i in range(3)]
    if has_solution:
        proxies.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Solving Point',
                       markerfacecolor='#FFD700', markersize=8, markeredgecolor='black')
        )

    ax.legend(
        handles=proxies,
        loc='upper right',
        fontsize=7,
        frameon=True,
        shadow=True,
        facecolor='white',
        edgecolor='gray'
    )

    # 调整视角
    ax.view_init(elev=20, azim=120)

    # 保存图片
    plt.savefig("static/solve/plot.png", dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------
# 通用工具函数
# --------------------------
def render_matrix(matrix):
    """将矩阵渲染为LaTeX格式（用于克莱姆法则）"""
    rows = [" & ".join([f"{x:.3f}" for x in row]) for row in matrix]
    return r" \\ ".join(rows)


def render_augmented(matrix):
    """
    生成增广矩阵的 LaTeX 表达式
    :param matrix: numpy 增广矩阵 (n x 4)
    :return: 完整的 LaTeX array 表达式
    """
    # 定义列格式：前三列右对齐，最后常数列用竖线分隔
    col_format = r"{rrr|r}"

    # 生成每一行的内容
    rows = []
    for row in matrix:
        # 将数值格式化为字符串，并用 & 连接
        str_row = [f"{x:.3f}" for x in row]
        rows.append(" & ".join(str_row))

    # 组合成完整的 array 环境
    latex_code = (
            r"\begin{array}" + col_format + "\n"
                                            r"  " + " \\\\\n  ".join(rows) + "\n"
                                                                             r"\end{array}"
    )
    return latex_code


# --------------------------
# 克莱姆法则实现
# --------------------------
def cramer_method(A, b, epsilon=1e-10):
    """返回格式：(解向量 或 None, 状态消息, 步骤数据)"""
    det_data = {}
    try:
        # 计算行列式D
        D = np.linalg.det(A)
        det_data["D"] = {
            "value": round(D, 3),
            "matrix": A.round(3).tolist(),
            "latex": render_matrix(A)
        }

        # 判断行列式是否为零
        if abs(D) < epsilon:
            # 分析方程组解的情况
            rank_A = np.linalg.matrix_rank(A)
            augmented = np.column_stack((A, b))
            rank_augmented = np.linalg.matrix_rank(augmented)

            if rank_A == rank_augmented:
                msg = "行列式D=0，方程组可能有无穷多解"
            else:
                msg = "行列式D=0，方程组无解"

            det_data["error"] = msg
            return None, msg, det_data  # 返回三元组

        # 正常计算Dx/Dy/Dz
        matrices = {}
        for i, name in enumerate(["Dx", "Dy", "Dz"]):
            modified_A = A.copy()
            modified_A[:, i] = b  # 替换第i列为常数项
            D_value = np.linalg.det(modified_A)

            matrices[name] = {
                "value": round(D_value, 3),
                "matrix": modified_A.round(3).tolist(),
                "latex": render_matrix(modified_A)
            }

        det_data.update(matrices)  # 合并到步骤数据

        # 计算结果
        x = round(matrices["Dx"]["value"] / D, 3)
        y = round(matrices["Dy"]["value"] / D, 3)
        z = round(matrices["Dz"]["value"] / D, 3)

        return (x, y, z), "克莱姆法则求解成功", det_data

    except Exception as e:
        error_msg = f"克莱姆法则计算错误: {str(e)}"
        det_data["error"] = error_msg
        return None, error_msg, det_data


# --------------------------
# 高斯消元法实现
# --------------------------
def gauss_elimination(A, b, epsilon=1e-10):
    """
    参数:
        A (np.ndarray): 3x3系数矩阵
        b (np.ndarray): 3x1常数项向量

    返回:
        tuple: (解向量, 状态消息, 步骤列表)
    """
    steps = []
    n = len(b)
    augmented = np.hstack((A, b.reshape(-1, 1))).astype(float)

    try:
        # 初始矩阵记录
        steps.append({
            'description': '初始增广矩阵',
            'matrix': augmented.round(3).copy(),
            'latex': render_augmented(augmented)
        })

        # 前向消元
        for i in range(n):
            # 部分主元选择
            max_row = np.argmax(np.abs(augmented[i:, i])) + i
            if max_row != i:
                augmented[[i, max_row]] = augmented[[max_row, i]]
                steps.append({
                    'description': f'行交换：行{i + 1} ↔ 行{max_row + 1}',
                    'matrix': augmented.round(3).copy(),
                    'latex': render_augmented(augmented)
                })

            pivot = augmented[i, i]
            if abs(pivot) < epsilon:
                raise np.linalg.LinAlgError("矩阵奇异，无法继续消元")

            # 归一化当前行（可选）
            # augmented[i] = augmented[i] / pivot

            # 消去下方元素
            for j in range(i + 1, n):
                factor = augmented[j, i] / pivot
                augmented[j, i:] -= factor * augmented[i, i:]

                steps.append({
                    'description': f'行{j + 1} ← 行{j + 1} - {factor:.3f}×行{i + 1}',
                    'matrix': augmented.round(3).copy(),
                    'latex': render_augmented(augmented)
                })

        # 回代求解
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (augmented[i, -1] - np.dot(augmented[i, i + 1:n], x[i + 1:])) / augmented[i, i]
            x[i] = round(x[i], 3)

        # 记录最终矩阵
        steps.append({
            'description': '行阶梯形矩阵',
            'matrix': augmented.round(3).copy(),
            'latex': render_augmented(augmented)
        })

        return x.tolist(), "求解成功", steps

    except np.linalg.LinAlgError as e:
        return None, f"消元错误: {str(e)}", steps
    except Exception as e:
        return None, f"计算错误: {str(e)}", steps



