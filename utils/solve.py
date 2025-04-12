import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_planes(A, b):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")

    x = y = np.linspace(-15, 15, 15)
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
    labels = ['平面1', '平面2', '平面3']

    for i in range(3):
        a, b_coeff, c = A[i]
        d = c if c != 0 else 1e-6
        Z = (b[i] - a * X - b_coeff * Y) / d
        surf = ax.plot_surface(X, Y, Z, color=colors[i], label=labels[i], **plane_params)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

    if has_solution:
        x_sol, y_sol, z_sol = solution
        ax.scatter(
            x_sol, y_sol, z_sol,
            s=150,
            color='#FFD700',
            linewidth=2.5,
            label='Solving Point',
            zorder=100
        )
        # XYZ三轴方向的参考线
        ax.plot([x_sol, x_sol], [y_sol, y_sol], [z_sol - 10, z_sol + 10], color='black', linestyle='--', linewidth=1)
        ax.plot([x_sol, x_sol], [y_sol - 10, y_sol + 10], [z_sol, z_sol], color='black', linestyle='--', linewidth=1)
        ax.plot([x_sol - 10, x_sol + 10], [y_sol, y_sol], [z_sol, z_sol], color='black', linestyle='--', linewidth=1)

        for coord, color in zip([x_sol, y_sol, z_sol], ['#FF0000', '#00FF00', '#0000FF']):
            ax.plot(
                [coord, coord], [y_sol, y_sol], [z_sol-3, z_sol+3],
                color=color,
                linestyle='-',
                alpha=1,
                linewidth=1.2
            )

    # 坐标轴设置
    ax.set_xlabel('X Axis', fontsize=8, labelpad=8)
    ax.set_ylabel('Y Axis', fontsize=8, labelpad=8)
    ax.set_zlabel('Z Axis', fontsize=8, labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # 图例优化
    ax.legend(
        loc='upper right',
        fontsize=6,
        markerscale=0.4,
        frameon=True,
        shadow=True,
        facecolor='white',
        edgecolor='gray'
    )

    ax.view_init(elev=30, azim=45)
    plt.savefig("static/solve/plot.png", dpi=200, bbox_inches='tight')  # 降低dpi至200
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



