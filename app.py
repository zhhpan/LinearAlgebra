import os

import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from flask_bootstrap import Bootstrap

from utils.add import process_vectors, clean_old_files
from utils.basis import basis_plot
from utils.determinant import generate_determinant_gif
from utils.feature import generate_transformation_gif
from utils.solve import cramer_method, gauss_elimination, plot_planes

app = Flask(__name__)
Bootstrap(app)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html" )


@app.route("/add", methods=["GET", "POST"])
def vector_calculator():
    if request.method == "GET":
        # 清理旧文件
        clean_old_files("static/animations")
        clean_old_files("temp_frames")

        return render_template("add.html",
                               vector_size=2,
                               dimension='2d',
                               operation='add',
                               v1=[0,0],
                               v2=[0,0],
                               result=[0,0])

    elif request.method == "POST":
        try:
            dimension = request.form.get("dimension", "2d")
            operation = request.form.get("operation", "add")
            vector_size = 3 if dimension == '3d' else 2

            # 获取向量数据
            v1 = [float(request.form.get(f"v1_{i}", 0)) for i in range(vector_size)]
            v2 = [float(request.form.get(f"v2_{i}", 0)) for i in range(vector_size)]

            # 处理可视化
            plot_html, gif_url ,result= process_vectors(v1, v2, operation, dimension)

            # 准备回填数据
            form_data = {
                'vector_size': vector_size,
                'dimension': dimension,
                'operation': operation,
                'plot_html': plot_html,
                'gif_url': gif_url,
                'result': result,
                'v1': v1,
                'v2': v2,
            }
            form_data.update({f'v1_{i}': v1[i] for i in range(vector_size)})
            form_data.update({f'v2_{i}': v2[i] for i in range(vector_size)})

            return render_template("add.html", **form_data)

        except ValueError as e:
            flash(f"输入错误: {str(e)}", "error")
        except Exception as e:
            flash(f"系统错误: {str(e)}", "error")

        # 出错时重定向回首页
        return redirect(url_for('/'))


@app.route('/solve', methods=["GET", "POST"])
def solve():
    # 初始化变量
    result = None
    message = ""
    show_steps = False
    method = "gauss"  # 默认方法
    matrix_data = {
        'a1': '1', 'b1': '3', 'c1': '2', 'd1': '4',
        'a2': '2', 'b2': '1', 'c2': '4', 'd2': '2',
        'a3': '3', 'b3': '2', 'c3': '5', 'd3': '1'
    } # 存储表单输入值
    det_data = {}  # 克莱姆法则数据（字典）
    gauss_data = []  # 高斯消元数据（列表）

    if request.method == "POST":
        try:
            # 从表单获取系数矩阵和常数项
            A = np.array([
                [float(request.form["a1"]), float(request.form["b1"]), float(request.form["c1"])],
                [float(request.form["a2"]), float(request.form["b2"]), float(request.form["c2"])],
                [float(request.form["a3"]), float(request.form["b3"]), float(request.form["c3"])]
            ], dtype=np.float64)

            b = np.array([
                float(request.form["d1"]),
                float(request.form["d2"]),
                float(request.form["d3"])
            ], dtype=np.float64)

            # 获取用户选择的求解方法和显示选项
            method = request.form.get("method", "gauss")
            show_steps = request.form.get("showSteps") == "on"

            # 根据方法调用对应的求解函数
            if method == "cramer":
                # 接收三元组：结果、错误消息、步骤数据
                result, msg, det_data = cramer_method(A, b)
                message = msg  # 将错误消息赋给message变量
            else:
                result, msg, gauss_data = gauss_elimination(A, b)
                message = msg
            # 保存用户输入用于回显
            for name in ["a1", "b1", "c1", "d1",
                         "a2", "b2", "c2", "d2",
                         "a3", "b3", "c3", "d3"]:
                matrix_data[name] = request.form.get(name, "")

            # 生成三维可视化
            if result and len(result) == 3:
                plot_planes(A, b)  # 假设已实现绘图函数


        except ValueError as e:
            message = f"输入错误：{str(e)}"
        except np.linalg.LinAlgError as e:
            message = f"矩阵计算错误：{str(e)}"
        except Exception as e:
            message = f"系统错误：{str(e)}"

            # 开发环境记录详细错误
            if os.environ.get('FLASK_ENV') == 'development':
                import traceback
                message += f"<pre>{traceback.format_exc()}</pre>"

    # 渲染模板时传递所有必要参数
    return render_template(
        "solve.html",
        result=result,
        message=message,
        show_steps=show_steps,  # 是否显示详细步骤
        method=method,  # 当前使用的方法
        matrix_data=matrix_data,  # 表单回显数据
        det_data=det_data,  # 克莱姆法则步骤数据
        gauss_data=gauss_data  # 高斯消元步骤数据
    )


@app.route("/plot1")
def plot_solve():
    return send_file("static/solve/plot.png", mimetype="image/png")
@app.route("/plot2")
def plot_basis():
    return send_file("static/basis/plot.gif", mimetype="text/png")

@app.route("/feature", methods=["GET", "POST"])
def feature():
    gif_url = None
    if request.method == 'POST':
        # 从表单获取矩阵元素
        a = float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        A = np.array([[a, b], [c, d]])
        # 生成 GIF
        filename = generate_transformation_gif(A)
        gif_url = f"static/feature/transformation.gif"
    return render_template('feature.html', gif_url=gif_url)

@app.route('/quadric', methods=["GET", "POST"])
def quadratic():
    return render_template('quadric.html')

@app.route('/determinant', methods=["GET", "POST"])
def determinate():
    gif_url = None
    if request.method == 'POST':
        # 从表单获取矩阵元素
        a = float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        A = np.array([[a, b], [c, d]])
        # 生成 GIF
        filename = generate_determinant_gif(A)
        gif_url = f"static/determinant/determinant_area.gif"
    return render_template('determinant.html', gif_url=gif_url)


@app.route('/basis', methods=["GET", "POST"])
def basis():
    basis_plot()
    return render_template('basis.html')

if __name__ == '__main__':
    app.run()
