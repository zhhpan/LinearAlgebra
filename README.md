
# 线性代数可视化工具

一个基于Flask的线性代数计算与可视化Web应用，提供向量运算、矩阵变换、方程组求解等功能的交互式可视化。

## ✨ 功能特性

- **向量运算**：支持2D/3D向量的加减法、叉积运算，并生成动画演示
- **矩阵变换**：可视化展示矩阵对标准基和特征向量的变换过程
- **行列式**：动态演示行列式的几何意义（面积/体积变化）
- **线性方程组求解**：
  - 高斯消元法（带步骤展示）
  - 克莱姆法则（带行列式计算过程）
- **基变换**：展示向量在不同基下的表示

## 🛠️ 技术栈

| 技术        | 用途                   |
|-------------|------------------------|
| Python/Flask | 后端框架               |
| NumPy/SymPy | 数学计算               |
| Matplotlib  | 2D/3D可视化            |
| Plotly      | 交互式图表             |
| Bootstrap   | 前端UI框架             |
| KaTeX       | 数学公式渲染           |

## 🚀 快速开始
访问http://114.215.206.6:5000/



## 🚀 本地部署
1. 克隆仓库
```bash
git clone https://github.com/zhhpan/LinearAlgebra.git
cd LinearAlgebra
 ```


2. 安装依赖
```bash
pip install -r requirements.txt
 ```

3. 运行应用
```bash
python app.py
 ```

4. 访问应用
```plaintext
http://localhost:5000
 ```

## 📂 项目结构
```plaintext
LinearAlgebra/
├── app.py                # Flask主应用
├── requirements.txt      # Python依赖
├── static/               # 静态资源
│   ├── animations/       # 生成的动画
│   ├── plots/           # 静态图表
│   └── katex/           # 数学公式渲染
├── templates/           # HTML模板
│   ├── add.html         # 向量运算页面
│   ├── solve.html       # 方程组求解
│   └── ...              
└── utils/               # 核心功能模块
    ├── add.py           # 向量运算逻辑
    ├── solve.py         # 方程组求解
    └── ...
 ```

## 📝 使用说明
1. 向量运算 ：
   
   - 选择2D或3D维度
   - 输入向量坐标
   - 选择运算类型（加法/减法/叉积）
   - 查看动画演示和计算结果
2. 方程组求解 ：
   
   - 输入3×3系数矩阵
   - 选择求解方法（高斯消元/克莱姆法则）
   - 查看详细计算步骤
3. 矩阵变换 ：
   
   - 输入2×2矩阵元素
   - 观察标准基和特征向量的变换过程
## 🤝 贡献指南
1. Fork本项目
2. 创建你的分支 ( git checkout -b feature/your-feature )
3. 提交修改 ( git commit -m 'Add some feature' )
4. 推送到分支 ( git push origin feature/your-feature )
5. 发起Pull Request
## 📜 许可证
MIT License © 2023