from io import BytesIO

import matplotlib
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter, AbstractMovieWriter, writers
import matplotlib.pyplot as plt
import os

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
def _validate_grabframe_kwargs(savefig_kwargs):
    if matplotlib.rcParams['savefig.bbox'] == 'tight':
        raise ValueError(
            f"{matplotlib.rcParams['savefig.bbox']=} must not be 'tight' as it "
            "may cause frame size to vary, which is inappropriate for animation."
        )
    for k in ('dpi', 'bbox_inches', 'format'):
        if k in savefig_kwargs:
            raise TypeError(
                f"grab_frame got an unexpected keyword argument {k!r}"
            )
@writers.register('pillow')
class PillowWriter(AbstractMovieWriter):
    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._frames = []

    def grab_frame(self, **savefig_kwargs):
        _validate_grabframe_kwargs(savefig_kwargs)
        buf = BytesIO()
        self.fig.savefig(
            buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        im = Image.frombuffer(
            "RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1)
        if im.getextrema()[3][0] < 255:
            # This frame has transparency, so we'll just add it as is.
            self._frames.append(im)
        else:
            # Without transparency, we switch to RGB mode, which converts to P mode a
            # little better if needed (specifically, this helps with GIF output.)
            self._frames.append(im.convert("RGB"))
    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=1)
# GIF 生成函数
def generate_transformation_gif(A, filename='static/feature/transformation.gif',
                                grid_range=(-3, 3), grid_step=0.5,
                                frames=60, interval=50):
    e1 = np.array([1, 0], dtype=float)
    e2 = np.array([0, 1], dtype=float)
    Ae1 = A @ e1
    Ae2 = A @ e2

    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vecs_norm = []
    for i in range(eig_vecs.shape[1]):
        v = eig_vecs[:, i]
        norm = np.linalg.norm(v)
        eig_vecs_norm.append(v / norm if norm != 0 else v)
    eig_vecs_norm = np.array(eig_vecs_norm).T
    eig1 = eig_vecs_norm[:, 0]
    eig2 = eig_vecs_norm[:, 1]
    Aeig1 = A @ eig1
    Aeig2 = A @ eig2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(grid_range)
    ax.set_ylim(grid_range)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    grid_lines = []
    # 垂直网格线
    for x in np.arange(grid_range[0], grid_range[1] + grid_step, grid_step):
        y_vals = np.linspace(grid_range[0], grid_range[1], 200)
        pts = np.column_stack((np.full_like(y_vals, x), y_vals))
        pts_trans = (A @ pts.T).T
        line, = ax.plot(pts[:, 0], pts[:, 1], color='lightgray', lw=0.5)
        grid_lines.append({'orig': pts, 'trans': pts_trans, 'line': line})
    # 水平网格线
    for y in np.arange(grid_range[0], grid_range[1] + grid_step, grid_step):
        x_vals = np.linspace(grid_range[0], grid_range[1], 200)
        pts = np.column_stack((x_vals, np.full_like(x_vals, y)))
        pts_trans = (A @ pts.T).T
        line, = ax.plot(pts[:, 0], pts[:, 1], color='lightgray', lw=0.5)
        grid_lines.append({'orig': pts, 'trans': pts_trans, 'line': line})

    q_Ae1 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                      color='red', label='A·e1')
    q_Ae2 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                      color='orange', label='A·e2')
    q_eig1 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                       color='magenta', width=0.005, label='Eigenvector 1')
    q_eig2 = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1,
                       color='green', width=0.005, label='Eigenvector 2')

    # 加入图例
    ax.legend(loc='upper left', fontsize='small')

    def update(frame):
        t = frame / (frames - 1)
        curr_e1 = (1 - t) * e1 + t * Ae1
        curr_e2 = (1 - t) * e2 + t * Ae2
        q_Ae1.set_UVC(curr_e1[0], curr_e1[1])
        q_Ae2.set_UVC(curr_e2[0], curr_e2[1])
        curr_eig1 = (1 - t) * eig1 + t * Aeig1
        curr_eig2 = (1 - t) * eig2 + t * Aeig2
        q_eig1.set_UVC(curr_eig1[0], curr_eig1[1])
        q_eig2.set_UVC(curr_eig2[0], curr_eig2[1])
        for gl in grid_lines:
            orig = gl['orig']
            trans = gl['trans']
            curr = (1 - t) * orig + t * trans
            gl['line'].set_data(curr[:, 0], curr[:, 1])
        return [q_Ae1, q_Ae2, q_eig1, q_eig2] + [gl['line'] for gl in grid_lines]

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    writer = PillowWriter(fps=1000/interval, metadata={'loop':1})
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    anim.save(filename, writer=writer)
    plt.close(fig)
    return filename
