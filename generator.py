import nbformat as nbf
import os

def create_jupyter_notebook(filename='template.ipynb'):
    nb = nbf.v4.new_notebook()
    
    import_title_cell = nbf.v4.new_markdown_cell("""# 引入""")
    
    import_cell = nbf.v4.new_code_cell("""import torch
import torchvision
from torch import nn
from torch.utils import data
from d2l import torch as d2l
from torchvision import transforms
from IPython import display""")
    
    animator_title_cell = nbf.v4.new_markdown_cell("""# 动画""")
    
    animator_cell = nbf.v4.new_code_cell(
"""class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)""")
    
    data_title_cell = nbf.v4.new_markdown_cell("""# 数据""")
    
    data_cell = nbf.v4.new_code_cell("""
""")
    
    model_title_cell = nbf.v4.new_markdown_cell("""# 模型""")
    
    model_cell = nbf.v4.new_code_cell("""
""")
    
    training_title_cell = nbf.v4.new_markdown_cell("""# 训练""")
    
    training_cell = nbf.v4.new_code_cell("""
""")

    # Add cells to the notebook
    nb['cells'] = [
        import_title_cell,
        import_cell,
        animator_title_cell,
        animator_cell,
        data_title_cell,
        data_cell,
        model_title_cell,
        model_cell,
        training_title_cell,
        training_cell
    ]
    
    # Write notebook to file
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Jupyter notebook template created: {filename}")

# Generate the notebook
create_jupyter_notebook()