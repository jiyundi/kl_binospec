# safe_plot.py

import matplotlib

def setup():
    """
    根据运行环境自动设置 matplotlib 的 backend：
    - 在终端环境中使用非交互式 'Agg'，防止弹出图形窗口；
    - 在 IPython 或 Spyder 中使用默认交互式 backend。
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            # 普通终端，使用非交互后端以避免报错和弹窗
            matplotlib.use('Agg')
        # IPython/Jupyter/Spyder 中保持默认交互式 backend
    except Exception:
        matplotlib.use('Agg')  # 出现异常时兜底为非交互后端
