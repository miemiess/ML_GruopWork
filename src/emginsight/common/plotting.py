"""实验报告图共用的 Matplotlib 设置。"""

import matplotlib


## @brief 让图中的中文正常显示，并使用普通减号。在创建图之前调用一次。
def use_chinese_font():
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
