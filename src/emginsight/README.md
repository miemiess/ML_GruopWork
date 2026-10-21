# 公共 Python 包

模块实现与边界以[架构约定](../../docs/architecture.md)为准，任务以[需求清单](../../docs/requirements.md)为准。

| 模块 | 需求 | 职责 |
| --- | --- | --- |
| core | R01、R02 | 公共类型、配置与异常 |
| datasets | R03-R05 | 三个独立数据适配器 |
| splitting | R06 | 分窗前按用户/会话划分 |
| preprocessing | R07、R09 | 记录处理、分窗、训练集标准化 |
| features | R08 | 确定性时域特征 |
| models | R10-R12 | 自定义基线、kNN、逻辑回归、岭回归 |
| evaluation | R13、R19 | 指标、图表与公平对比 |
| artifacts | R14 | 运行记录、状态保存恢复 |
| pipelines | R15 | 调用模块完成训练、评估与推理 |

当前只有包占位与文档，不能运行训练。Notebook 和 demo 后续通过已安装的包调用函数，不复制实现。公共模块不依赖 pipelines、demo 或 Notebook。
