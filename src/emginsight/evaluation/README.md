# evaluation 模块

对应需求：R13、R19。职责：统一计算分类/回归指标、逐用户结果和公平对比图表。

规划文件：metrics.py、plots.py、comparisons.py。当前仅有目录和包占位，具体实现由需求任务添加。

接口以[架构约定](../../../docs/architecture.md)为准，验收以[需求清单](../../../docs/requirements.md)为准。单元测试放在 tests/unit/evaluation/；编排流程的跨模块测试统一放在 tests/integration/。
