# preprocessing 模块

对应需求：R07、R09。职责：记录内滤波、分窗、目标聚合及仅训练集拟合的变换。

规划文件：windowing.py、transforms.py。当前仅有目录和包占位，具体实现由需求任务添加。

接口以[架构约定](../../../docs/architecture.md)为准，验收以[需求清单](../../../docs/requirements.md)为准。单元测试放在 tests/unit/preprocessing/；编排流程的跨模块测试统一放在 tests/integration/。
