# datasets 模块

对应需求：R03-R05。职责：分别解析三个数据集，输出 Recording；记录单位、标签、时间对齐与来源。

规划文件：uci_gestures.py、putemg_force.py、zenodo_fatigue.py。当前仅有目录和包占位，具体实现由需求任务添加。

接口以[架构约定](../../../docs/architecture.md)为准，验收以[需求清单](../../../docs/requirements.md)为准。单元测试放在 tests/unit/datasets/；编排流程的跨模块测试统一放在 tests/integration/。
