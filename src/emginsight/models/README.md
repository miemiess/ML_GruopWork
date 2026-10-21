# models 模块

对应需求：R10-R12。职责：实现基线、kNN、逻辑回归与岭回归及状态恢复，不承担文件读写。

规划文件：baselines.py、knn.py、logistic.py、ridge.py。当前仅有目录和包占位，具体实现由需求任务添加。

接口以[架构约定](../../../docs/architecture.md)为准，验收以[需求清单](../../../docs/requirements.md)为准。单元测试放在 tests/unit/models/；编排流程的跨模块测试统一放在 tests/integration/。
