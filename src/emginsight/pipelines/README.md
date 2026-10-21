# pipelines 模块

对应需求：R15。职责：按配置调用公共模块完成训练、评估和推理，不复制算法。

规划文件：train.py、infer.py、__main__.py。当前仅有目录和包占位，具体实现由需求任务添加。

接口以[架构约定](../../../docs/architecture.md)为准，验收以[需求清单](../../../docs/requirements.md)为准。编排流程的跨模块测试统一放在 tests/integration/。
