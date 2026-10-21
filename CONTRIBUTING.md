# EMGInsight 贡献指南

六位成员共同维护项目，通过 Issues 选择工作，不固定分配个人角色。

## 开始贡献

创建 Issue，说明问题、预期产出、受影响模块、数据假设与验收条件。并行开发前先约定接口变更。

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/short-description
```

修复问题或修改文档时可使用 `fix/` 或 `docs/` 分支。每次提交聚焦一个明确的修改，再推送分支：

```bash
git add path/to/changed-file
git commit -m "说明本次具体修改"
git push -u origin feature/short-description
```

创建面向 `main` 的 PR，关联 Issue，说明目的、检查方式、结果与限制。通常由另一位成员审阅后合并；在功能分支上解决冲突并重新运行相关检查。经明确约定的小范围文档修改可以直接提交到 `main`，本次中文化按此方式提交。不要强制推送共同维护的 `main`。

## 实现约定

- 可复用逻辑放在 `src/emginsight/`，Notebook 调用公共包中的函数。
- 手势、力量和疲劳数据分别使用独立适配器。
- 保留用户、采集会话、原始记录、通道、时间戳与标签来源。
- 特征使用 `X: (n_windows, n_features)`，标签使用 `y: (n_windows,)` 或 `(n_windows, n_outputs)`，同时保留分组标识；在文档字符串中说明形状和单位。
- 模型提供 `fit(X, y)` 与 `predict(X)`，注明分类标签及回归输出形状。
- 使用 Python/NumPy 实现核心算法，明确标注用于参考的库模型，不使用深度学习。
- 按照[研究与评估规范](docs/research_protocol.md)防止数据泄漏并开展评估。

## 提交前检查

- 测试实现后运行相关测试，在 PR 或提交说明中记录准确命令和结果。
- 修改算法时，补充有意义的合成数据、已知答案、边界情况及参考模型对比测试。
- 修改数据处理时，检查维度、标签、单位、时间对齐、分组互斥及仅在训练集上拟合的变换。
- 重启并按顺序运行修改过的 Notebook，移除私人路径、原始数据大段输出和大型结果。
- 行为变化时同步更新安装与配置说明。
- 检查差异中是否包含密钥、原始数据、参与者元信息、模型二进制文件或缺乏依据的结论。
- 小型指标与图表放在 `results/summaries/` 和 `results/figures/`；大型本地文件放在 `results/runs/`。

目前未配置自动化持续集成（CI）或强制分支保护。审阅时应核对实际验证证据，不能把尚未实现的检查写成已经通过。
