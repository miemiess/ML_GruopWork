# 贡献与协作开发流程

先阅读[需求清单](docs/requirements.md)和[架构约定](docs/architecture.md)，选择一个可独立验收的需求。成员不固定角色，一个分支通常完成一个需求或事先约定的子任务。

仓库本轮清理后只保留 `main`。后续由组员按任务自行建立短期分支，合并后删除；不长期保留每人一个分支。本次规划直接提交主线，后续功能开发遵循以下流程。

## 1. 认领需求，让大家看到进度

[GitHub Issues](https://github.com/miemiess/ML_GruopWork/issues) 是任务状态的唯一记录位置，需求文档定义范围与验收标准，不维护另一套易过期的进度表。

1. 在需求清单选定编号，例如 R08（时域特征）。
2. 在 Issues 搜索 `R08`，先检查是否已有认领记录；已有则协调，不重复开工。
3. 没有时点击 **New issue**，选择“开发任务认领”模板；标题为 `[进行中][R08] 实现 RMS、MAV 与 WL`。
4. 在正文填写姓名、GitHub 账号、需求编号、分支、依赖、交付路径及预计完成时间。预计时间由本人填写。
5. 有权限时将 **Assignees** 设为自己；无权限则在 Issue 留下认领信息，请维护者确认和分配。
6. 多人合作先拆分子任务，约定文件边界；不要让两人同时修改同一公共接口而不沟通。

状态统一为：**待认领 → 进行中 → 待审阅 → 已完成**；依赖未解决时使用**受阻**。状态写在标题前缀，受阻时正文说明阻塞项及下一步。提交 PR 时贴链接并改为待审阅；合并并达到验收条件后关闭 Issue，完成状态以关闭记录为准。重新打开时说明原因并恢复适当标题。

进度更新在同一 Issue 中写明“已完成、当前工作、阻塞、相关提交/PR”。不要将没有证据的任务标记完成。停做或转交时更新认领信息，便于别人接手。

## 2. 首次获取仓库与环境

首次参与时执行：

```bash
git clone https://github.com/miemiess/ML_GruopWork.git
cd ML_GruopWork
```

按照[README 环境安装](README.md#环境安装)配置 Python。克隆只执行一次，后续在已有仓库内操作。有写权限的组员使用本仓库分支；没有推送权限时联系维护者添加协作者，或从个人 Fork 发 PR，不共用账号凭据。

## 3. 从最新 main 建立自己的分支

先检查工作区：

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c feature/r08-time-features
```

上例用于 R08，实际开发请替换成自己的编号和任务名。命名建议为 `feature/rNN-short-name`，修复使用 `fix/rNN-short-name`，纯文档使用 `docs/rNN-short-name`。同一需求多人开发时增加子任务后缀。

切换前如有未提交工作，先在原任务分支提交，或使用 `git stash push -u -m "临时保存任务"`，以后在正确分支恢复。若 `pull --ff-only` 失败，先查看分叉原因，不使用强制重置覆盖工作。

## 4. 按模块开发与自检

- 根据[模块职责表](docs/architecture.md#分区与职责)进入对应目录，不另建一套完整项目。
- 先阅读依赖需求和公共接口；依赖未合并时使用符合契约的合成数据，不能自行改变约定。
- 可复用函数写在 `src/emginsight/`，Notebook 和 demo 只调用公共接口。
- 实现和测试一起提交。真实数据、模型文件与本地环境不要加入 Git。
- 有跨模块影响时先在 Issue 说明，再同步更新架构文档。

实现测试后运行对应模块测试，例如 R08：

```bash
python -m pytest tests/unit/features
git diff --check
git diff
git status
```

当前尚无算法测试；测试目录为空时 pytest 的退出码 5 表示没有测试，不能写成“测试通过”。模块落地后必须提供需求清单指定的有效测试。

## 5. 提交并推送分支

先查看差异，再添加本次任务涉及的具体文件：

```bash
git add src/emginsight/features/time_domain.py tests/unit/features/test_time_domain.py
git commit -m "feat(R08): 实现时域肌电特征"
git push -u origin feature/r08-time-features
```

以上文件名是 R08 实现后的示例，当前未实现时不会存在。后续继续修改可再次 commit，随后执行 `git push`。**一个 PR 可以包含多次提交；推送同一分支会更新同一个 PR，不需要每次重新创建。**

## 6. 同步其他人的修改与处理冲突

在自己的任务分支上，先提交或暂存当前改动，然后：

```bash
git fetch origin
git merge origin/main
```

无冲突时运行受影响测试并推送。存在冲突时：

1. 用 `git status` 找到冲突文件，与相关模块成员确认正确内容。
2. 编辑并移除 `<<<<<<<`、`=======`、`>>>>>>>` 标记，保留经过确认的组合。
3. `git add` 已解决文件，再执行 `git commit`。
4. 重新运行相关测试，执行 `git push`。

若暂时无法解决，可使用 `git merge --abort` 返回合并前状态。不要用“全部采用我的版本”掩盖接口冲突。默认采用 merge 同步，不要求新手重写提交历史。

## 7. 创建 PR 并请求审阅

在 GitHub 打开 **Pull requests → New pull request**，选择：

- **base：main**
- **compare：自己的任务分支**

标题示例：`[R08] 实现时域特征与已知答案测试`。按照 PR 模板写明需求编号、变更文件、公共接口、验收证据、测试命令与结果、剩余限制。完全完成某个任务时，在正文写 `Closes #实际Issue编号`；若只完成子任务，使用 `Refs #实际Issue编号` 并说明未完成部分。

需要提前讨论时创建 Draft PR；准备好后标为可审阅，请至少一位其他组员检查。审阅者重点检查接口一致性、数据泄漏、测试证据和需求覆盖。根据反馈继续在原分支提交和推送。

## 8. 合并到 main

由有写权限的成员确认：需求验收通过、相关测试通过、冲突已解决、至少一位其他成员完成审阅。没有自动 CI 时必须附本地验证结果，不能省略。

通常使用 GitHub 的 **Create a merge commit / Merge pull request**，保留分支提交历史，便于新手安全清理本地分支。若团队选择 Squash 或 Rebase 合并，先在 PR 说明。此流程是协作约定，当前仓库没有新增强制保护规则。

合并后检查 Issue：完整任务关闭并记录验收链接；部分任务继续保持打开。不要仅因 PR 被关闭就把需求标成完成。

## 9. 删除完成的分支并更新本地

确认 GitHub 显示 PR 已合并后，点击 **Delete branch** 删除远端任务分支，再执行：

```bash
git switch main
git pull --ff-only origin main
git branch -d feature/r08-time-features
git fetch --prune
```

若未在网页删除远端分支，可在确认 PR 已合并后执行：

```bash
git push origin --delete feature/r08-time-features
```

`git branch -d` 拒绝删除时先检查，尤其是 Squash/Rebase 合并后原提交可能不在 main 历史中。确认 PR 内容均已合入且没有独有工作后，才可用 `git branch -D feature/r08-time-features` 清理本地分支；不确定就保留并询问维护者。禁止删除 `main` 或强制推送共享主线。

开始下一个任务时，从最新 main 新建分支，不复用已经合并的分支。

## 10. 公共完成标准

每项需求均须有：实现、接口说明、针对性测试、相关配置、验收证据和 Issue/PR 关联。算法及实验必须明确自定义实现与库参考，保留训练/验证/测试边界；数据受许可与可行性限制时记录证据，经小组确认后更新范围，不把未完成研究伪装成结果。

官方参考：[Issue 负责人设置](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/assigning-issues-and-pull-requests-to-other-github-users)、[PR 合并](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/merging-a-pull-request)、[Git 分支删除](https://git-scm.com/docs/git-branch)。
