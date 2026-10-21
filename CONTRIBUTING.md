# 怎么一起开发

先在 [README](README.md) 选任务，在 [Issues](https://github.com/miemiess/ML_GruopWork/issues) 写“姓名、准备做什么、目前进度”。已有同类任务时先沟通。环境安装也在 README 中。

下面以“手势实验”为例，分支名可换成自己的任务名。

## 1. 建自己的分支

在仓库目录执行；先确认没有没保存的修改：

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c feature/gesture
```

所有成员共享 main，平时在自己的任务分支开发。有未提交工作时先提交到原任务分支，再切换；不要强行覆盖。

## 2. 开发、检查、提交

按 README 的目录分工放代码，先跑通自己的例子和相关测试。确认 `git status` 中只有本次要提交的文件，再执行：

```bash
git diff
git add .
git commit -m "完成手势实验的第一版"
git push -u origin feature/gesture
```

后续继续修改仍在这个分支 commit，然后 `git push`。**可以提交很多次，通常一个任务只建一个 PR。** 原始数据、虚拟环境和大型模型不要上传。

## 3. 让组员检查并合并

在 GitHub 选择 **Pull requests → New pull request**：

- base 选择 `main`，compare 选择自己的分支。
- 写清“完成了什么、怎么运行、还有什么问题”，贴上任务 Issue。
- 请另一位组员看一下，修改有问题的地方；推送后会自动更新同一个 PR。
- 确认能运行后，由有权限的成员点击 **Merge pull request** 合并。建议使用默认合并提交方式。

完成整个任务后关闭对应 Issue；只完成一部分就继续保持打开，并更新进度。

## 4. 合并后收尾

GitHub 显示已合并后，点击 **Delete branch** 删除远端分支，再在本地执行：

```bash
git switch main
git pull --ff-only origin main
git branch -d feature/gesture
git fetch --prune
```

删除被拒绝时先保留并核对，不强行删除。下一个任务从最新 main 重新建分支。

## 5. 开发期间别人更新了 main 怎么办？

先把当前改动提交在自己的分支，再执行：

```bash
git fetch origin
git merge origin/main
```

没有冲突就重新检查并推送。有冲突时，和修改该文件的同学一起确认内容，解决后 `git add`、`git commit`、`git push`。暂时不会解决可用 `git merge --abort` 取消这次合并，不要直接覆盖别人的代码。

## 6. 提交要求

提交时一定要写Message,简单注明提交内容
请维护 docs/开发日志.md ，格式为:修改者-时间-开发内容
