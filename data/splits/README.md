# 固定数据划分（R06）

此目录只提交经检查的划分元信息，不放信号、特征或私人身份资料。建议用 JSON 保存 schema_version、dataset_id、来源清单校验值、策略、种子及每条记录对应的 train/validation/test。

用户留出时检查 subject_id 互斥；会话留出时检查 (subject_id, session_id) 组合键互斥。数据变化后重新审计，不能自动沿用旧文件。使用公开数据中已有的匿名标识，文件内注明分组协议。

精确字段由 R01/R06 实现并与[架构契约](../../docs/architecture.md)保持一致，当前没有实际划分。
