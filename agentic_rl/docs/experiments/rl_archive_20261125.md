# 2026-11-25 RL 版本备案

此备案分支保存 30 任务 routed Judge、官方满分组处理、查询轮筛选、独立学习率的可选 Muon 及其固定版本 veRL 补丁。默认优化器是 AdamW。服务器并列版本的检查入口为 `agentic_rl/scripts/run_routed_release.sh`，默认输出训练命令，显式传入 `--optimizer muon` 选择 Muon。

| 原部署记录 | 值 |
| --- | --- |
| 现用入口解析路径 | `/root/autodl-tmp/agentic-rl-train30-20261115/launch_rl_train30.sh` |
| 该入口 SHA-256 | `e4c602faa84c1902b84c6ba585ccc9b10c812b321542340bf346bc548c23c267` |
| 历史 20 步实验记录的部署提交 | `3343941551e59fc9ec1cde87ececa85615491bb2` |
| 20 步实际 runtime_config.yaml SHA-256 | `0531ca964516a6251117bcbd0b7a4944379e2f65a7e77b60530955f6327a354c` |
| Tau2 / veRL | `a2c024725189473d2d7cea3a5cfdbcc67478e41f` / `483b8a009ba3a97563edee3a19887e4862b8094a` |
| 服务器历史源码、配置与 veRL 差异存档 | `/root/autodl-tmp/archives/rl_before_routed_muon_20261125/server-code-before-routed-muon.tar.gz` |
| 归档 SHA-256 | `119616f7f266387cd51975dc3c07fd84a7f9069d981f990273ef6644b44f24a8` |

归档内的清单核验了 481 个条目，包含原入口、20 步运行配置以及当时服务器 veRL 的实际源码差异。GitHub 基线为 `f3f8227`，本次候选从 `ec2521b` 上的工作树整理。服务器凭据由原有环境文件在运行时提供。
