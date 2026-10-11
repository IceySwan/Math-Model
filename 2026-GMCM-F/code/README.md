# 正式代码

本目录保留四问的定稿算法、绘图、数值复核与论文辅助程序，共 48 个 Python 文件。`q2/protocol.json` 和 `q3/protocol.json` 保存算法配置。

| 路径 | 用途 |
| --- | --- |
| `paths.py` | 统一管理附件、输出与论文路径 |
| `q1/` | 质量评分、冲突消解和领域配比响应面 |
| `q2/` | 广义标度律、质量响应与联合参数抽样 |
| `q3/` | 预算优化、饱和判据与稳健配置 |
| `q4/` | 有序分位前沿、历史验证与情景预测 |
| `pipeline/` | 重算驱动、证据核验与源码快照 |
| `figures/` | 图表生成、核验与论文图件同步 |
| `verify/` | 数值一致性和模型复核 |
| `tools/` | 论文程序片段与排版辅助 |

## 本地输入

原始附件由环境变量 `F_ATTACHMENTS` 指定。附件根目录包含 `A_data_value/`、`B_scaling_laws/`、`C_efficiency_evolution/`；默认路径为仓库根目录下的 `real_attachments/`。

Q2 当前入口还读取以下本地辅助结果：

- `data/validation/q2_reference/results.json`
- `data/validation/q2_reference/p_term_sensitivity.csv`

这些辅助文件与原始附件均由使用者在本地准备；GitHub 发布内容限于源码与算法配置。各阶段产生的中间结果保存在 `data/`，后续阶段按 Q1、Q2、Q3、Q4 顺序读取上游结果。`pipeline/compare_preserved_runs.py` 等历史对照入口还需要相应的本地对照结果。绘图及数值复核入口使用运行结果作为输入。

## 环境与命令

Python 依赖版本保存于 `requirements.txt`，沿用原工作区清单。绘图使用 XeLaTeX 与宋体；部分 PDF 验收入口使用 Poppler 工具。

从仓库根目录进入 `code/` 后执行 PowerShell 命令：

```powershell
python -m pip install -r requirements.txt
$env:F_ATTACHMENTS = 'D:\F附件\real_attachments'

python q1\solve_q1.py --stage quality --output-dir ..\data\outputs\q1
python q1\solve_q1.py --stage mixture --output-dir ..\data\outputs\q1
# Q2 需先准备上面列出的本地辅助结果
python pipeline\rerun_core.py q2
python pipeline\rerun_core.py q3
python pipeline\rerun_core.py q4_fit
python pipeline\rerun_core.py q4_forecast
python pipeline\rerun_slowdown.py
python pipeline\finalize_evidence.py
```

论文路径已在此发布副本的 `paths.py` 中对齐为 `paper/`，附件默认路径对齐为 `real_attachments/`。求解与绘图逻辑沿用正式源码。
