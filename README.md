# Qlib 能力验证 · 本机 A 股日线

对 [microsoft/qlib](https://github.com/microsoft/qlib) 做一次**能证伪的功能验证**：用它自己的六层
闭环（数据 → 表达式 → 因子 → 数据集 → 模型 → 回测）在**本机已有的 A 股日线库**上完整跑一遍，
每条判据都必须"能红"，每个数字都必须能追到一个实测来源。

本轮判据是**能否跑通**，不是收益好不好。结论提前说：六层全部跑通了，但**这个窗口里没有可检测的
横截面信号**，所以回测那条曲线不是 alpha 证据（理由见下，且是可核对的数字）。

- 环境：Windows 11 + Python 3.12.10 venv，`pyqlib==0.9.7`（官方 `cp312-win_amd64` wheel，
  **不需要 WSL/Docker** —— 这条是实测下载验证的，不是查文档得来的）。
- 数据源：`D:/Desktop/个人事项/codex/量化分析/data/market.duckdb`，**全程只读**（见"安全边界"）。
- 版本事实：numpy 2.5.4 / pandas 3.0.6 / lightgbm 4.7.0 / duckdb 1.5.5 / mlflow 3.17.1。
  Qlib 官方矩阵写的是 numpy/pandas 的旧版本，这里**跑在更新的 2.x/3.x 上且读取路径正确**
  （与源库逐日逐字段交叉校验通过），这是本轮一个独立的验证结论。

## 目录三分：tools 是产线、tests 是判据、research 是结论

```
config/settings.py     唯一的常量出口（路径、区间、切分、标的池规则）
tools/                 正式运行：只读导出 CSV、建 Qlib 二进制库与标的池
tests/                 功能判据：六层各自的断言 + 守卫守卫（不产出研究结论）
research/              研究产出：全流程跑一遍，唯一允许写 artifacts/ 的地方
artifacts/             研究输出（gitignored）：factors/predictions/report/metrics.json/FINDINGS.md
qlib_data/             Qlib 二进制库（gitignored，可由 tools/ 重建）
vendor/qlib/           上游源码，只为拿 scripts/dump_bin.py（gitignored；wheel 里不带这个文件）
docs/superpowers/      设计文档与 11 任务计划（含全部实测坑）
```

依赖单向：`research/` 与 `tools/` 可以读 `config/`，`tests/` 可以读前三者，**`research/` 不 import `tests/`**
（测试不是库）。`tests/` 与 `research/` 都不许写 `qlib_data/` 或源数据库。

## 怎么跑

```bash
./.venv/Scripts/python.exe tools/smoke_test.py                  # 先证安装：导入 + Cython 扩展 + 装配点
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # 见下方"依赖"
./.venv/Scripts/python.exe tools/export_bars.py                 # 只读导出 → tools/_export/csv/
./.venv/Scripts/python.exe tools/build_qlib_data.py             # 建库 → qlib_data/
./.venv/Scripts/python.exe -m pytest tests/ -q                  # 六层判据（默认 warning=error）
./.venv/Scripts/python.exe research/alpha158_lgbm_backtest.py   # 全流程 → artifacts/
```

`pytest` 默认就是严格 warning 模式（`pytest.ini` 里 `filterwarnings = error`），只豁免一条点名的
上游噪音，理由写在 `pytest.ini` 注释里。想看完整警告列表：`pytest -W default`。

实测状态（2026-10-16）：`pytest tests/` 在严格模式下 **102 passed**，
除那一条被点名的上游 RuntimeWarning 外**没有任何其他警告**。
计时三次实测分别是 **535s / 831s / 272.73s**（同一批用例，差别只在当时机器负载 ——
831s 那次与研究脚本同时在跑，272.73s 那次是刚推完之后单独跑的）。
**别把任何一次当基准耗时**，写成单一数字会让人拿它判断"变慢了"，那是假信号。
配置本身也做了变异验证（故意注入一条 `DeprecationWarning` → 当场 FAILED），
所以"error 模式生效"不是读配置读出来的，是试出来的。

**两条独立的验证链互为对照**：`research/` 自己重算全部结论数字（不 import `tests/`、
不抄常量），结果与 `tests/` 的钉桩逐项一致 —— 51 轮、谷底第 1 轮、每段 5 档分值、
两段 IC 与零分布 sd/分位、并列规模、`cum_return`、混合费率；
唯一差异在 `cum_return` 的第 15 位小数（浮点求和顺序级），比带宽小十一个数量级。
也就是说这条曲线是被两边各自量出来的，不是抄出来的。

## 六层实测结论

| 层 | 判据文件 | 结论 | 关键实测数字 / 判据 |
|---|---|---|---|
| 1 数据 | `tests/test_01_data_layer.py` | 通过 | Qlib 日历与源库**逐日集合相等**（先按有序列表断言非空+严格递增再降为集合）；比对窗口以库自身截止日定界；停牌日 "Qlib NaN ↔ 源库 0 行" 配对成立 |
| 2 表达式 | `tests/test_02_expression_ops.py` | 通过 | `Ref` 精确移一天、`Mean`/`Std`/`Rank` 与 pandas 手算逐点比对；跨标的批量 rank 仍落在 [0,1]；全部表达式对齐在同一长度 |
| 3 因子 | `tests/test_03_alpha_factors.py` | 通过 | Alpha158 特征帧 `(159208, 158)`、原始未归一化含 NaN（NaN 单元格 306421，活列 147213），`VWAP0` 整列 NaN；标签表达式就是 `Ref($close,-2)/Ref($close,-1)-1`；fit-window 统计量泄漏守卫双向 |
| 4 数据集 | `tests/test_04_dataset_split.py` | 通过 | 三段非空、日期区间互斥且有序、同一标的不跨段、feature 与 label 索引一致；**留出段尾部不出现在任何段里**；test 标签恰好向留出尾部伸 2 行/标的（段末 660 行 = 330 只 × 2 天） |
| 5 模型 | `tests/test_05_model_train.py` | 通过 | **51 轮、谷底在第 1 轮**；每段只有 **5** 个不同分值；IC 落在 200 次置换零分布之内 |
| 6 回测 | `tests/test_06_backtest.py` | 通过 | 181 个交易日全覆盖、报表列名与 `positions` 结构钉住、成本/换手单位自洽、并列规模全部钉住 |

模型层与回测层的数字（定种子 `seed=66, deterministic=True` 之后重基线；两次独立 session
逐字节相同，所以可复现）：

- `train l2 0.99387966 → 0.97408369`（非递增），`valid l2` 第 1 轮见底 `0.99683246`、末轮 `1.00050671`
  —— 越过"永远预测 0"的零信息基线 1.0。**train 降 / valid 升 = 窗口内无可学信号，不是早停掐早了。**
- pooled Spearman IC：valid `-0.00127837`、test `-0.00410591`；200 次置换零分布 sd
  `0.00578831` / `0.00337610`，模型分位 `0.265` / `0.785`，**两段都没跑赢 95 分位**。
- 打分退化：`topk=30` 而 test 段 181 个交易日**每天只有 5 档分值**；分数线上并列标的数
  中位 47、最多 **249** 只；**107/181 天的整份 topk 名单由同分挑谁决定，而不是排名决定**。
- 回测装配自洽：`total_cost` 与 `total_turnover` 都是**金额**、`cost`/`turnover` 都是**率**，
  每单位成交额的混合费率必须落在配置的 `[5bp, 15bp]` 内。

**为什么回测收益不能当本事**：`cum_return` 的钉桩值是 `+12.98%`（基准 `-5.88%`），而**同一份数据、
同一套参数、只把 RNG 定下来之前**这条曲线是 ≈0.00%。换个随机种子漂十几个百分点，是"没有稳定信号"
的反证。研究脚本 `research/` 会重算一遍这些数字并写进 `artifacts/FINDINGS.md`，不抄测试里的常量。

## Qlib 功能地图（按包内 `.py` 文件数实测）

| 包 | `.py` 数 | 本轮状态 | 说明 / 未覆盖原因 |
|---|---|---|---|
| `qlib.data` | 20 | **已实跑** | `D.calendar / features / D.instruments / list_instruments`，二进制 store 读取路径 |
| `qlib.backtest` | 12 | **已实跑** | `backtest()`、`SimulatorExecutor`、`report.Indicator`、账户与成本撮合 |
| `qlib.contrib` | 102 | **部分实跑** | 实跑：`data.handler.Alpha158`、`data.ops` 表达式算子、`model.gbdt.LGBModel`、`strategy.signal_topk`(实为 `signal_strategy`)。其余 zoo 成员（`CatBoost`/`XGB`/`TFT`/`ALSTM`/`GATs` 等）**仅导入甚至未安装依赖**，qlib 自己在 import 时就打了 `ModuleNotFoundError ... skipped` |
| `qlib.model` | 18 | **仅导入** | 基类与 trainer 骨架；本轮只有一个模型实例，未覆盖训练器矩阵 |
| `qlib.workflow` | 16 | **已实跑但受限** | `R.set_uri / start / end_exp` 走通了，但 mlflow 3.x 的文件后端被置为维护模式（见"集成坑"）；`Recorder` 的完整实验管理、`GroupExpManager` 未覆盖 |
| `qlib.utils` / `qlib.cli` / `qlib.strategy` | 11 / 3 / 2 | **仅导入 / 未覆盖** | `cli` 的 yaml 全流程入口本轮没用（我们走 API 而不是 `qrun`） |
| `qlib.rl` | 38 | **未覆盖** | 强化学习交易子系统：需要额外依赖与分钟级/撮合环境，本轮"六层闭环 + 日线"的范围不含它 |

未覆盖 ≠ 不能用，是"本轮没有为它产出可证伪判据"。**明确清单**（不含糊）：

- `qrun` + YAML 全流程入口（本轮走 API，没用官方装配路径）；
- mlflow 实验追踪 UI / `Recorder` 与 `GroupExpManager` 的完整实验管理（只走通 `R.set_uri` 最小路径）；
- `qlib.contrib.rolling` 滚动训练（`BaseRolling` / `RollingAlgorithm`）；
- `cvxpy` 组合优化器与 Portfolio 层的其他组合构建方式；
- PIT 财务数据链路（`dump_pit`）—— 本机库只有日线行情字段，没有财务点值；
- `qlib.rl` 订单执行 RL 全套（38 个文件，依赖 gym/torch）；
- 20 余个 PyTorch 模型（未装 torch；qlib import 时自己就打了自己跳过的日志）；
- 分钟/高频数据与嵌套决策（`NestedExecutor`）。

一个容易看错的能力点：表达式层的 `Rank` **不是**横截面排名，它是**标的内的滚动百分位**；
横截面 Rank/标准化只在 processor 层（`CSRankNorm` / `CSZScoreNorm`）间接经过。

要扩展的话，优先级最高的三块是：`qrun` + yaml（检验官方装配路径）、
嵌套决策 / 高频回测（检验 `NestedExecutor`）、Alpha360 + 其他模型对照（检验覆盖面）。

## 已知弱点与可见性缺口（不藏，逐条说明）

最终审阅 triage 里的 B 组裁定。它们不阻塞交付，但读代码的人必须先知道：

- **有些计数器只在失败时才可见**：数据层的 `nan_verified`（"Qlib NaN 且源库 0 行"的停牌配对）
  绿色运行里看不到数目。理论上一个被放宽的 instruments 区间也会产出"NaN 且 0 行"而被判为已验证；
  收紧办法是"该停牌日前后都必须有源库 bar"，本轮没做。
- **有少数断言是同义反复**，不可能独立失败（`write_csvs` 的文件数断言、`test_03` 的两条
  被上面三条算术蕴含的断言）。它们提供的是特定性而不是覆盖，别把它们当防线。
- **几条边界在当前数据上不可达**：`QUERY` 没有 `ORDER BY`（若有重复行，取哪条不确定，实测 dup=0）；
  `<= 0` 判定对 NaN 为 False（实测 NULL 价=0）；`to_qlib_code` 收到无点号输入会静默返回非法码
  （实测所有调用点都先过 `classify_symbol`）。
- **`alpha158_kwargs` 交出的是可变 dict 本体**而非副本；当前没有测试改写它，所以不成问题，
  但改测试的人要留意。
- **一批 store-coupled 字面量**会在重建库时变红（见下一节），这是设计意图不是脆。

## 重建库时会撞红什么（故意设计成撞）

`tests/` 里有一批** store-coupled 字面量**：`159208`（Alpha158 交付行数）、`306421` / `147213`
（NaN 单元格数）、`660` / `330`（test 段末两天的行数与有价标的数）、以及模型层与回测层的全部钉桩
（见"数字出处规则"）。**重跑 `tools/` 建库、加历史深度、或改 universe 都会让它们变红**，
而且红得自解释 —— 那是设计意图（提醒"结论的前提变了"），不是测试脆。
正确处置顺序：先重跑 `research/` 拿到新的 `metrics.json`，再按新的实测值改常量，
**不许反过来先改常量让测试闭嘴，更不许放宽带宽**。

## 集成坑（全部是本机实测踩出来的，不是文档转述）

1. **mlflow 3.x 拒用文件系统后端，qlib 0.9.7 默认就走它。** 任何调用 `model.fit()` 的入口都必须
   同时满足两件事：① 在 import 前设 `MLFLOW_ALLOW_FILE_STORE=true`；② 用 **qlib 自己的**
   `R.set_uri(<本地目录>)` 指定跟踪目录 —— **不能用 `mlflow.set_tracking_uri()`**，因为
   `qlib/workflow/recorder.py:337` 起 recorder 时会用 ExpManager 的 `default_uri`（默认 None）
   原样覆写回去，实测仓库根因此长出过 `mlruns/` 目录。`fit` 的崩溃点在 qlib 侧
   `qlib/contrib/model/gbdt.py:90` 的无条件 `R.log_metrics`：树已经训好了，异常死在落盘上。
2. **`dump_bin.py` 不在 wheel 里**，必须从源码仓库取（本仓库 `vendor/qlib/scripts/`）。
   它的 fire CLI 参数名是 **`--data_path`**，不是 `--csv_path`（照记忆写会直接失败）；
   且只产出 `all.txt`，**不会**生成分市场池文件 —— 所以 `cn_stocks/cn_etf/cn_index` 三个池
   是 `tools/build_qlib_data.py` 自己按代码前缀写出来的。
3. **`D.calendar(top_symbol=...)` 在 0.9.7 不存在**（`BaseProvider.calendar` 签名只有
   `start_time/end_time/freq/future`）。旧语义"用该标的在 instruments 里的起止切全局日历"
   在测试里按同义实现（`tests/test_01_data_layer.py::D_calendar`），取的仍是 Qlib 自己的起止数据。
4. **`qlib.contrib.strategy.signal_topk` 这个模块路径不存在**，正确是 `signal_strategy`。
5. **`LGBModel.predict` 原生返回无名 Series**（`gbdt.py:92-96`），`predictions["test"].iloc[:, 0]`
   作用在 Series 上是硬 `IndexError: Too many indexers`。交付形状统一为**单列 `score` 的 DataFrame**。
6. **`Booster.num_trees()` 会误导**：它区分不了"训了 1 轮"和"训了 51 轮"，
   所以"早停是否起作用"只能看 `best_iteration` + 逐轮 evals 曲线。
7. **qlib 默认配置不可复现**：`subsample/colsample < 1` 而无种子，实测同一份库两次运行
   `best_iteration` 在 2 与 52 之间跳、下游累计收益漂 20.7 个百分点。补 `seed` + `deterministic=True`
   之后（`deterministic` 会强制 row-wise 划分，否则多线程仍抖）两次独立 session 逐字节相同。
8. **`D.calendar()` 返回 numpy 数组**：`assert seg_days` 会 `ValueError: truth value ... ambiguous`，
   判空必须 `len(...) > 0`。
9. **mlflow 3.x 的文件后端在含非 ASCII 字符的路径下建不出 run**（2026-10-16 最小复现）。
   同一个脚本只差 URI 路径：`R.set_uri("<仓库内>/mlruns")` → `R.start()` 抛
   `MlflowException: Run '<id>' not found`（`file_store.py:777 → 1209 → 810`：目录建得出来、
   `_get_run_info` 找不到它）；换成 `C:\Users\...\Temp\...` 就正常。
   本仓库路径含中文（`个人事项`），所以**任何仓库内的 tracking 目录都会踩**。
   `tests/` 侧因为把跟踪目录放在 pytest 临时目录而一直没暴露这个差异，
   `research/` 第一次按"产物都放 artifacts/"的直觉写就撞上了 —— 现在跟踪目录也走临时目录，
   `artifacts/` 里只留有研究意义的文件。
10. **handler 的时间窗必须覆盖所有 segment，否则 qlib 一声不响地给你空帧。**
    `Alpha158(end_time=SEGMENTS["valid"][1])` 配 `DatasetH(segments=含 test)`：test 段
    `prepare` 出来是 0 行、`predict` 也是 0 行，回测照样"成功"跑完并交出一条没有交易的空曲线。
    本轮计划原文就带的这个 bug，直到研究脚本对空数组取 `min()` 才炸出来。
    现在 `research/` 里紧跟一条 fail-closed：任何一段 prepared 行数为 0 立刻退出并说明原因。
11. **`atexit` 钩子会在解释器退出时往已关闭的 stdout 写日志**，表现为一串
   `--- Logging error --- ValueError: I/O operation on closed file`。在 fixture 内当场
   `R.end_exp(...)` 收尾即可（qlib 自己的注释说实验结束后钩子不再生效）。

## 安全边界（硬约束）

- `market.duckdb` 一律 `read_only=True`、只 `SELECT`，任何写入路径都不存在；
  **不向 `量化分析/` 目录写任何东西**（导出目标在 `tools/_export/csv/`）。
- `qlib_data/` 由 `tools/` 建、被 `tests/` 只读消费；测试改动它是不可能的（没有写句柄）。
- 时间口径：库内 `ts` 是 `TIMESTAMPTZ`，取日必须
  `CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE)`；裸 `CAST(ts AS DATE)` 会按 UTC 切日、
  整体错一天（本仓库踩过，已在 `tools/export_bars.py` 与 `tests/` 两侧钉住）。
- `bars.asset_type` 字段**不可信**（实测同一代码既标 stock 又标 index），所以分类一律走
  代码前缀规则（`config/settings.py::classify_symbol`），指数不会被放进可交易池。
- 只保留 `market='CN'` 且 `frequency='1d' AND adjust='qfq'` 的行 —— 混进非 CN 标的会把境外
  交易日的空 bar 灌进全局日历，把区间污染成"有日历无数据"。

## 数据容量：本轮结论的边界

主系统的 A 股日线**从 2021-08-26 才开始有**，2021/2022/2023 各只有 1 只标的，可交易 universe
只有 330 只（test 段末两个交易日只剩 330 只有价）。所以：

- 官方示例常用的 2018-2022 训练窗在这份库上**不存在**，切分只能落在 `2025-01-01 .. 2026-09-30`；
- "窗口内无可学信号"**不能**外推成"Qlib 找不到 alpha"或"A 股无效"，它只描述这一段数据；
- 想拿这份库做真实研究，第一件事是**加历史深度和标的数**，不是调模型。

## 依赖

`pyqlib`、`numpy`、`pandas`、`lightgbm`、`duckdb`、`pytest`、`scipy`、`pyarrow`、`mlflow`。
`requirements.txt` 由本机 venv 实测版本导出（见文件），装不上时以
`pyqlib==0.9.7` + Python 3.12 为准。

## 数字出处规则（本 README 里每个数怎么来的）

规则：**要么能指到一个测试常量名，要么能指到一次可重跑的实测命令，否则不许写进本文**。
按这个标准，本文的数字分成三类：

1. **钉在测试里的实测常量** —— 直接看符号名，改数据/改库/改模型就会红：
   `tests/test_05_model_train.py`（`ROUNDS_TRAINED` `BEST_ITERATION` `TRAIN_L2_*` `VALID_L2_*`
   `POOLED_ABS_IC` `NULL_SD` `DISTINCT_SCORES_CEILING`）·
   `tests/test_06_backtest.py`（`CUM_RETURN` `MEAN_TURNOVER` `SUM_COST` `MAX_HOLDINGS`
   `DAILY_DISTINCT_SCORES` `CUT_TIE_MEDIAN` `FULLY_TIED_DAYS`）·
   `tests/test_03_alpha_factors.py`（`DELIVERED_NAN_CELLS` `SPEC_LABEL_EXPR`）。
   这些常量的**带宽**（`*_BAND`）与定种子之前完全一致 —— 只有被测量随配置变更重取过。
2. **`artifacts/metrics.json`** —— 由 `research/alpha158_lgbm_backtest.py` 每次重算，
   包括 IC、零分布分位、并列规模与全部回测指标。本文只指路径，不复制数字，避免两处漂移。
3. **当场可重量的环境事实** —— 版本来自 `pip freeze`（见 `requirements.txt`），
   功能地图的文件数来自 `find .venv/.../qlib/<pkg> -name "*.py" | wc -l`（合计 230 个 `.py`），
   库容量来自 `D.list_instruments(D.instruments(market="all"))`。

**明确"未实测所以不写"的东西**：实盘成本冲击、滑点影响、模型的样本外收益预期、
以及任何"A 股是否有效"层面的判断 —— 本轮数据和判据都不支持这些结论。

## 未纳入版本控制的东西

`qlib_data/`、`tools/_export/`、`artifacts/`、`mlruns/`、`vendor/`、`.venv/` 全部 gitignore。
过程台账（每条裁定的现场证据、被撤回的说法、遗留项 triage）在
`.superpowers/sdd/2026-10-15-qlib-capability-verification/progress.md` ——
SDD 工具链默认自忽略该目录，所以它在**本机**而不在远端；本 README 里每个数字都能对着
`tests/` 的常量名找到出处，不依赖台账也能核对。
