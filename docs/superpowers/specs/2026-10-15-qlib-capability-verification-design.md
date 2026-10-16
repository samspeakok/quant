# Qlib 能力验证工程 — 设计

日期：2026-10-15
范围：`D:\Desktop\个人事项\quant-l\agent space`
状态：已与用户确认五个设计小节，批准实施（含 git init）

## 1. 目标与非目标

**目标**

用本机已有的真实行情数据，逐层验证 Microsoft Qlib 的六层核心闭环是否真的可用，并把验证结论沉淀成可重复执行的资产：

1. 数据层 — 本地数据导入 Qlib 二进制存储并被正确读回
2. 表达式层 — Qlib 表达式引擎计算因子
3. 因子层 — Alpha158 / Alpha360 产出因子矩阵
4. 数据集层 — DatasetH 切分训练/验证/测试
5. 模型层 — LightGBM 训练并预测
6. 回测层 — TopkDropout 策略 + 撮合 + 收益与归因报告

第 6 层（回测）是终点，`research/` 里跑出一个真实可复现的策略模板。

**非目标**

- 不做 qrun YAML 工作流与 mlflow 实验管理（工程包装，留待后续）
- 不做滚动训练、cvxpy 组合优化、PIT 财务数据、强化学习订单执行
- 不下载官方社区行情包（本机数据已足够且质量更高）
- 不修改、不写入 `codex\量化分析` 下任何文件

## 2. 已核实的环境事实

这些是实测结果，不是假设，后续实现不得与此冲突：

| 事实 | 值 | 核实方式 |
|---|---|---|
| Python | 3.12.10 (64bit) | `python --version` |
| pyqlib | 0.9.7 | 已安装 |
| 安装方式 | 官方 `cp312-cp312-win_amd64` wheel | pip 实际下载成功 |
| Cython 扩展 | `qlib.data._libs.rolling/expanding` 可加载 | smoke test |
| numpy / pandas | 2.5.4 / 3.0.6 | pip 解析结果 |
| 数据源 | `market.duckdb` 表 `bars` | 只读查询 |
| 原始行 | 465,006 行 / 715 标的 / 1980-01-04 → 2026-10-14 | SQL count |
| 可用行 | `frequency='1d' AND adjust='qfq'` = 360,089 行 / 708 标的 | SQL count |
| 资产构成 | stock 642 / etf 68 / index 18 / future 2 / fx 1 | SQL group by |
| `dump_bin.py` | **不在 wheel 内**，仅存在于源码仓库 `scripts/` | find 全盘确认 |

**已知未验证风险**：numpy 2.5.4 + pandas 3.0.6 是激进的新版本。离线导入与 `qlib.init()` 已通过，但真正读取 `.bin` 数据的路径尚未验证。本次工程的第 1 层测试即为该风险的裁决点。若失败，回退方案是新建固定 `pandas<3 numpy<2` 的并行环境。

## 3. 目录结构

```
agent space/
├─ .venv/              已有，Python 3.12 + pyqlib 0.9.7 + pytest
├─ vendor/qlib/        clone 源码；仅为 dump_bin.py 与读实现（gitignored）
├─ qlib_data/          【生成物】calendars / instruments / features（gitignored）
├─ tools/              一次性数据工程，唯一容忍脏数据的层
│   ├─ export_bars.py      duckdb 只读 → dump_bin 认的 CSV
│   ├─ build_qlib_data.py  调 dump_bin.py 生成 .bin + 建 instrument 分组
│   └─ _export/            【生成物】中间 CSV（gitignored）
├─ tests/              pytest，按层断言，只验证不产出
│   ├─ conftest.py         qlib_data 缺失时整体 skip 并给出提示
│   ├─ test_01_data_layer.py
│   ├─ test_02_expression_ops.py
│   ├─ test_03_alpha_factors.py
│   ├─ test_04_dataset_split.py
│   ├─ test_05_model_train.py
│   └─ test_06_backtest.py
├─ research/           正式运行，产出真实因子/模型/回测报告
│   └─ alpha158_lgbm_backtest.py
├─ artifacts/          【生成物】research 输出（gitignored）
├─ config/
│   └─ settings.py     数据源路径、日期区间、标的池、参数 —— 不在脚本里硬编码
└─ docs/superpowers/specs/
```

依赖方向单向：`tools/` → `qlib_data/` → `tests/` 与 `research/`。
`tests/` 和 `research/` 互不依赖，也都不触发 `tools/`。

## 4. 数据层契约

### 4.1 只读安全

`market.duckdb` 正被主系统写入（实测修改时间当天上午，同目录存在 `housekeeping.lock`）。因此：

- 所有 duckdb 连接一律 `read_only=True`
- 只 SELECT，不执行任何写、不执行 CHECKPOINT/VACUUM
- 产物只写入 `agent space`，绝不落到 `量化分析/`

### 4.2 过滤条件

只取 `frequency = '1d' AND adjust = 'qfq'`。

原因：库中混有 1,595 行 5m 数据和 103,322 行 `adjust='none'`（未复权）数据。混入不同频率会打乱交易日历；混入不同复权口径会让同一标的的价格序列出现断层，导致收益率计算静默错误。

`future`(2) 与 `fx`(1) 直接排除，样本太少且交易规则与股票不可比。

### 4.3 代码格式转换

`600519.SH` → `SH600519`，`000001.SZ` → `SZ000001`。取 `.split('.')[1].upper() + split('.')[0]`。

取 `split('.')[1].upper() + split('.')[0]`，即 `600519.SH` → `SH600519`，`000001.SZ` → `SZ000001`。

Qlib `region='cn'` 的符号约定为大写字母前缀 + 数字。`dump_bin.py` 具体是靠文件名还是靠某一列识别符号，**以第 4.4 节读源码的结果为准，此处不预先断言**。无法解析的后缀（如 wind 风格的 `.W`）记为脏数据并报告，不静默丢弃。

### 4.4 字段映射

| Qlib 字段 | 来源列 |
|---|---|
| open/high/low/close | 同名列 |
| volume | `volume` |
| amount | `amount` |
| factor | 固定 1.0（数据已是 qfq，无需二次复权） |

CSV 列名与日期格式**不凭记忆约定**。`dump_bin.py` 通过 `--date_field_name` / `--symbol_field_name` 接收列名，所以实现第一步是读 clone 下来的 `vendor/qlib/scripts/dump_bin.py` 源码，确认它要求的列名、日期格式与目录布局，再据此定 `export_bars.py` 的输出。本表只约定语义映射，不约定字面表头。

### 4.5 instrument 分组

分三个独立池，**绝不合并**：

- `cn_stocks.txt` — 642 只股票
- `cn_etf.txt` — 68 只 ETF
- `cn_index.txt` — 18 只指数

指数没有可成交价，若混入可选池会凭空抬高回测收益。研究默认标的池为 `cn_stocks`。

lake 中存在 `hs300_pit` 指数成分历史时点快照，本轮**不接入**（列为非目标），但在 `cn_index.txt` 注释里标注其位置，供后续消除幸存者偏差时取用。

## 5. 测试设计

六层，按依赖顺序编号。每层一个文件，可单独运行。

### 5.1 各层判据

| 层 | 判据（断言，而非"没报错就算过"） |
|---|---|
| 01 数据层 | `qlib.init(provider_uri=qlib_data, region='cn')` 成功；日历非空且严格递增；`D.features` 返回非空 DataFrame；**交叉校验见 5.2** |
| 02 表达式 | `Ref($close,1)`、`Mean($close,5)`、`Std($return,20)`、`Rank($volume)` 结果长度对齐、无全 NaN、手算样本值一致 |
| 03 因子 | Alpha158 / Alpha360 产出的特征列数**等于 handler 自身 `get_feature_config()` 声明的字段数**（用配置自证，不写死 158 这种魔数，避免改配置就误报）；无整列 NaN；`$close` 列与数据层读回值一致 |
| 04 数据集 | DatasetH 三个切片时间区间互不重叠，且 train 全部早于 valid 早于 test |
| 05 模型 | 预测输出行数等于输入样本数；预测值方差非零（排除"常数预测"这种假收敛）；IC 用双向判据：绝对值严格大于 0 且小于 0.5。下限排除纯噪声；**上限同样重要** —— IC 绝对值异常大几乎一定是数据泄漏（用了未来函数），那不该算"通过"而该报警 |
| 06 回测 | 报告 DataFrame 含收益与风险列，换手率与成本在合理区间，持仓数不超过 topk |

### 5.2 第 1 层的核心：与数据源交叉校验

这是本设计最重要的部分。`test_01` 不止验证"Qlib 能读出数据"，而是拿 Qlib 读出的收盘价**回查 duckdb 源表的具体数值**。

抽样方案固定，不写"随机抽 N 个"：

- 三个 instrument 池（stocks / etf / index）各随机抽 **min(20, 池内标的数)** 个 —— index 池仅 18 只，写死 20 会让测试直接失败。合计约 58 个
- 每个被抽中标的取其**首个交易日、末个交易日、以及中间随机 3 个交易日**
- 另加一组刻意构造的边界样本：日历中存在缺口的标的（长期停牌、中途上市）各 1 个

断言（相对误差 1e-6）：

```
qlib_close(SH600519, 2024-08-29) == bars.close WHERE symbol='600519.SH' AND ts='2024-08-29'
```

**符号方向**：Qlib 侧是 `SH600519`，duckdb 侧是 `600519.SH`。交叉校验必须在测试内独立做一次反向映射，**不能复用 `tools/export_bars.py` 里的映射函数** —— 否则映射本身写反时，测试会跟着一起错，交叉校验彻底失效。这是这条测试存在的意义所在。

**为什么必须做这一步**：`dump_bin` 的日历错位是这类转换的静默故障之王。若某标的缺失交易日没有占位、或特征数组与日历长度不对齐，Qlib 读出来完全正常、不报任何错，但每根 K 线整体偏移了若干天。后果是：因子用了"未来"数据、回测收益虚高、结论全错且毫无征兆。只测"能 import、能调用"抓不到它，必须比对原始数值。

因此 `tests/conftest.py` 会同时持有 duckdb 只读连接与 qlib provider，供交叉校验使用。

### 5.3 缺数据时的行为

`qlib_data/` 不存在 → 全部测试 **skip**，消息明确写出重建命令 `python tools/export_bars.py && python tools/build_qlib_data.py`。

不静默通过（假装成功），也不自动触发导数据（测试不该做重活）。

### 5.4 研究脚本

`research/alpha158_lgbm_backtest.py`：真实日期区间跑 Alpha158 + LightGBM + TopkDropout(topk=50, n_drop=3)。

默认区间 train 2018-01-01~2022-12-31 / valid 2023-01-01~2023-12-31 / test 2024-01-01~2025-12-31。**2026 年的数据刻意不进入任何区间**，留作日后检验结论是否腐化的样本外参照 —— 数据可用到今天 10-14，把 2026 花在训练里没有意义。

输出到 `artifacts/`：

- 因子矩阵 parquet
- 模型预测与 IC/ICIR 汇总
- 回测净值曲线数据 + 累计收益、超额收益、最大回撤、换手率
- 一份人类可读的 markdown 结论

## 6. 错误处理边界

三个文件夹三种策略，不混淆：

- **`tools/`** 唯一容忍脏数据。丢行、代码无法解析、价格 ≤ 0、重复 (标的,日期) 全部计数并打印报告，非零退出仅当可用行远低于预期（< 360,089 的 90%）。
- **`tests/`** 一律硬失败，不吞异常。任何被 except 掉的错误都是设计缺陷。
- **`research/`** 假设数据已经过 tests 验证，不做二次防御，只做参数校验。

## 7. 新增依赖

`pytest`（装入 `.venv`，无冲突）。当前环境仅有 pyqlib 及其依赖树。

## 8. 交付验收

1. `python tools/export_bars.py && python tools/build_qlib_data.py` 成功建库并打印数据体检报告
2. `pytest tests/` 六层全绿（第 1 层含与 duckdb 的数值交叉校验）
3. `python research/alpha158_lgbm_backtest.py` 产出 `artifacts/` 下一套完整回测结论
4. 确认未污染主系统数据目录。注意：**不能用 mtime / size 判断** —— 主系统在持续写入 `market.duckdb`，这些指标本来就一直变。改为两条可判定的检查：
   - 导出前后对 `量化分析/data` 做**文件名清单**快照，断言无新增文件（尤其不得出现 `.wal` / `.tmp` / 备份文件）
   - `export_bars.py` 里 duckdb 连接参数固定为 `read_only=True`，并在 `test_01` 中加一条断言：以写模式打开该库必须抛异常
5. spec 与实现一同提交到本地 `main` 分支

预计 10~20 分钟，大头是 360k 行导出 CSV 与 dump_bin 建库。
