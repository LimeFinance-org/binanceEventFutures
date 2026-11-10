# 币安事件合约 AI 自动训练与预测系统

研究 BTC / ETH 未来 10 分钟、30 分钟涨跌的四项预测任务。系统自动完成数据采集、特征研究、模型搜索、
样本外验证、概率校准、信号阈值优化、回测与风控，并提供每分钟实时预测服务和中文控制台。

**约定**（来自需求方）：入场价 = 决策分钟的收盘指数价；赔率固定 80%（猜对 100 → 180，猜错本金归零，平局退本金），
盈亏平衡胜率 = 1/1.8 = **55.56%**。70% 是研究目标，不是保证。

> 事件合约没有官方 API，系统只给出开仓提示，下单需要在币安 App 中手动完成。
> 所有收益数字都是固定赔率模拟，不是真实订单结算收益。

文档：[架构设计](docs/架构设计.md) · [提示词评审与修改建议](docs/提示词优化建议.md) · [优化后的提示词](docs/优化后的提示词.md)

---

## 一、在 GPU 服务器上训练（主流程，纯命令行）

```bash
git clone <本仓库地址> becai && cd becai
bash deploy/install.sh          # 建虚拟环境、装依赖、装 CUDA 12.8 版 PyTorch 并做 GPU 自检
```

```bash
make test                       # 自检：全部自动测试（离线，几分钟）
```

```bash
make data                       # 下载历史数据（2021 年至今，约 3~4 GB，可重复执行、只补缺失）
```

```bash
make train                      # 全自动训练。默认搜索预算 24 小时，可在 config/local.yaml 调整
```

训练是可中断的。被打断后用同一个运行 ID 继续，已完成的阶段和试验不会重跑：

```bash
make train RUN=run_20261110_120000
```

训练结束后看结果：

- `runs/<运行ID>/report.md` —— 中文报告：四项任务在独立测试集上的胜率、置信区间、是否达到 70%、组合回测、压力测试；
- `runs/<运行ID>/leaderboard.json` —— 各模型的样本外排行榜；
- `registry/` —— 模型版本库，新版本优于线上版本时自动上线。

建议长时间训练放在 `tmux` / `screen` 里，或者用下面的控制台启动（任务以后台子进程运行，关掉网页不影响）。

### 配置

`deploy/install.sh` 会把 `config/gpu_server.example.yaml` 复制为 `config/local.yaml`，里面是针对
RTX PRO 6000 96GB / 22 核 / 110GB 的推荐值。常用项：

| 配置项 | 含义 |
|---|---|
| `data.start` | 历史数据起点 |
| `search.time_budget_hours` | 模型搜索总时间预算 |
| `search.parallel_workers` | GPU 上同时运行的试验数 |
| `search.families.<模型族>.trials` | 每个模型族的试验上限，0 = 禁用 |
| `validation.holdout_days` | 独立测试集天数 |
| `signal.*` | 目标胜率、最小覆盖率、最少信号数 |
| `payout.fixed` | 固定赔率（默认 0.80） |
| `risk.*` | 风控上限。**`max_open_positions`、`max_order_usdt` 请按 App 交易规则页的实际数值填写** |

全部配置项及说明见 [`config/default.yaml`](config/default.yaml)。

## 二、每分钟实时预测

```bash
make live
```

每分钟收盘后约 1.5 秒输出四项预测，全部写入 `data/becai.sqlite`（包括未开仓的预测及跳过原因）。
需要服务器能访问 `fapi.binance.com`（指数价格行情）。实时推理只用 CPU，可以和 GPU 训练同时运行。

没有网络或想先验证流程时，可以用历史数据回放整个实时流程：

```bash
make replay
```

## 三、中文控制台（可选）

控制台运行在 GPU 服务器上，只是训练产出文件和预测库的展示层，不开也不影响训练和预测。

```bash
make web
```

默认只监听服务器本机的 8600 端口。在自己电脑上开一条 SSH 隧道，然后浏览器打开 `http://localhost:8600`：

```bash
ssh -L 8600:127.0.0.1:8600 用户名@GPU服务器地址
```

页面：总览 · 实时预测 · 历史信号 · 风险与订单 · 数据管理 · 训练任务（含 GPU 监控、断点恢复）·
模型排行榜 · 特征与周期 · 样本外回测 · 模型版本（上线/回退）· 设置（赔率、风控上限）。

如果要直接对外开放端口，必须在配置里设置 `web.auth_token`。

长期运行可用 systemd 或 Docker，见 [`deploy/systemd/README.md`](deploy/systemd/README.md) 和 [`deploy/docker-compose.yml`](deploy/docker-compose.yml)。
调度器（`python -m becai.cli scheduler`）负责每日数据更新、每 7 天自动再训练、训练异常中断后自动恢复。

## 四、训练流水线做了什么

| 阶段 | 内容 |
|---|---|
| prepare | 构建 1 分钟面板和约 350 个特征；数据质量检查；**特征泄漏检测**（不通过则中止）；分钟标签近似误差评估 |
| screen | 特征研究：按数据源 / 时间周期 / 特征族评估样本外重要性 |
| search | Optuna 搜索：逻辑回归、LightGBM、XGBoost、MLP、GRU、TCN、Transformer、多尺度融合；特征子集和周期也是搜索维度 |
| finalize | 候选模型做完整 walk-forward，生成样本外排行榜 |
| select | 集成；"复杂模型必须显著优于简单模型"检验；概率校准；信号阈值与市场状态过滤；过拟合概率 PBO |
| ablation | 消融实验：逐个去掉数据源、限制周期 |
| holdout | 在**独立测试集**上一次性评估（有访问台账，默认不允许重复评估）；组合回测；压力测试 |
| register | 登记模型版本；优于线上版本才自动上线 |

所有命令：

```bash
.venv/bin/python -m becai.cli --help
```

## 五、目录结构

```
becai/
  data/         历史归档下载、实时行情、面板组装
  features/     特征生成、泄漏检测
  labels.py     事件合约标签、分钟标签近似误差
  validation/   Purged walk-forward、统计检验、测试集台账
  models/       模型库（表格模型、神经网络）与搜索空间
  search/       特征研究、Optuna 搜索、训练流水线、实验记录
  calibration.py  概率校准
  signals/      信号策略优化
  risk/         全局风控
  backtest/     逐分钟回测、压力测试
  live/         实时预测服务、预测记录库
  registry.py   模型版本库、统一预测器
  jobs.py       任务管理、调度器
  web/          中文控制台
config/         默认配置、GPU 服务器示例配置
deploy/         安装脚本、systemd、Docker
docs/           架构设计、提示词评审
tests/          自动测试
```
