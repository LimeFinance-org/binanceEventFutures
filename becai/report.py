"""生成中文 Markdown 训练报告（如实报告，未达标也照写）。"""
from __future__ import annotations

from pathlib import Path

from .utils import load_json


def _pct(x, d: int = 2) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def _ci(ci) -> str:
    return "—" if not ci or ci[0] is None else f"[{100 * ci[0]:.2f}%, {100 * ci[1]:.2f}%]"


def write_report(run_dir: str | Path) -> str:
    d = Path(run_dir)
    rep, prep = load_json(d / "holdout_report.json"), load_json(d / "prepare.json", {})
    L = [f"# 训练运行报告 {rep['run_id']}", "",
         f"- 训练数据区间：{rep['train_range'][0]} ~ {rep['train_range'][1]}",
         f"- 独立测试集：{rep['window'][0]} ~ {rep['window'][1]}（第 {rep['ledger'].get('sequence', 1)} 次评估"
         + ("，**强制重复评估，结果存在选择偏差**" if rep["ledger"].get("forced") else "") + "）",
         f"- 泄漏检测：{'通过' if prep.get('leakage', {}).get('passed') else '未通过'}", "",
         "> 说明：下表全部来自模型从未见过的独立测试集。赔率按固定值模拟，不是真实订单结算收益。", "",
         "## 一、四项任务测试集结果", "",
         "| 任务 | 模型 | 全部分钟方向准确率 | 信号数 | 覆盖率 | 信号胜率 | 95%置信区间(区块自助) | 去重叠胜率 | 每单位期望 | 70%目标 | 策略状态 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    status_cn = {"target_met": "验证期达标", "positive_ev": "验证期正期望", "no_edge": "未发现优势（不开仓）"}
    for tn, r in rep["tasks"].items():
        s = r["signals"]
        ev = r.get("expected_value_per_unit", {})
        L.append(f"| {tn} | {r['model']['family']} | {_pct(r.get('direction_all', {}).get('accuracy'))} | {s.get('n', 0)} | "
                 f"{_pct(s.get('coverage'), 3)} | {_pct(s.get('win_rate'))} | {_ci(s.get('block_bootstrap_ci'))} | "
                 f"{_pct((s.get('non_overlap') or {}).get('win_rate'))} (n={(s.get('non_overlap') or {}).get('n', 0)}) | "
                 f"{'—' if not ev else format(ev['mean'], '+.4f')} | {r.get('target_assessment', '无信号')} | {status_cn.get(r['policy']['status'], r['policy']['status'])} |")
    be = next(iter(rep["tasks"].values()))["breakeven_win_rate"]
    L += ["", f"盈亏平衡胜率（赔率 {next(iter(rep['tasks'].values()))['payout']:.0%}）：**{_pct(be)}**。", "",
          "## 二、与简单基线、抛硬币的比较（测试集对数损失，越低越好）", "",
          "| 任务 | 选中模型 | 简单基线(逻辑回归) | 抛硬币 |", "|---|---|---|---|"]
    for tn, r in rep["tasks"].items():
        L.append(f"| {tn} | {r.get('direction_all', {}).get('logloss', float('nan')):.5f} | "
                 f"{(r.get('simple_baseline') or {}).get('logloss', float('nan')):.5f} | {r['coin_flip_logloss']:.5f} |")
    pf = rep["portfolio"]
    m = pf["main"]
    L += ["", "## 三、组合回测（经过全局风控，固定赔率模拟）", "", f"> {pf['disclaimer']}", "",
          f"- 信号 {m['n_signals']} 条，风控通过 {m['n_trades']} 笔，拒绝 {m['n_rejected']} 条",
          f"- 胜率 {_pct(m['win_rate'])}，总盈亏 {m['total_pnl']:+.2f} USDT，最大回撤 {m['max_drawdown']:.2f} USDT，"
          f"盈利天数占比 {_pct(m['positive_days_ratio'])}", "",
          "| 赔率情景 | 交易数 | 总盈亏 | 最大回撤 |", "|---|---|---|---|"]
    for k, s in pf["scenarios"].items():
        L.append(f"| {float(k):.0%} | {s['n_trades']} | {s['total_pnl']:+.2f} | {s['max_drawdown']:.2f} |")
    if isinstance(pf.get("stress"), dict) and "note" not in pf["stress"]:
        L += ["", "压力测试（按天区块重采样 + 胜率折减）：", "", "| 胜率折减 | 期末盈亏P5 | 中位数 | P95 | 亏损概率 | 最大回撤P95 |", "|---|---|---|---|---|---|"]
        for s in pf["stress"].values():
            L.append(f"| {s['win_rate_haircut']:.0%} | {s['pnl_p05']:+.1f} | {s['pnl_p50']:+.1f} | {s['pnl_p95']:+.1f} | {_pct(s['prob_loss'])} | {s['max_drawdown_p95']:.1f} |")
    L += ["", "## 四、分任务稳定性", ""]
    for tn, r in rep["tasks"].items():
        stab = r.get("stability")
        if not stab:
            L += [f"### {tn}", "", "无有效信号。", ""]
            continue
        L += [f"### {tn}", ""]
        for dim, cn in (("vol", "波动率"), ("trend", "趋势"), ("session", "时段"), ("month", "月份")):
            L.append(f"- {cn}：" + "；".join(f"{x['value']} {_pct(x['win_rate'], 1)} (n={x['n']})" for x in stab.get(dim, [])))
        if r.get("entry_delay_1m"):
            L.append(f"- 入场延迟 1 分钟时胜率：{_pct(r['entry_delay_1m']['win_rate'])} (n={r['entry_delay_1m']['n']})")
        e = r.get("edge_vs_breakeven", {})
        if e.get("z") is not None:
            L.append(f"- 相对盈亏平衡的显著性：z={e['z']:.2f}（考虑 {e['n_trials_considered']} 次搜索试验后需 z>{e['z_required']:.2f}），"
                     f"{'显著' if e['significant'] else '不显著'}")
        L.append("")
    la = prep.get("label_approximation", {})
    if la:
        L += ["## 五、分钟标签对秒级结算的近似误差", ""]
        for sym, a in la.items():
            for h, x in a["horizons"].items():
                L.append(f"- {sym} {h}分钟：平局率 {_pct(x['tie_rate'], 3)}，脆弱标签占比 {_pct(x['fragile_rate'])}，"
                         f"入场延迟1分钟标签一致率 {_pct(x['agree_delay_1m'])}，指数/现货方向一致率 {_pct(x.get('agree_index_vs_spot'))}")
            for h, x in (a.get("seconds") or {}).items():
                L.append(f"  - {h}分钟（现货1秒代理）：" + "，".join(f"延迟{k.split('_')[-1]} 一致率 {_pct(v)}" for k, v in x.items() if k.startswith("agree")))
        L.append("")
    text = "\n".join(L)
    (d / "report.md").write_text(text, encoding="utf-8")
    return text
