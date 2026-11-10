/* 币安事件合约 AI 控制台前端（原生 JS，无外部依赖） */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 2) => (v == null || isNaN(v) ? "—" : (100 * v).toFixed(d) + "%");
const num = (v, d = 2) => (v == null || isNaN(v) ? "—" : Number(v).toFixed(d));
const sgn = (v, d = 2) => (v == null || isNaN(v) ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(d));
const tm = (ms) => (ms ? new Date(ms).toISOString().replace("T", " ").slice(0, 19) : "—");
const tmShort = (ms) => (ms ? new Date(ms).toISOString().slice(11, 16) : "—");
const ci = (c) => (!c || c[0] == null ? "—" : `[${pct(c[0])}, ${pct(c[1])}]`);
const CN = { HIGHER: "HIGHER 看涨", LOWER: "LOWER 看跌", NO_TRADE: "NO_TRADE 不开仓", UP: "涨", DOWN: "跌", TIE: "平", UNKNOWN: "未知" };
const STATUS = { target_met: "验证期达标(≥70%)", positive_ev: "验证期正期望", no_edge: "未发现优势·不开仓" };

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  const tok = localStorage.getItem("becai_token");
  if (tok) headers["X-Auth-Token"] = tok;
  const r = await fetch(path, { ...opts, headers, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* 非 JSON */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
const act = async (fn) => { try { await fn(); } catch (e) { alert("操作失败：" + e.message); } };

function table(cols, rows, empty = "暂无数据") {
  if (!rows || !rows.length) return `<div class="wrap"><table><tr><td class="muted">${esc(empty)}</td></tr></table></div>`;
  const head = cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.t)}</th>`).join("");
  const body = rows.map((r) => "<tr>" + cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.f ? c.f(r) : esc(r[c.k])}</td>`).join("") + "</tr>").join("");
  return `<div class="wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
const card = (k, v, s = "") => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;

/* ---------- 画图 ---------- */
function setup(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = Number(cv.getAttribute("height"));
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  return { g, w, h, ink: css.getPropertyValue("--ink"), muted: css.getPropertyValue("--muted"), line: css.getPropertyValue("--line"),
           up: css.getPropertyValue("--up"), down: css.getPropertyValue("--down"), accent: css.getPropertyValue("--accent") };
}
function barChart(cv, items, unit = "") {            // items: [{name, value}] 横向条形图
  const c = setup(cv), g = c.g;
  if (!items.length) { g.fillStyle = c.muted; g.fillText("暂无数据", 12, 20); return; }
  const left = 150, right = 70, top = 8, rowH = Math.max(14, (c.h - 16) / items.length);
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  const zero = left + ((c.w - left - right) * (Math.min(0, ...items.map((i) => i.value)) < 0 ? 0.3 : 0));
  const scale = (c.w - right - zero) / max;
  g.font = "12px sans-serif"; g.textBaseline = "middle";
  items.forEach((it, i) => {
    const y = top + i * rowH + rowH / 2;
    g.fillStyle = c.muted; g.textAlign = "right"; g.fillText(String(it.name).slice(0, 22), left - 8, y);
    const wv = it.value * scale;
    g.fillStyle = it.value >= 0 ? c.up : c.down;
    g.fillRect(wv >= 0 ? zero : zero + wv, y - rowH * 0.3, Math.abs(wv), rowH * 0.6);
    g.fillStyle = c.ink; g.textAlign = "left"; g.fillText(num(it.value, 2) + unit, Math.max(zero, zero + wv) + 6, y);
  });
  g.strokeStyle = c.line; g.beginPath(); g.moveTo(zero, top); g.lineTo(zero, c.h - 8); g.stroke();
}
function lineChart(cv, series, opt = {}) {           // series: [{name, color, pts:[[x,y]], axis:'l'|'r'}]
  const c = setup(cv), g = c.g, L = 52, R = opt.right ? 52 : 14, T = 22, B = 26;
  const all = series.flatMap((s) => s.pts);
  if (!all.length) { g.fillStyle = c.muted; g.fillText("暂无数据", 12, 20); return; }
  const xs = all.map((p) => p[0]), x0 = Math.min(...xs), x1 = Math.max(...xs);
  const rng = (ax) => { const ys = series.filter((s) => (s.axis || "l") === ax).flatMap((s) => s.pts.map((p) => p[1])); if (!ys.length) return [0, 1];
    let a = Math.min(...ys, ...(opt.include || [])), b = Math.max(...ys, ...(opt.include || [])); if (a === b) { a -= 1; b += 1; } return [a, b]; };
  const yl = rng("l"), yr = rng("r");
  const X = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (c.w - L - R);
  const Y = (y, ax) => { const r = ax === "r" ? yr : yl; return T + (1 - (y - r[0]) / (r[1] - r[0])) * (c.h - T - B); };
  g.font = "11px sans-serif"; g.strokeStyle = c.line; g.fillStyle = c.muted; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = T + (i / 4) * (c.h - T - B); g.beginPath(); g.moveTo(L, y); g.lineTo(c.w - R, y); g.stroke();
    g.textAlign = "right"; g.fillText((opt.yfmt || num)(yl[1] - (i / 4) * (yl[1] - yl[0])), L - 6, y + 4);
    if (opt.right) { g.textAlign = "left"; g.fillText((opt.rfmt || num)(yr[1] - (i / 4) * (yr[1] - yr[0])), c.w - R + 6, y + 4); }
  }
  g.textAlign = "center";
  for (let i = 0; i <= 4; i++) { const x = x0 + (i / 4) * (x1 - x0); g.fillText((opt.xfmt || num)(x), X(x), c.h - 8); }
  (opt.hlines || []).forEach((hl) => { g.strokeStyle = hl.color || c.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(L, Y(hl.y, "l")); g.lineTo(c.w - R, Y(hl.y, "l")); g.stroke(); g.setLineDash([]);
    g.fillStyle = hl.color || c.muted; g.textAlign = "left"; g.fillText(hl.label, L + 4, Y(hl.y, "l") - 4); });
  let lx = L;
  series.forEach((s) => {
    const col = s.color || c.accent; g.strokeStyle = col; g.lineWidth = 1.8; g.beginPath();
    s.pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1], s.axis || "l")) : g.moveTo(X(p[0]), Y(p[1], s.axis || "l")))); g.stroke();
    g.fillStyle = col; g.fillRect(lx, 6, 10, 3); g.fillStyle = c.muted; g.textAlign = "left"; g.fillText(s.name, lx + 14, 11); lx += g.measureText(s.name).width + 34;
  });
}

/* ---------- 预测卡片 ---------- */
function predCard(p) {
  const d = p.decision || "NO_TRADE";
  const u = (p.p_up || 0) * 100, dn = (p.p_down || 0) * 100, f = (p.p_flat || 0) * 100;
  const tag = p.executable ? `<span class="badge ${d}">${CN[d]} · 可执行</span>` : d !== "NO_TRADE" ? `<span class="badge warn">${CN[d]} · 仅研究</span>` : `<span class="badge">${CN.NO_TRADE}</span>`;
  const why = p.executable ? `建议本金 ${num(p.stake)} USDT，赔率 ${pct(p.payout, 0)}，期望 ${sgn(p.ev, 3)}/USDT` : esc([p.skip_reason, p.risk_reasons].filter(Boolean).join("：")) || "—";
  return `<div class="pred"><div class="t"><span>${esc(p.symbol.replace("USDT", ""))} · ${p.horizon} 分钟</span>${tag}</div>
    <div class="bar"><i class="u" style="width:${u}%"></i><i class="f" style="width:${f}%"></i><i class="d" style="width:${dn}%"></i></div>
    <div class="row"><span class="up">上涨 ${p.p_up == null ? "—" : u.toFixed(1) + "%"}</span><span>持平 ${p.p_flat == null ? "—" : f.toFixed(2) + "%"}</span><span class="down">下跌 ${p.p_down == null ? "—" : dn.toFixed(1) + "%"}</span></div>
    <div class="row"><span>决策时刻 ${tmShort(p.ts)} UTC</span><span>入场指数价 ${num(p.entry_price)}</span></div>
    <div class="why">${why}</div></div>`;
}
const PRED_COLS = [
  { t: "时间(UTC)", f: (r) => tm(r.ts) }, { t: "任务", f: (r) => `${esc(r.symbol.replace("USDT", ""))} ${r.horizon}m` },
  { t: "上涨", num: 1, f: (r) => pct(r.p_up, 1) }, { t: "下跌", num: 1, f: (r) => pct(r.p_down, 1) }, { t: "持平", num: 1, f: (r) => pct(r.p_flat, 2) },
  { t: "决策", f: (r) => `<span class="badge ${r.executable ? r.decision : r.decision !== "NO_TRADE" ? "warn" : ""}">${esc(r.decision)}${r.executable ? "" : r.decision !== "NO_TRADE" ? "·研究" : ""}</span>` },
  { t: "跳过原因 / 风控", f: (r) => esc([r.skip_reason, r.risk_reasons].filter(Boolean).join("：")) },
  { t: "入场价", num: 1, f: (r) => num(r.entry_price) }, { t: "结算价", num: 1, f: (r) => num(r.settle_price) },
  { t: "实际", f: (r) => (r.actual ? CN[r.actual] || r.actual : "未到期") },
  { t: "方向对错", f: (r) => (r.correct == null ? "—" : r.correct ? '<span class="up">✓</span>' : '<span class="down">✗</span>') },
  { t: "模拟盈亏", num: 1, f: (r) => (r.pnl == null ? "—" : `<span class="${r.pnl >= 0 ? "up" : "down"}">${sgn(r.pnl)}</span>`) },
  { t: "延迟ms", num: 1, f: (r) => num(r.latency_ms, 0) }, { t: "版本", k: "version" },
];

/* ---------- 各页面 ---------- */
const state = { tab: "overview", page: 0, run: null, task: null, runs: [] };

async function loadHeader() {
  const o = await api("/api/overview");
  $("#hstat").innerHTML = [
    `<span class="badge ${o.live_alive ? "ok" : "bad"}">实时服务 ${o.live_alive ? "运行中" : "未运行"}</span>`,
    `<span class="badge ${o.champion ? "ok" : "warn"}">线上模型 ${esc(o.champion || "无")}</span>`,
    `<span class="badge">赔率 ${pct(o.payout.fixed, 0)} · 盈亏平衡 ${pct(o.payout.breakeven)}</span>`,
    o.running_jobs.length ? `<span class="badge warn">训练中 ${o.running_jobs.length}</span>` : "",
  ].join("");
  return o;
}
async function loadOverview() {
  const o = await loadHeader();
  const hb = o.live ? o.live.value : null;
  $("#ov-cards").innerHTML = [
    card("线上模型版本", esc(o.champion || "无"), o.champion_meta ? "测试窗口 " + esc((o.champion_meta.holdout_window || []).join(" ~ ")) : "请先完成训练"),
    card("实时服务", o.live_alive ? '<span class="up">运行中</span>' : '<span class="down">未运行</span>', hb ? `最近一轮 ${tm(hb.ts)}，耗时 ${num(hb.latency_ms, 0)}ms` : "python -m becai.cli live"),
    card("未到期合约", hb && hb.risk ? hb.risk.open_positions : 0, hb && hb.risk ? `本金合计 ${num(hb.risk.open_stake)} USDT` : ""),
    card("今日风险占用", hb && hb.risk ? `${num(hb.risk.daily_loss_exposure)} / ${num(hb.risk.daily_loss_limit, 0)}` : "—", "已实现亏损 + 未到期本金 / 每日上限"),
    card("研究目标胜率", pct(o.target_win_rate, 0), `赔率 ${pct(o.payout.fixed, 0)} 下盈亏平衡 ${pct(o.payout.breakeven)}`),
    card("训练任务", o.running_jobs.length ? "运行中" : "空闲", o.running_jobs.map((j) => "#" + j.id + " " + esc(j.run_id || "")).join(" ")),
  ].join("");
  $("#ov-preds").innerHTML = o.latest.length ? o.latest.map(predCard).join("") : '<div class="muted">暂无预测。启动实时服务后每分钟生成四项预测。</div>';
  $("#ov-today").innerHTML = statsTable(o.today);
  if (hb && (Object.keys(hb.errors || {}).length || Object.keys(hb.stale || {}).length))
    $("#ov-today").insertAdjacentHTML("beforebegin", `<div class="banner">行情数据异常：${esc(JSON.stringify({ ...hb.errors, ...hb.stale }))}</div>`);
}
function statsTable(rows) {
  return table([
    { t: "任务", f: (r) => `${esc(r.symbol.replace("USDT", ""))} ${r.horizon}m` }, { t: "预测数", num: 1, k: "n_pred" },
    { t: "方向准确率(全部预测)", num: 1, f: (r) => (r.n_settled ? pct(r.n_correct / r.n_settled) + ` (n=${r.n_settled})` : "—") },
    { t: "开仓信号数", num: 1, k: "n_signal" },
    { t: "信号胜率", num: 1, f: (r) => (r.n_signal_settled ? pct(r.n_signal_win / r.n_signal_settled) + ` (n=${r.n_signal_settled})` : "—") },
    { t: "可执行信号", num: 1, k: "n_exec" },
    { t: "可执行胜/负", num: 1, f: (r) => `${r.n_exec_win}/${r.n_exec_loss}` },
    { t: "模拟盈亏(USDT)", num: 1, f: (r) => `<span class="${r.sim_pnl >= 0 ? "up" : "down"}">${sgn(r.sim_pnl)}</span>` },
  ], rows, "暂无统计");
}
async function loadLive() {
  const d = await api("/api/live/latest");
  $("#live-cards").innerHTML = d.predictions.length ? d.predictions.map(predCard).join("") : '<div class="muted">暂无预测</div>';
  $("#live-table").innerHTML = table(PRED_COLS, await api("/api/live/history?limit=80"));
}
async function loadHistory() {
  const q = new URLSearchParams({ limit: 100, offset: state.page * 100 });
  if ($("#h-symbol").value) q.set("symbol", $("#h-symbol").value);
  if ($("#h-horizon").value) q.set("horizon", $("#h-horizon").value);
  if ($("#h-signals").checked) q.set("only_signals", "true");
  const [rows, st] = await Promise.all([api("/api/live/history?" + q), api("/api/live/stats?days=" + $("#h-days").value)]);
  $("#h-stats").innerHTML = statsTable(st.by_task) + `<p class="note">${esc(st.note)}</p>`;
  $("#h-skips").innerHTML = table([{ t: "原因", k: "reason" }, { t: "次数", num: 1, k: "n" }], st.skip_reasons);
  const dp = [...st.daily_pnl].reverse(); let cum = 0;
  lineChart($("#h-pnl"), [{ name: "累计模拟盈亏 (USDT)", pts: dp.map((d) => [d.day, (cum += d.pnl)]) }], { xfmt: (d) => new Date(d * 864e5).toISOString().slice(5, 10), include: [0] });
  $("#h-table").innerHTML = table(PRED_COLS, rows);
  $("#h-page").textContent = `第 ${state.page + 1} 页`;
}
async function loadRisk() {
  const d = await api("/api/risk"), s = d.state || {}, c = d.config;
  $("#risk-cards").innerHTML = [
    card("未到期合约数", `${s.open_positions ?? 0} / ${c.max_open_positions}`, "平台持仓数上限以 App 规则页为准"),
    card("未到期本金", `${num(s.open_stake ?? 0)} / ${num(c.max_total_open_usdt, 0)}`, "USDT"),
    card("今日风险占用", `${num(s.daily_loss_exposure ?? 0)} / ${num(c.daily_loss_limit_usdt, 0)}`, "已实现亏损 + 未到期本金"),
    card("今日已实现盈亏", sgn(s.daily_realized ?? 0), "按赔率模拟"),
    card("连续亏损", `${s.consecutive_losses ?? 0} / ${c.max_consecutive_losses}`, s.cooldown_until > Date.now() ? "冷静期至 " + tm(s.cooldown_until) : "未处于冷静期"),
    card("最大回撤", num(s.max_drawdown ?? 0), "服务启动以来 (USDT)"),
  ].join("");
  $("#risk-open").innerHTML = table([
    { t: "币种", k: "symbol" }, { t: "期限", f: (r) => r.horizon + "m" }, { t: "方向", f: (r) => (r.direction > 0 ? "HIGHER" : "LOWER") },
    { t: "本金", num: 1, f: (r) => num(r.stake) }, { t: "赔率", num: 1, f: (r) => pct(r.payout, 0) }, { t: "开仓", f: (r) => tm(r.open_ts) },
    { t: "到期", f: (r) => tm(r.expiry_ts) }, { t: "入场价", num: 1, f: (r) => num(r.entry_price) },
  ], s.positions || [], "当前没有未到期合约");
  const os = d.order_stats;
  $("#risk-orders").innerHTML = `<p class="note">真实订单 ${os.n || 0} 笔，胜 ${os.win || 0} 负 ${os.loss || 0}，盈亏 ${sgn(os.pnl || 0)} USDT（其中 ${os.n_user_confirmed || 0} 笔为用户确认的实际结果）。${esc(d.note)}</p>` + table([
    { t: "ID", k: "id" }, { t: "时间", f: (r) => tm(r.ts) }, { t: "币种", k: "symbol" }, { t: "期限", f: (r) => r.horizon + "m" }, { t: "方向", k: "direction" },
    { t: "本金", num: 1, f: (r) => num(r.stake) }, { t: "赔率", num: 1, f: (r) => pct(r.payout, 0) }, { t: "入场价", num: 1, f: (r) => num(r.entry_price) },
    { t: "结果", f: (r) => (r.result ? `${esc(r.result)} <span class="muted">(${r.result_source === "user" ? "实际" : "近似"})</span>` : "未到期") },
    { t: "盈亏", num: 1, f: (r) => (r.pnl == null ? "—" : sgn(r.pnl)) },
    { t: "更正实际结果", f: (r) => ["WIN", "LOSS", "TIE"].map((x) => `<button class="small" data-order="${r.id}" data-res="${x}">${x}</button>`).join(" ") },
  ], d.orders, "暂无登记订单");
  $("#risk-config").innerHTML = table([{ t: "参数", k: "k" }, { t: "当前值", k: "v" }], Object.entries(c).map(([k, v]) => ({ k, v })));
}
async function loadData() {
  $("#data-table").innerHTML = table([
    { t: "数据集", k: "dataset" }, { t: "币种", k: "symbol" }, { t: "文件数", num: 1, k: "files" }, { t: "行数", num: 1, f: (r) => (r.rows || 0).toLocaleString() },
    { t: "起始", k: "start" }, { t: "截至", k: "end" }, { t: "大小MB", num: 1, k: "size_mb" }, { t: "未完整月份", f: (r) => esc((r.incomplete_months || []).join(", ")) },
  ], await api("/api/data/status"), "尚未下载数据");
}
async function loadTrain() {
  const [jobs, gpu, runs] = await Promise.all([api("/api/jobs"), api("/api/gpu"), api("/api/runs")]);
  $("#gpu-cards").innerHTML = (gpu.gpus.length ? gpu.gpus.map((g) => card(`GPU ${g.index} ${esc(g.name)}`, num(g.util, 0) + "%", `显存 ${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(0)} GB · ${num(g.temp, 0)}°C · ${g.power_w == null ? "—" : num(g.power_w, 0) + "W"}`)) : [card("GPU", "未检测到", "nvidia-smi 不可用")])
    .concat([card("CPU", num(gpu.cpu_percent, 0) + "%", `${gpu.cpu_count} 核 · 负载 ${gpu.load_avg.map((x) => x.toFixed(1)).join(" / ")}`), card("内存", `${gpu.ram_used_gb.toFixed(1)} / ${gpu.ram_total_gb.toFixed(0)} GB`, "")]).join("");
  const run = runs[0];
  if (run) {
    const p = run.progress || {}, done = Object.keys(run.stages || {}).length;
    $("#t-progress").innerHTML = `<div class="card"><div class="k">运行 ${esc(run.run_id)}</div><div class="v" style="font-size:16px">${esc(p.stage_cn || "—")} <span class="muted">${esc(p.detail || "")}</span></div>
      <div class="progress"><i style="width:${(100 * done) / 8}%"></i></div><div class="s">已完成 ${done}/8 个阶段 · 更新于 ${tm(p.updated)}</div></div>`;
    const ss = await api(`/api/runs/${encodeURIComponent(run.run_id)}/search`);
    $("#t-studies").innerHTML = table([
      { t: "搜索任务", k: "name" }, { t: "模型族", k: "family" }, { t: "试验", num: 1, f: (r) => `${r.done ?? 0}/${r.total ?? "?"}` },
      { t: "最佳技能值(×10⁴)", num: 1, f: (r) => num(r.best_skill_1e4, 2) }, { t: "状态", f: (r) => (r.finished ? "完成" : "进行中") },
    ], Object.entries(ss.studies || {}).map(([name, v]) => ({ name, ...v })), "尚未开始搜索");
  } else { $("#t-progress").innerHTML = '<div class="muted">尚无训练运行</div>'; $("#t-studies").innerHTML = ""; }
  $("#t-jobs").innerHTML = table([
    { t: "#", k: "id" }, { t: "类型", f: (r) => (r.type === "train" ? "训练" : "数据更新") }, { t: "状态", f: (r) => `<span class="badge ${r.status === "done" ? "ok" : r.status === "running" ? "warn" : r.status === "queued" ? "" : "bad"}">${esc(r.status_cn)}</span>` },
    { t: "运行ID", k: "run_id" }, { t: "开始", f: (r) => tm(r.started) }, { t: "结束", f: (r) => tm(r.finished) }, { t: "信息", f: (r) => esc((r.message || "").slice(0, 80)) },
    { t: "操作", f: (r) => `<button class="small" data-log="${r.id}">日志</button> ` + (r.status === "running" ? `<button class="small" data-stop="${r.id}">停止</button>` : "") + (["interrupted", "stopped", "failed"].includes(r.status) && r.type === "train" && r.run_id ? `<button class="small" data-resume="${r.id}">从断点恢复</button>` : "") },
  ], jobs, "暂无任务");
}
async function ensureRuns() {
  state.runs = await api("/api/runs");
  if (!state.run && state.runs.length) state.run = (state.runs.find((r) => r.has_report) || state.runs[0]).run_id;   // 默认选最近一个已出测试集报告的运行
  $$(".run-select").forEach((s) => { s.innerHTML = state.runs.map((r) => `<option ${r.run_id === state.run ? "selected" : ""}>${esc(r.run_id)}</option>`).join(""); });
  const cfg = await api("/api/config");
  const tasks = cfg.symbols.flatMap((s) => cfg.horizons.map((h) => `${s}_${h}m`));
  if (!state.task) state.task = tasks[0];
  $$(".task-select").forEach((s) => { s.innerHTML = tasks.map((t) => `<option ${t === state.task ? "selected" : ""}>${esc(t)}</option>`).join(""); });
  return !!state.run;
}
const runApi = (item, withTask) => api(`/api/runs/${encodeURIComponent(state.run)}/${item}` + (withTask ? `?task=${encodeURIComponent(state.task)}` : ""));
async function loadBoard() {
  if (!(await ensureRuns())) { $("#b-table").innerHTML = '<div class="muted">尚无训练运行</div>'; return; }
  const [lb, sel, trials] = await Promise.all([runApi("leaderboard"), runApi("selection", true), runApi("trials")]);
  $("#b-table").innerHTML = table([
    { t: "排名", f: (r) => r._i }, { t: "模型族", k: "family" }, { t: "复杂度", num: 1, k: "complexity" }, { t: "任务数", num: 1, k: "n_tasks" }, { t: "特征数", num: 1, k: "n_features" },
    { t: "样本外对数损失", num: 1, f: (r) => num(r.oof_logloss, 5) }, { t: "技能值×10⁴", num: 1, f: (r) => num(r.oof_skill_1e4, 2) },
    { t: "AUC", num: 1, f: (r) => num(r.oof_auc, 4) }, { t: "方向准确率", num: 1, f: (r) => pct(r.oof_accuracy) }, { t: "校准误差ECE", num: 1, f: (r) => num(r.oof_ece, 4) },
    { t: "优于抛硬币的折占比", num: 1, f: (r) => pct(r.folds_beating_coin, 0) }, { t: "各折技能波动", num: 1, f: (r) => num(r.fold_skill_std, 2) },
    { t: "训练秒", num: 1, f: (r) => num(r.fit_seconds, 0) }, { t: "ID", k: "id" },
  ], (lb[state.task] || []).map((r, i) => ({ ...r, _i: i + 1 })), "排行榜尚未生成（需完成 finalize 阶段）");
  if (sel && sel.chosen) {
    const p = sel.policy, v = p.validation || {};
    $("#b-select").innerHTML = `<div class="cards">${[
      card("选中模型", esc(sel.chosen.family), `成员 ${sel.chosen.members.length} 个 · 权重 ${sel.chosen.weights.map((w) => w.toFixed(2)).join("/")}`),
      card("概率校准", esc(sel.calibration.method), Object.entries(sel.calibration.scores).map(([k, x]) => `${k}:${num(x, 5)}`).join(" ")),
      card("信号策略", esc(STATUS[p.status] || p.status), p.threshold == null ? "不发出开仓信号" : `置信度阈值 ${num(p.threshold, 3)}` + (Object.keys(p.excluded || {}).length ? " · 过滤 " + esc(JSON.stringify(p.excluded)) : "")),
      card("验证期信号胜率", pct(v.win_rate), v.n ? `n=${v.n} · 置信下界 ${pct(v.lcb)} · 覆盖率 ${pct(v.coverage, 2)}` : "—"),
      card("回测过拟合概率 PBO", sel.pbo.pbo == null ? "—" : pct(sel.pbo.pbo, 0), sel.pbo.pbo == null ? esc(sel.pbo.note || "") : "越低越好；>50% 说明选择不可靠"),
    ].join("")}</div><h2>"复杂模型必须证明优于简单模型"检验</h2>` + table([
      { t: "较简单", k: "simple" }, { t: "较复杂", k: "complex" }, { t: "对数损失改善×10⁴", num: 1, f: (r) => sgn(r.improvement_1e4, 2) },
      { t: "置信下界×10⁴", num: 1, f: (r) => sgn(r.lower_bound_1e4, 2) }, { t: "结论", f: (r) => (r.adopt_complex ? '<span class="up">显著更优，采用复杂模型</span>' : "不显著，保留简单模型") },
    ], sel.simplicity_comparisons, "只有一个复杂度层级");
  } else $("#b-select").innerHTML = '<div class="muted">尚未完成模型选择</div>';
  $("#b-trials").innerHTML = table([
    { t: "时间", f: (r) => tm(r.created) }, { t: "搜索任务", k: "study" }, { t: "#", num: 1, k: "trial" }, { t: "状态", f: (r) => esc(String(r.status).slice(0, 40)) },
    { t: "技能值×10⁴", num: 1, f: (r) => (r.value == null ? "—" : num((Math.LN2 - r.value) * 1e4, 2)) }, { t: "特征数", num: 1, f: (r) => r.params?.n_features ?? "—" },
    { t: "用时秒", num: 1, f: (r) => num(r.duration, 0) }, { t: "参数", f: (r) => `<span class="muted">${esc(JSON.stringify(r.params?.params || {}).slice(0, 140))}</span>` },
  ], trials.filter((t) => t.study.includes(state.task) || t.study.includes("shared") || t.study.endsWith(state.task.split("_")[0])).slice(0, 60), "暂无试验记录");
}
async function loadFeatures() {
  if (!(await ensureRuns())) return;
  const [sc, ab] = await Promise.all([runApi("screen", true), runApi("ablation", true)]);
  const bars = (rows) => (rows || []).map((r) => ({ name: r.name, value: r.delta_logloss_1e4 }));
  barChart($("#f-source"), bars(sc.by_source)); barChart($("#f-scale"), bars(sc.by_scale).map((b) => ({ ...b, name: b.name === "0" ? "即时值(无窗口)" : b.name + " 分钟" }))); barChart($("#f-family"), bars(sc.by_family));
  $("#f-groups").innerHTML = table([
    { t: "分组", k: "group" }, { t: "增益占比", num: 1, f: (r) => pct(r.gain, 2) }, { t: "跨折稳定性", num: 1, f: (r) => pct(r.stability, 0) }, { t: "特征数", num: 1, k: "n_features" },
  ], (sc.groups || []).slice(0, 30), "特征研究尚未完成");
  $("#f-ablation").innerHTML = (ab.note ? `<p class="note">${esc(ab.note)}（基准模型 ${esc(ab.base_model || "")}，技能值 ${num(ab.base_skill_1e4, 2)}）</p>` : "") + table([
    { t: "实验", k: "type" }, { t: "对象", k: "name" }, { t: "剩余特征数", num: 1, k: "n_features" },
    { t: "对数损失变化×10⁴", num: 1, f: (r) => `<span class="${r.delta_logloss_1e4 > 0 ? "up" : "down"}">${sgn(r.delta_logloss_1e4, 2)}</span>` },
    { t: "解读", f: (r) => (r.delta_logloss_1e4 > 0.5 ? "有贡献" : r.delta_logloss_1e4 < -0.5 ? "去掉反而更好" : "影响不明显") },
  ], ab.rows, "消融实验尚未完成");
}
async function loadBacktest() {
  if (!(await ensureRuns())) return;
  const [rep, sel, prep, ledger] = await Promise.all([runApi("holdout"), runApi("selection", true), runApi("prepare"), api("/api/ledger")]);
  const diag = sel.policy_diagnostics;
  if (diag && diag.chosen_curve) lineChart($("#bt-curve"), [
    { name: "胜率", pts: diag.chosen_curve.map((c) => [c.threshold, c.win_rate]) }, { name: "置信下界", color: "#888", pts: diag.chosen_curve.map((c) => [c.threshold, c.lcb]) },
    { name: "覆盖率(右轴)", color: "#4a7bd0", axis: "r", pts: diag.chosen_curve.map((c) => [c.threshold, c.coverage]) },
  ], { right: true, yfmt: (v) => pct(v, 0), rfmt: (v) => pct(v, 1), xfmt: (v) => num(v, 2), hlines: [{ y: diag.target, label: "目标 " + pct(diag.target, 0) }, { y: diag.breakeven, label: "盈亏平衡 " + pct(diag.breakeven, 1) }] });
  $("#bt-ledger").innerHTML = table([{ t: "测试窗口", f: (r) => tm(r.window[0]) + " ~ " + tm(r.window[1]) }, { t: "运行", k: "run_id" }, { t: "第几次评估", num: 1, k: "sequence" }, { t: "强制", f: (r) => (r.forced ? '<span class="down">是</span>' : "否") }, { t: "时间", f: (r) => tm(r.time) }], ledger, "尚无评估记录");
  const la = prep.label_approximation || {};
  $("#bt-label").innerHTML = table([
    { t: "币种", k: "sym" }, { t: "期限", k: "h" }, { t: "平局率", num: 1, f: (r) => pct(r.tie_rate, 3) }, { t: "脆弱标签占比", num: 1, f: (r) => pct(r.fragile_rate) },
    { t: "延迟1分钟一致率", num: 1, f: (r) => pct(r.agree_delay_1m) }, { t: "指数/现货一致率", num: 1, f: (r) => pct(r.agree_index_vs_spot) },
    { t: "延迟1秒", num: 1, f: (r) => pct(r.sec?.agree_delay_1s) }, { t: "延迟5秒", num: 1, f: (r) => pct(r.sec?.agree_delay_5s) }, { t: "延迟10秒", num: 1, f: (r) => pct(r.sec?.agree_delay_10s) }, { t: "延迟30秒", num: 1, f: (r) => pct(r.sec?.agree_delay_30s) },
  ], Object.entries(la).flatMap(([sym, a]) => Object.entries(a.horizons).map(([h, x]) => ({ sym, h: h + "m", ...x, sec: (a.seconds || {})[h] }))), "尚未评估");
  if (!rep || !rep.tasks) { $("#bt-head").innerHTML = '<div class="banner">该运行尚未进行独立测试集评估。</div>'; ["#bt-cards", "#bt-stab", "#bt-port", "#bt-stress"].forEach((s) => ($(s).innerHTML = "")); ["#bt-rel", "#bt-equity"].forEach((s) => setup($(s))); if (!diag) setup($("#bt-curve")); return; }
  const r = rep.tasks[state.task], s = r.signals, ev = r.expected_value_per_unit;
  $("#bt-head").innerHTML = `<div class="banner">独立测试集 ${esc(rep.window.join(" ~ "))}（训练区间 ${esc(rep.train_range.join(" ~ "))}）。第 ${rep.ledger.sequence} 次评估${rep.ledger.forced ? "，<b>强制重复评估，存在选择偏差</b>" : ""}。收益为固定赔率 ${pct(r.payout, 0)} 模拟，不是真实订单结算收益。</div>`;
  $("#bt-cards").innerHTML = [
    card("70% 目标评估", esc(r.target_assessment || "无信号"), esc(STATUS[r.policy.status] || "")),
    card("方向准确率(全部分钟)", pct(r.direction_all?.accuracy), `n=${r.direction_all?.n ?? 0} · AUC ${num(r.direction_all?.auc, 4)}`),
    card("有效信号胜率", pct(s.win_rate), `n=${s.n} · 覆盖率 ${pct(s.coverage, 2)} · 每天 ${num(s.per_day, 1)} 条`),
    card("95% 置信区间", ci(s.block_bootstrap_ci), `区块自助法 · 有效样本 ${num(s.n_effective, 0)}`),
    card("去重叠胜率", pct(s.non_overlap?.win_rate), `n=${s.non_overlap?.n ?? 0} · ${ci(s.non_overlap?.ci)}`),
    card("每 1 USDT 期望", ev ? sgn(ev.mean, 4) : "—", ev ? `区间 [${sgn(ev.ci[0], 3)}, ${sgn(ev.ci[1], 3)}] · 盈亏平衡 ${pct(r.breakeven_win_rate)}` : ""),
    card("相对盈亏平衡显著性", r.edge_vs_breakeven?.z == null ? "—" : "z=" + num(r.edge_vs_breakeven.z, 2), r.edge_vs_breakeven?.z == null ? "" : `需 z>${num(r.edge_vs_breakeven.z_required, 2)}（${r.edge_vs_breakeven.n_trials_considered} 次试验校正）· ${r.edge_vs_breakeven.significant ? "显著" : "不显著"}`),
    card("入场延迟1分钟胜率", pct(r.entry_delay_1m?.win_rate), `n=${r.entry_delay_1m?.n ?? 0}`),
    card("对数损失", num(r.direction_all?.logloss, 5), `简单基线 ${num(r.simple_baseline?.logloss, 5)} · 抛硬币 ${num(r.coin_flip_logloss, 5)}`),
  ].join("");
  lineChart($("#bt-rel"), [{ name: "实际上涨频率", pts: (r.reliability || []).map((b) => [b.pred, b.actual]) }, { name: "理想校准", color: "#888", pts: [[0.3, 0.3], [0.7, 0.7]] }], { yfmt: (v) => pct(v, 0), xfmt: (v) => pct(v, 0) });
  const st = r.stability || {};
  $("#bt-stab").innerHTML = table([{ t: "维度", k: "dim" }, { t: "状态", k: "value" }, { t: "信号数", num: 1, k: "n" }, { t: "胜率", num: 1, f: (x) => pct(x.win_rate) }, { t: "95%区间", f: (x) => ci(x.ci) }],
    [["vol", "波动率"], ["trend", "趋势"], ["session", "时段"], ["weekend", "周末"], ["month", "月份"]].flatMap(([k, cn]) => (st[k] || []).map((x) => ({ dim: cn, ...x }))), "该任务在测试集上没有开仓信号");
  const pf = rep.portfolio, m = pf.main;
  $("#bt-port").innerHTML = `<div class="cards">${[
    card("信号 / 通过风控", `${m.n_signals} / ${m.n_trades}`, `拒绝 ${m.n_rejected} 条`), card("胜率", pct(m.win_rate), `胜 ${m.wins} 负 ${m.losses} 平 ${m.ties}`),
    card("模拟总盈亏", sgn(m.total_pnl), `投入本金 ${num(m.total_staked, 0)} USDT · 回报率 ${pct(m.roi_on_staked)}`), card("最大回撤", num(m.max_drawdown), "USDT"),
    card("盈利天数占比", pct(m.positive_days_ratio, 0), `共 ${m.n_days} 天 · 最差单日 ${sgn(m.worst_day)}`),
  ].join("")}</div><p class="note">${esc(pf.disclaimer)} 风控拒绝原因：${esc(JSON.stringify(m.rejected_reasons))}</p>`;
  lineChart($("#bt-equity"), [{ name: "组合累计模拟盈亏 (USDT)", pts: m.equity_curve }], { xfmt: (t) => new Date(t).toISOString().slice(5, 10), include: [0] });
  $("#bt-stress").innerHTML = table([{ t: "赔率情景", f: (x) => pct(x.payout, 0) }, { t: "交易数", num: 1, k: "n_trades" }, { t: "总盈亏", num: 1, f: (x) => sgn(x.total_pnl) }, { t: "最大回撤", num: 1, f: (x) => num(x.max_drawdown) }], Object.values(pf.scenarios))
    + "<br>" + table([{ t: "胜率折减", f: (x) => pct(x.win_rate_haircut, 0) }, { t: "期末盈亏 P5", num: 1, f: (x) => sgn(x.pnl_p05, 1) }, { t: "中位数", num: 1, f: (x) => sgn(x.pnl_p50, 1) }, { t: "P95", num: 1, f: (x) => sgn(x.pnl_p95, 1) },
      { t: "亏损概率", num: 1, f: (x) => pct(x.prob_loss, 0) }, { t: "最大回撤 P95", num: 1, f: (x) => num(x.max_drawdown_p95, 1) }, { t: "触及日亏上限频率", num: 1, f: (x) => pct(x.daily_limit_breach_rate, 1) }],
      pf.stress && !pf.stress.note ? Object.values(pf.stress) : [], "无交易，未做压力测试");
}
async function loadModels() {
  const d = await api("/api/models");
  $("#m-table").innerHTML = table([
    { t: "版本", f: (r) => esc(r.version) + (r.is_champion ? ' <span class="badge ok">线上</span>' : "") }, { t: "创建时间", f: (r) => tm(r.created) }, { t: "测试窗口", f: (r) => esc((r.holdout_window || []).join(" ~ ")) },
    { t: "各任务测试集(模型 / 信号数 / 胜率)", f: (r) => Object.entries(r.summary || {}).map(([k, v]) => `${esc(k)}: ${esc(v.family)} / ${v.signals ?? 0} / ${pct(v.win_rate, 1)}`).join("<br>") },
    { t: "组合模拟盈亏", num: 1, f: (r) => sgn(r.portfolio?.total_pnl) }, { t: "操作", f: (r) => (r.is_champion ? "" : `<button class="small" data-promote="${esc(r.version)}">设为线上</button>`) },
  ], d.versions, "暂无模型版本");
}
async function loadSettings() {
  const [cfg, odds] = await Promise.all([api("/api/config"), api("/api/odds")]);
  $("#s-token").value = localStorage.getItem("becai_token") || "";
  $("#s-mode").value = cfg.payout.mode; $("#s-fixed").value = cfg.payout.fixed;
  $("#s-be").textContent = `盈亏平衡胜率 = 1/(1+赔率) = ${pct(1 / (1 + cfg.payout.fixed))}`;
  const tasks = cfg.symbols.flatMap((s) => cfg.horizons.flatMap((h) => ["HIGHER", "LOWER"].map((d) => ({ s, h, d }))));
  $("#s-odds").innerHTML = cfg.payout.mode === "manual" ? `<p class="note">手工录入模式：赔率超过 ${odds.max_age_seconds} 秒未更新即视为过期，此时只输出研究预测。</p><div class="kv">` + tasks.map((t) => {
    const cur = odds.manual.find((o) => o.symbol === t.s && o.horizon === t.h && o.direction === t.d);
    return `<label>${esc(t.s.replace("USDT", ""))} ${t.h}m ${t.d}<input type="number" step="0.01" data-odds="${t.s}|${t.h}|${t.d}" value="${cur ? cur.payout : ""}" placeholder="如 0.80"></label>`; }).join("") + "</div>" : "";
  const keys = ["stake_usdt", "max_order_usdt", "max_open_positions", "max_total_open_usdt", "max_symbol_open_usdt", "max_directional_usdt", "min_spacing_minutes", "max_same_direction_streak", "daily_loss_limit_usdt", "max_consecutive_losses", "cooldown_minutes", "min_edge"];
  const cn = { stake_usdt: "单笔本金 USDT", max_order_usdt: "单笔上限 USDT", max_open_positions: "未到期合约数上限", max_total_open_usdt: "未到期本金总额上限", max_symbol_open_usdt: "单币种未到期本金上限", max_directional_usdt: "相关性加权同向敞口上限", min_spacing_minutes: "同任务最小开仓间隔(分钟)", max_same_direction_streak: "连续同向开仓上限", daily_loss_limit_usdt: "每日亏损上限 USDT", max_consecutive_losses: "连续亏损触发冷静", cooldown_minutes: "冷静期(分钟)", min_edge: "最低期望收益/USDT" };
  $("#s-risk").innerHTML = '<div class="kv">' + keys.map((k) => `<label>${cn[k]}<input type="number" step="any" data-risk="${k}" value="${cfg.risk[k]}"></label>`).join("") + "</div>";
  $("#s-config").textContent = JSON.stringify(cfg, null, 2);
}

const LOADERS = { overview: loadOverview, live: loadLive, history: loadHistory, risk: loadRisk, data: loadData, train: loadTrain, board: loadBoard, features: loadFeatures, backtest: loadBacktest, models: loadModels, settings: loadSettings };
async function refresh() { try { await LOADERS[state.tab](); if (state.tab !== "overview") loadHeader(); } catch (e) { console.error(e); } }
function show(tab) { state.tab = tab; $$("nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab)); $$(".tab").forEach((s) => s.classList.toggle("on", s.id === "tab-" + tab)); refresh(); }

document.addEventListener("click", (e) => {
  const t = e.target;
  if (t.matches("nav button")) return show(t.dataset.tab);
  if (t.dataset.log) return act(async () => { $("#t-log").textContent = (await api(`/api/jobs/${t.dataset.log}/log`)) || "(空)"; });
  if (t.dataset.stop) return act(async () => { if (confirm("确定停止该任务？已完成的阶段会保留，可稍后恢复。")) { await api(`/api/jobs/${t.dataset.stop}/stop`, { method: "POST" }); refresh(); } });
  if (t.dataset.resume) return act(async () => { await api(`/api/jobs/${t.dataset.resume}/resume`, { method: "POST" }); refresh(); });
  if (t.dataset.promote) return act(async () => { if (confirm(`把 ${t.dataset.promote} 设为线上版本？`)) { await api(`/api/models/${encodeURIComponent(t.dataset.promote)}/promote`, { method: "POST" }); refresh(); } });
  if (t.dataset.order) return act(async () => { await api(`/api/orders/${t.dataset.order}/result`, { method: "POST", body: { result: t.dataset.res } }); refresh(); });
});
document.addEventListener("change", (e) => {
  if (e.target.matches(".run-select")) { state.run = e.target.value; refresh(); }
  if (e.target.matches(".task-select")) { state.task = e.target.value; refresh(); }
});
$("#h-refresh").onclick = () => { state.page = 0; refresh(); };
$("#h-prev").onclick = () => { state.page = Math.max(0, state.page - 1); refresh(); };
$("#h-next").onclick = () => { state.page += 1; refresh(); };
$("#data-sync").onclick = () => act(async () => { const r = await api("/api/jobs", { method: "POST", body: { type: "data-sync" } }); alert("已启动数据更新任务 #" + r.id + "，可在「训练任务」页查看日志"); });
$("#t-start").onclick = () => act(async () => {
  const body = { type: "train", run_id: $("#t-runid").value.trim() || undefined, until: $("#t-until").value || undefined, force_holdout: $("#t-force").checked || undefined };
  const r = await api("/api/jobs", { method: "POST", body }); alert("已启动训练任务 #" + r.id); refresh();
});
$("#m-rollback").onclick = () => act(async () => { if (confirm("回退到上一个线上版本？")) { const r = await api("/api/models/rollback", { method: "POST" }); alert("已回退到 " + r.version); refresh(); } });
$("#o-add").onclick = () => act(async () => {
  await api("/api/orders", { method: "POST", body: { symbol: $("#o-symbol").value, horizon: Number($("#o-horizon").value), direction: $("#o-direction").value, stake: Number($("#o-stake").value), payout: Number($("#o-payout").value), entry_price: Number($("#o-entry").value) || null } });
  refresh();
});
$("#s-token-save").onclick = () => { localStorage.setItem("becai_token", $("#s-token").value); alert("已保存"); };
$("#s-payout-save").onclick = () => act(async () => {
  await api("/api/settings", { method: "POST", body: { payout: { mode: $("#s-mode").value, fixed: Number($("#s-fixed").value) } } });
  for (const inp of $$("[data-odds]")) { if (inp.value) { const [symbol, horizon, direction] = inp.dataset.odds.split("|"); await api("/api/odds", { method: "POST", body: { symbol, horizon: Number(horizon), direction, payout: Number(inp.value) } }); } }
  refresh();
});
$("#s-risk-save").onclick = () => act(async () => {
  const risk = {}; $$("[data-risk]").forEach((i) => (risk[i.dataset.risk] = Number(i.value)));
  const r = await api("/api/settings", { method: "POST", body: { risk } }); $("#s-msg").textContent = r.note;
});
$$("canvas").forEach((cv) => { cv.style.height = cv.getAttribute("height") + "px"; });   // 固定画布高度，避免无数据时被按宽高比拉伸
setInterval(() => { if (["overview", "live", "train", "risk"].includes(state.tab)) refresh(); }, 10000);
refresh();
