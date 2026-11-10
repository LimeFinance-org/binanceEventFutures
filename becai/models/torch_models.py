"""深度学习候选模型：MLP / GRU / TCN / Transformer / 多时间尺度融合，多任务输出。

输入由三部分组成（是否使用由搜索决定）：
  * 多尺度序列：对逐分钟通道做 s 分钟聚合后，取最近 L 步（步长 s），每个尺度一个编码器；
  * 表格特征：筛选后的前 n_tab 个特征；
  * 币种嵌入：跨币种共享模型时使用。
输出：每个预测期限一个 logit（多任务学习）。全部窗口在设备上按索引即时抽取，不落盘。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bundle import DataBundle
from .base import Candidate, Task


# ---------------- 网络结构 ----------------
class GRUEncoder(nn.Module):
    def __init__(self, c_in: int, d: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(c_in, d)
        self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)

    def forward(self, x):
        out, _ = self.gru(self.inp(x))
        return out[:, -1]


class _TCNBlock(nn.Module):
    def __init__(self, d: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(d, d, k, dilation=dilation)
        self.c2 = nn.Conv1d(d, d, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # 因果卷积：只在左侧补零
        y = self.drop(F.gelu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.gelu(self.c2(F.pad(y, (self.pad, 0)))))
        return x + y


class TCNEncoder(nn.Module):
    def __init__(self, c_in: int, d: int, layers: int, dropout: float, kernel: int = 3):
        super().__init__()
        self.inp = nn.Conv1d(c_in, d, 1)
        self.blocks = nn.Sequential(*[_TCNBlock(d, kernel, 2 ** i, dropout) for i in range(layers)])

    def forward(self, x):
        return self.blocks(self.inp(x.transpose(1, 2)))[:, :, -1]


class TransformerEncoder(nn.Module):
    """分块（patch）Transformer：先把序列切成小块做嵌入，再做自注意力。"""

    def __init__(self, c_in: int, d: int, layers: int, dropout: float, seq_len: int, patch: int = 4, heads: int = 4):
        super().__init__()
        self.patch = max(1, min(patch, seq_len))
        self.embed = nn.Conv1d(c_in, d, self.patch, stride=self.patch)
        n_tok = seq_len // self.patch
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        L = (x.shape[1] // self.patch) * self.patch
        tok = self.embed(x[:, -L:].transpose(1, 2)).transpose(1, 2) + self.pos
        h = self.norm(self.enc(tok))
        return torch.cat([h[:, -1], h.mean(1)], dim=1)


class FusionNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, layers, drop = cfg["d_model"], cfg["n_layers"], cfg["dropout"]
        self.encoders = nn.ModuleList()
        width = 0
        for _ in cfg["scales"]:
            if cfg["arch"] == "gru":
                self.encoders.append(GRUEncoder(cfg["n_channels"], d, layers, drop)); width += d
            elif cfg["arch"] == "tcn":
                self.encoders.append(TCNEncoder(cfg["n_channels"], d, layers, drop)); width += d
            else:
                self.encoders.append(TransformerEncoder(cfg["n_channels"], d, layers, drop, cfg["seq_len"], cfg.get("patch", 4), cfg.get("heads", 4)))
                width += 2 * d
        self.tab = None
        if cfg["n_tab"] > 0:
            self.tab = nn.Sequential(nn.Linear(cfg["n_tab"], cfg["tab_hidden"]), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(cfg["tab_hidden"], cfg["tab_hidden"]), nn.GELU())
            width += cfg["tab_hidden"]
        self.sym = nn.Embedding(cfg["n_symbols"], 8) if cfg["n_symbols"] > 1 else None
        width += 8 if self.sym is not None else 0
        self.trunk = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, cfg["head_hidden"]), nn.GELU(), nn.Dropout(drop))
        self.heads = nn.Linear(cfg["head_hidden"], cfg["n_out"])

    def forward(self, seqs: list[torch.Tensor], tab: torch.Tensor | None, sym: torch.Tensor) -> torch.Tensor:
        parts = [enc(x) for enc, x in zip(self.encoders, seqs)]
        if self.tab is not None:
            parts.append(self.tab(tab))
        if self.sym is not None:
            parts.append(self.sym(sym))
        return self.heads(self.trunk(torch.cat(parts, dim=1)))


# ---------------- 候选模型 ----------------
class TorchCandidate(Candidate):
    family = "torch"

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.symbols = list(dict.fromkeys(s for s, _ in self.tasks))
        self.horizons = sorted({h for _, h in self.tasks})
        p = self.params
        self.scales = [int(s) for s in p.get("scales", [1])] if p.get("arch", "mlp") != "mlp" else []
        self.seq_len = int(p.get("seq_len", 64))
        self.net: FusionNet | None = None
        self.norm: dict = {}
        self.net_cfg: dict = {}
        self.info: dict = {}

    # ---- 张量准备 ----
    def _dev(self) -> torch.device:
        return torch.device("cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu")

    def _agg_channels(self, ch: np.ndarray, s: int) -> np.ndarray:
        """s 分钟因果聚合：agg[i] = mean(ch[i-s+1 .. i])。"""
        if s == 1:
            return np.asarray(ch, dtype=np.float32)
        cs = np.cumsum(np.asarray(ch, dtype=np.float64), axis=0)
        out = cs.copy()
        out[s:] = cs[s:] - cs[:-s]
        return (out / s).astype(np.float32)

    def _tensors(self, bundle: DataBundle, dev: torch.device, lo: int, hi: int) -> dict:
        """把 [lo, hi) 行（含回看余量）搬到设备。返回的索引以 lo 为零点。"""
        need = max([self.seq_len * s for s in self.scales], default=0) + max(self.scales, default=0)
        lo2 = max(0, lo - need)
        seq = []
        for s in self.scales:
            per_sym = [self._agg_channels(bundle.symbols[sym].channels[max(0, lo2 - s):hi], s)[min(s, lo2):] for sym in self.symbols]
            seq.append(torch.from_numpy(np.stack(per_sym)).to(dev))                 # [S, n, C]
        tab = None
        if self.features:
            cols = self.feature_idx(bundle)
            tab = torch.from_numpy(np.stack([np.asarray(bundle.symbols[sym].X[lo2:hi])[:, cols] for sym in self.symbols])).to(dev)
        return {"seq": seq, "tab": tab, "offset": lo2}

    def _fit_norm(self, T: dict, sym_ids: torch.Tensor, rows: torch.Tensor) -> None:
        sub = rows[:: max(1, len(rows) // 200_000)]
        sid = sym_ids[:: max(1, len(rows) // 200_000)]
        self.norm = {"seq": [], "tab": None}
        for x in T["seq"]:
            v = x[sid, sub]
            self.norm["seq"].append((v.mean(0).cpu().numpy(), (v.std(0) + 1e-6).cpu().numpy()))
        if T["tab"] is not None:
            v = T["tab"][sid, sub]
            mean = torch.nanmean(v, dim=0)
            var = torch.nanmean((v - mean) ** 2, dim=0)
            mean = torch.nan_to_num(mean, nan=0.0)
            std = torch.nan_to_num(var.sqrt(), nan=1.0).clamp_min(1e-6)
            self.norm["tab"] = (mean.cpu().numpy(), std.cpu().numpy())

    def _batch(self, T: dict, sid: torch.Tensor, rows: torch.Tensor, dev: torch.device):
        seqs = []
        for k, s in enumerate(self.scales):
            off = torch.arange(self.seq_len - 1, -1, -1, device=dev) * s
            idx = (rows[:, None] - off[None, :]).clamp_min(0)
            x = T["seq"][k][sid[:, None], idx]
            m, sd = self._norm_t["seq"][k]
            seqs.append(((x - m) / sd).clamp(-8, 8))
        tab = None
        if T["tab"] is not None:
            m, sd = self._norm_t["tab"]
            tab = torch.nan_to_num(((T["tab"][sid, rows] - m) / sd).clamp(-5, 5), nan=0.0)
        return seqs, tab

    def _norm_to(self, dev: torch.device) -> None:
        self._norm_t = {"seq": [(torch.tensor(m, device=dev), torch.tensor(s, device=dev)) for m, s in self.norm["seq"]],
                        "tab": None if self.norm["tab"] is None else tuple(torch.tensor(a, device=dev) for a in self.norm["tab"])}

    def _samples(self, bundle: DataBundle, rows: np.ndarray):
        """(币种id, 行, 标签[H], 掩码[H])；至少一个期限标签有效的行。"""
        sids, rr, ys, ms = [], [], [], []
        for i, sym in enumerate(self.symbols):
            sd = bundle.symbols[sym]
            lab = np.stack([sd.labels[h][rows] for h in self.horizons], axis=1)
            mask = (lab >= 0) & sd.feat_ok[rows][:, None]
            keep = mask.any(axis=1)
            sids.append(np.full(int(keep.sum()), i)); rr.append(rows[keep])
            ys.append(np.clip(lab[keep], 0, 1).astype(np.float32)); ms.append(mask[keep].astype(np.float32))
        return np.concatenate(sids), np.concatenate(rr), np.concatenate(ys), np.concatenate(ms)

    # ---- 训练 ----
    def fit(self, bundle: DataBundle, train_rows: np.ndarray, es_rows: np.ndarray) -> dict:
        p = self.params
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        dev = self._dev()
        lo, hi = int(min(train_rows.min(), es_rows.min())), int(max(train_rows.max(), es_rows.max())) + 1
        T = self._tensors(bundle, dev, lo, hi)
        to_t = lambda a, dt=None: torch.as_tensor(a, device=dev, dtype=dt)  # noqa: E731
        s_tr, r_tr, y_tr, m_tr = self._samples(bundle, train_rows)
        s_es, r_es, y_es, m_es = self._samples(bundle, es_rows)
        es_max = int(p.get("es_max_samples", 150_000))
        if len(r_es) > es_max:
            sel = np.arange(0, len(r_es), len(r_es) // es_max + 1)
            s_es, r_es, y_es, m_es = s_es[sel], r_es[sel], y_es[sel], m_es[sel]
        s_tr_t, r_tr_t = to_t(s_tr, torch.long), to_t(r_tr - T["offset"], torch.long)
        y_tr_t, m_tr_t = to_t(y_tr), to_t(m_tr)
        s_es_t, r_es_t, y_es_t, m_es_t = to_t(s_es, torch.long), to_t(r_es - T["offset"], torch.long), to_t(y_es), to_t(m_es)
        self._fit_norm(T, s_tr_t, r_tr_t)
        self._norm_to(dev)
        self.net_cfg = {
            "arch": p.get("arch", "mlp"), "scales": self.scales, "seq_len": self.seq_len, "n_channels": len(bundle.channel_names),
            "d_model": int(p.get("d_model", 64)), "n_layers": int(p.get("n_layers", 2)), "dropout": float(p.get("dropout", 0.1)),
            "patch": int(p.get("patch", 4)), "heads": int(p.get("heads", 4)), "n_tab": len(self.features),
            "tab_hidden": int(p.get("tab_hidden", 128)), "head_hidden": int(p.get("head_hidden", 128)),
            "n_symbols": len(self.symbols), "n_out": len(self.horizons),
        }
        net = FusionNet(self.net_cfg).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=float(p.get("lr", 1e-3)), weight_decay=float(p.get("weight_decay", 1e-4)))
        epochs, steps, bs = int(p.get("max_epochs", 30)), int(p.get("steps_per_epoch", 400)), int(p.get("batch_size", 4096))
        steps = max(1, min(steps, len(r_tr) // bs + 1))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(p.get("lr", 1e-3)), total_steps=epochs * steps)
        use_amp = bool(p.get("amp", True)) and dev.type == "cuda"

        def run(seqs, tab, sid):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                return net(seqs, tab, sid).float()

        def evaluate() -> float:
            net.eval()
            tot, cnt = 0.0, 0.0
            with torch.no_grad():
                for i in range(0, len(r_es_t), 16384):
                    sl = slice(i, i + 16384)
                    seqs, tab = self._batch(T, s_es_t[sl], r_es_t[sl], dev)
                    loss = F.binary_cross_entropy_with_logits(run(seqs, tab, s_es_t[sl]), y_es_t[sl], reduction="none")
                    tot += float((loss * m_es_t[sl]).sum()); cnt += float(m_es_t[sl].sum())
            return tot / max(cnt, 1.0)

        best, best_state, bad, hist = float("inf"), None, 0, []
        patience = int(p.get("patience", 4))
        n = len(r_tr_t)
        for ep in range(epochs):
            net.train()
            for _ in range(steps):
                b = torch.randint(0, n, (bs,), device=dev)
                seqs, tab = self._batch(T, s_tr_t[b], r_tr_t[b], dev)
                loss = F.binary_cross_entropy_with_logits(run(seqs, tab, s_tr_t[b]), y_tr_t[b], reduction="none")
                loss = (loss * m_tr_t[b]).sum() / m_tr_t[b].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
            val = evaluate()
            hist.append(val)
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.eval()
        self.info = {"es_logloss": best, "epochs": len(hist), "n_train": int(n), "n_params": sum(q.numel() for q in net.parameters())}
        del T
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return self.info

    # ---- 预测 ----
    def predict(self, bundle: DataBundle, rows: np.ndarray) -> dict[Task, np.ndarray]:
        dev = self._dev()
        self.net.to(dev).eval()
        self._norm_to(dev)
        out = {t: np.full(len(rows), np.nan) for t in self.tasks}
        if len(rows) == 0:
            return out
        T = self._tensors(bundle, dev, int(rows.min()), int(rows.max()) + 1)
        r_t = torch.as_tensor(rows - T["offset"], device=dev, dtype=torch.long)
        use_amp = bool(self.params.get("amp", True)) and dev.type == "cuda"
        with torch.no_grad():
            for i, sym in enumerate(self.symbols):
                probs = []
                for j in range(0, len(rows), 16384):
                    rr = r_t[j:j + 16384]
                    sid = torch.full((len(rr),), i, device=dev, dtype=torch.long)
                    seqs, tab = self._batch(T, sid, rr, dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        logit = self.net(seqs, tab, sid).float()
                    probs.append(torch.sigmoid(logit).cpu().numpy())
                pr = np.concatenate(probs)
                ok = bundle.symbols[sym].feat_ok[rows]
                for k, h in enumerate(self.horizons):
                    if (sym, h) in out:
                        out[(sym, h)] = np.where(ok, pr[:, k], np.nan)
        return out

    def _save_state(self, d: Path) -> None:
        torch.save({"state": self.net.state_dict(), "net_cfg": self.net_cfg, "norm": self.norm, "info": self.info}, d / "model.pt")

    def _load_state(self, d: Path) -> None:
        st = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        self.net_cfg, self.norm, self.info = st["net_cfg"], st["norm"], st.get("info", {})
        self.net = FusionNet(self.net_cfg)
        self.net.load_state_dict(st["state"])
        self.net.eval()
