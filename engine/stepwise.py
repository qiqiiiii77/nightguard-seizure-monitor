"""逐步条件满足（stepwise fulfillment）—— 替代"加权凑分"的报警判据。

为什么需要它
------------
我们现在是 noisy-OR 加权求和：任意信号都能往 risk 里"凑分"。
实测（engine/gate.py 的 influence 表）：
  · motion_burst 单独满分 = 0.400，刚好越过 θ=0.40
  · 三个 doubt=0.5 的中等信号叠加 = 0.387，差一点就报
也就是说报警可以由"几个都不太确定的信号"拼出来，没有任何一个
信号真正达标。

van Andel 2017（Seizure 49:56-62，NightWatch 前身）原文把算法描述为
"based on stepwise fulfillment of conditions during increases in
either feature" —— 必须**某个特定信号**先达标，才进入下一级判断。
他们从那篇的 2.3-5.7 次/夜 走到 Arends 2018 的 0.04/h，
结构就是这种逐级判据。

设计：只做减法
-------------
和 LLM 裁决、个体化 novelty 一样挂在引擎候选之后，只允许否决。
引擎 θ=0.40 的 17/17 召回是上界；任何一级判据失灵时按"保留"降级，
宁可多报不可漏报。

三条通路（任一成立即保留报警）
  A 运动通路：体动达标 → 且有节律性 → 且持续够久
  B 自主神经通路：心率升幅达标 → 且持续 ≥45s（医生判据）
  C 组合通路：体动与心率**同一时间窗**同步上升（单独都不够，同步才算）
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _f(x, d=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


@dataclass
class StepwiseConfig:
    # --- A 运动通路 ---
    # 这三个数来自 validation/sweep_stepwise.py 的 64 组网格扫描。
    # 我第一版凭直觉写 3.0 / 0.35 / 10.0，结果 163 个误报里 147 个
    # 和全部 17 次发作走同一条 A 通路 —— 等于没筛。
    # 扫描结论：15× / 0.45 / 30s 是零漏报组合里误报最低的（0.84/h 全集）。
    # 留一患者交叉验证 16/17 @ 0.81/h —— 这才是对新患者的真实预期。
    mot_ratio_min: float = 15.0       # 体动峰值至少是个人基线的几倍
    psd_min: float = 0.45            # 1.5-5Hz 节律占比
    mot_sustain_s: float = 30.0       # 体动高于门限的持续秒数

    # --- B 自主神经通路 ---
    hr_rise_bpm: float = 30.0         # 医生判据：升 30-60bpm
    hr_sustain_s: float = 45.0        # 医生判据：持续 >45 秒

    # --- C 同步通路 ---
    # A 收紧后，51 个误报改从 C 漏过来，C 成了最松的门；而 C 只保住 1 次发作。
    # 下面三个数同样来自 LOO：16 次留一里有 13 次选中这一组。
    sync_mot_ratio: float = 8.0       # 同步时各自的门限可以放松，但不能太松
    sync_hr_rise: float = 35.0
    sync_window_s: float = 8.0        # 两者峰值相距不超过这么久

    hop_s: float = 2.0


def _runs_over(vals, thr, hop):
    """超过阈值的最长连续秒数。"""
    best = cur = 0
    for v in vals:
        if v is not None and v > thr:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * hop


def evaluate(windows: list, baseline: dict, cfg: StepwiseConfig = None) -> dict:
    """返回 {keep, path, steps} —— keep=False 表示三条通路都没走通。"""
    cfg = cfg or StepwiseConfig()
    if not windows:
        return {"keep": True, "path": None, "steps": ["无窗口数据→保留"]}

    bm = max(_f(baseline.get("motion"), 1e-4), 1e-6)
    bh = _f(baseline.get("hr"), 0.0)

    mot = [_f(w.get("motion")) for w in windows]
    psd = [_f(w.get("motion_psd")) for w in windows]
    # hr_q: 1.0=高置信（见 wearable_features.py:138）
    hr = [(_f(w.get("hr")) if w.get("hr") is not None
           and (w.get("hr_q") is None or _f(w.get("hr_q"), 1.0) >= 0.5)
           else None) for w in windows]

    t = [_f(w.get("t")) for w in windows]
    steps = []

    # ---------------- A 运动通路 ----------------
    mot_ratio = (max(mot) / bm) if mot else 0.0
    a1 = mot_ratio >= cfg.mot_ratio_min
    steps.append(f"A1 体动 {mot_ratio:.1f}×基线 ≥{cfg.mot_ratio_min} → {'过' if a1 else '否'}")
    if a1:
        a2 = max(psd) >= cfg.psd_min if psd else False
        steps.append(f"A2 节律占比 {max(psd) if psd else 0:.2f} ≥{cfg.psd_min} → {'过' if a2 else '否'}")
        if a2:
            sus = _runs_over(mot, bm * cfg.mot_ratio_min * 0.5, cfg.hop_s)
            a3 = sus >= cfg.mot_sustain_s
            steps.append(f"A3 体动持续 {sus:.0f}s ≥{cfg.mot_sustain_s} → {'过' if a3 else '否'}")
            if a3:
                return {"keep": True, "path": "A-运动", "steps": steps}

    # ---------------- B 自主神经通路 ----------------
    hv = [x for x in hr if x is not None]
    if hv and bh > 0:
        rise = max(hv) - bh
        b1 = rise >= cfg.hr_rise_bpm
        steps.append(f"B1 心率升 {rise:.0f}bpm ≥{cfg.hr_rise_bpm} → {'过' if b1 else '否'}")
        if b1:
            sus = _runs_over(hr, bh + cfg.hr_rise_bpm * 0.6, cfg.hop_s)
            b2 = sus >= cfg.hr_sustain_s
            steps.append(f"B2 心率持续 {sus:.0f}s ≥{cfg.hr_sustain_s} → {'过' if b2 else '否'}")
            if b2:
                return {"keep": True, "path": "B-自主神经", "steps": steps}
    else:
        steps.append("B 跳过（无可用心率或无基线）")

    # ---------------- C 同步通路 ----------------
    if hv and bh > 0 and mot:
        im = int(max(range(len(mot)), key=lambda i: mot[i]))
        ih = max((i for i in range(len(hr)) if hr[i] is not None),
                 key=lambda i: hr[i], default=None)
        if ih is not None:
            gap = abs(t[im] - t[ih])
            c1 = (mot_ratio >= cfg.sync_mot_ratio
                  and (max(hv) - bh) >= cfg.sync_hr_rise
                  and gap <= cfg.sync_window_s)
            steps.append(
                f"C 同步：体动 {mot_ratio:.1f}× / 心率升 {max(hv)-bh:.0f}bpm / "
                f"峰值相距 {gap:.0f}s ≤{cfg.sync_window_s} → {'过' if c1 else '否'}")
            if c1:
                return {"keep": True, "path": "C-同步", "steps": steps}

    return {"keep": False, "path": None, "steps": steps}
