"""逐秒持续性判据 —— 替掉"整窗压成一个标量再卡阈值"。

为什么换
--------
上一轮四路融合把整个事件窗压成 2 个标量（EEG line length 的 P90、
EMG 最长连续窗的分位），LOO 只到 0.321/h。瓶颈定位在聚合方式：
有发作的 `eeg_ll_p` 最小值是 0.00，整窗聚合值落在它自己背景分布
的最低端，零漏报约束被这一个事件钉死，门槛提不上去。

行业做法不是这样。两个独立来源都用"逐段 + 持续性"：
  · NightWatch（Lazeron 2022）最终判据：
    水平体位 且（心率超阈 或 **运动值连续超阈 ≥7 秒**）
  · SeizeIT2 自己的后处理：**10 个 1 秒段里至少 8 个**判为发作

共同点：先逐秒/逐段做一个便宜的二值判断，再要求**时间上连成一片**。
这样做抗的是两类误报：
  · 短突发伪迹（翻身、碰撞）—— 连不成片
  · 长而断续的噪声（电极接触不良）—— 同样连不成片
而发作是连续数十秒的稳定高频活动，正好能连成片。

阈值一律用"本记录背景分位"，不用绝对值
（教训见 [[nightguard-absolute-thresholds-fail]]：
 20 个 recording 的背景 P99 从 0 跨到 200，绝对阈值无意义）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SustainConfig:
    # 逐段判"异常"的分位门：该段特征超过本记录背景分布的这个分位
    seg_pct: float = 0.90
    # NightWatch 用 7 秒连续。我们的段是 1 秒、步长 0.5 秒，
    # 所以 7 秒 ≈ 14 段（含重叠）。
    min_run_s: float = 7.0
    # SeizeIT2 式的滑动投票：窗口内至少这么多比例的段异常。
    # 比"严格连续"宽松一点，容忍单段掉落（发作中途的伪迹）。
    vote_win_s: float = 10.0
    vote_frac: float = 0.8
    step_s: float = 0.5


def longest_run_s(flags, step_s: float):
    """最长连续 True 的秒数。None 视为中断。"""
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * step_s


def best_vote_frac(flags, cfg: SustainConfig = None):
    """滑动投票：返回所有窗口里"异常段占比"的最大值。

    对应 SeizeIT2 的 8/10 规则 —— 不要求严格连续，
    允许发作中途有个别段被伪迹打断。
    """
    cfg = cfg or SustainConfig()
    n = max(1, int(round(cfg.vote_win_s / cfg.step_s)))
    if len(flags) < n:
        # 段数不够一个投票窗：用全部段的占比，不要直接判否
        vals = [f for f in flags if f is not None]
        return (sum(1 for f in vals if f) / len(vals)) if vals else None
    best = 0.0
    for s in range(0, len(flags) - n + 1):
        seg = [f for f in flags[s:s + n] if f is not None]
        if not seg:
            continue
        best = max(best, sum(1 for f in seg if f) / len(seg))
    return best


def sustain_evidence(seg_vals, bg_vals, cfg: SustainConfig = None):
    """逐段判异常，再看时间上连不连成片。

    seg_vals: 事件窗内每段的特征值（按时间顺序）
    bg_vals:  **本记录**背景段的同一特征值（用来定分位门）

    返回 {thr, run_s, vote, n_seg} 或 None（数据不足）。
    run_s = 最长连续异常秒数；vote = 滑动投票的最大异常占比。
    """
    cfg = cfg or SustainConfig()
    vals = [v for v in seg_vals if v is not None]
    bg = sorted(v for v in bg_vals if v is not None)
    if len(vals) < 3 or len(bg) < 20:
        return None
    k = min(len(bg) - 1, max(0, int(round(cfg.seg_pct * (len(bg) - 1)))))
    thr = bg[k]
    flags = [(None if v is None else v > thr) for v in seg_vals]
    return {
        "thr": float(thr),
        "run_s": float(longest_run_s(flags, cfg.step_s)),
        "vote": best_vote_frac(flags, cfg),
        "n_seg": len(vals),
    }


def sustain_keep(ev, cfg: SustainConfig = None):
    """持续性是否达标。ev=None（数据不足）→ True，降级放行不漏报。"""
    cfg = cfg or SustainConfig()
    if ev is None:
        return True
    if ev["run_s"] >= cfg.min_run_s:
        return True
    return ev["vote"] is not None and ev["vote"] >= cfg.vote_frac
