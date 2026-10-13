"""耳后双通道（behind-the-ear）EEG 的发作期特征。

为什么加这一路：
  SeizeIT2 自带的 EEG 是 **Byteflies SensorDot 的耳后导联**（电极贴在
  耳后颞骨上），不是临床头皮 EEG —— 属于我们能复刻的可穿戴形态。
  同一数据集、同为强直-阵挛的已发表基准（arXiv 2403.13066）：
    · 耳后 EEG 单独：100% 灵敏度 @ 10.3 FP/24h
    · EMG 单独：      97.7% @ 30.9 FP/24h
    · ACC 单独：      95.5% @ 13.9 FP/24h
    · **EEG + EMG 融合：97.7% @ 0.4 FP/24h ≈ 0.017/h**
  0.017/h 低于我们 0.05/h 的目标，而且是在同一批数据上做出来的。
  EEG 是其中最强的单通道，我们之前整条链路没有它。

通道标签按患者变：左侧起源的是 'BTEleft SD'，右侧的是 'BTEright SD'，
都带一个 'CROSStop SD'（跨头导联）。所以读取必须按标签前缀匹配，
不能写死通道名（已验：sub-046 是 BTEleft，sub-103 是 BTEright）。

特征选择沿用癫痫检测的常规量，并照抄数据集自己的后处理门槛：
  · 原文后处理：丢掉幅值在 13-150µV 之外的段，且要求 10 个 1 秒段里
    至少 8 个被判为发作才报警。前者是伪迹门（太小=没接触，太大=运动伪迹），
    后者是时间持续性要求。两条都直接复用。
  · 线长（line length）：对发作期的节律性放电很敏感，计算便宜。
  · 频带能量比：发作期 theta/alpha 段能量占比上升。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EegBteConfig:
    fs: float = 256.0
    win_s: float = 1.0           # 原文用 1 秒段
    step_s: float = 0.5
    # 幅值门（伪迹门）：原文（SeizeIT2 自己的后处理）用 13-150µV。
    # **上限不能照抄 150**，实测会把发作数据大批丢掉：
    #   这批耳后 EEG 的 1 秒段峰峰值，发作段中位 181µV、P90 1155µV；
    #   13-150 的门只留 43% 的发作段（误报段留 19%）。
    # 门的本意是挡"电极没接触"（幅值≈0）和"大运动伪迹"（幅值爆表），
    # 不是挡发作本身。放宽上限保留本意：13-1000µV → 发作留 88%，误报留 69%。
    # 判别力交给 line length / 频带比，不靠幅值门。
    amp_min_uv: float = 13.0
    amp_max_uv: float = 1000.0
    # 原文：10 个 1 秒段里至少 8 个判为发作
    vote_win: int = 10
    vote_need: int = 8


def _goertzel_power(x, fs: float, f0: float):
    """单频点功率（Goertzel）—— 比整条 FFT 便宜，engine 不引 numpy/scipy。"""
    n = len(x)
    if n == 0:
        return 0.0
    k = int(0.5 + n * f0 / fs)
    w = 2.0 * math.pi * k / n
    cw, sw = math.cos(w), math.sin(w)
    c = 2.0 * cw
    s1 = s2 = 0.0
    for v in x:
        s0 = v + c * s1 - s2
        s2, s1 = s1, s0
    return (s1 * s1 + s2 * s2 - c * s1 * s2) / (n * n)


def band_power(x, fs: float, lo: float, hi: float, step: float = 2.0):
    """[lo,hi) 频带功率：在带内按 step 取若干频点求和（够用的粗估）。"""
    f, tot = lo, 0.0
    while f < hi:
        tot += _goertzel_power(x, fs, f)
        f += step
    return tot


def line_length(x):
    """线长：相邻样本差的绝对值之和。对节律性放电敏感、计算便宜。"""
    if len(x) < 2:
        return 0.0
    return sum(abs(x[i] - x[i - 1]) for i in range(1, len(x)))


def seg_features(x, fs: float):
    """一个段的特征。x 必须是 µV。"""
    n = len(x)
    if n == 0:
        return None
    mean = sum(x) / n
    xc = [v - mean for v in x]
    amp = max(xc) - min(xc)
    var = sum(v * v for v in xc) / n
    theta = band_power(xc, fs, 4.0, 8.0)
    alpha = band_power(xc, fs, 8.0, 13.0)
    beta = band_power(xc, fs, 13.0, 30.0)
    delta = band_power(xc, fs, 1.0, 4.0)
    tot = delta + theta + alpha + beta + 1e-12
    return {
        "amp": amp,
        "rms": math.sqrt(var),
        "ll": line_length(xc) / n,
        "theta_r": theta / tot,
        "alpha_r": alpha / tot,
        "beta_r": beta / tot,
        "ta_r": (theta + alpha) / tot,
    }


def feature_series(sig_uv, cfg: EegBteConfig = None):
    """返回 [(段起始秒, 特征dict 或 None)]。

    None 表示该段被幅值门丢弃（原文的 13-150µV 规则）——
    注意区分"被丢弃"和"特征低"：前者是电极没贴好/大运动伪迹，
    不能当成"没有发作"的证据。
    """
    cfg = cfg or EegBteConfig()
    w = int(round(cfg.win_s * cfg.fs))
    step = max(1, int(round(cfg.step_s * cfg.fs)))
    out = []
    for s in range(0, max(0, len(sig_uv) - w + 1), step):
        seg = sig_uv[s:s + w]
        f = seg_features(seg, cfg.fs)
        if f is None or not (cfg.amp_min_uv <= f["amp"] <= cfg.amp_max_uv):
            out.append((s / cfg.fs, None))
        else:
            out.append((s / cfg.fs, f))
    return out
