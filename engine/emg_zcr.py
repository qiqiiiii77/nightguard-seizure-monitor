"""三角肌单通道 sEMG 的强直期检测 —— 复用 Conradsen 2011 (IEEE TBME) 的成熟架构。

为什么用这个而不是继续调我们自己的体动门槛：
  · 该文用**同一块肌肉**（三角肌）的单通道 sEMG，在 11 位患者身上拿到
    100% 灵敏度、平均延迟 13.7s、误报 0.04/h —— 正好是我们的目标量级。
  · 关键一句：误报里只有约 10% 发生在夜间（12pm-8am）。
    夜间专用系统的实际误报率约"每十天一次"。我们正是夜间专用。
  · 该文还指出同一组参数可通用于所有患者（generic），不需要个体化调参。
  · SPEAC（Brain Sentinel，FDA 已清）走的同一条路：29/29 GTCS 全检出。

原文方法（逐条照搬）：
  1. 150Hz Butterworth 高通
  2. 零交叉计数，带 ±50µV 回差（hysteresis）—— 只有当前后峰分别超过
     +50µV 和 -50µV 时才计一次，用来把本底噪声（SD≈15µV，3SD≈50µV）挡在外面
  3. 1 秒窗、75% 重叠
  4. 阈值 ≈241 counts/窗（三角肌），连续 19 个窗超阈 → 报警

我们必须改的一处（以及为什么）：
  SeizeIT2 的 EMG 采样率是 256Hz → Nyquist 128Hz，**150Hz 高通在物理上不存在**。
  原文自己提到更早的工作发现"发作相对正常活动在 100Hz 以上占比很大"，
  所以这里退到 100Hz 高通（100-128Hz 这一段）。
  频带变窄 → 同一段信号的零交叉绝对数必然低于原文的 241，
  因此阈值**不能照抄 241**，必须在我们的数据上重新定（见 run_emg_zcr.py 的扫描）。
  这是适配，不是调参：窗长/重叠/回差/连续窗判据全部保持原文结构。

单位：EDF 的 phys 单位是 µV 还是 mV 由文件头决定，调用方负责把信号
换算成 **µV** 再传进来，否则 ±50 的回差完全不对。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EmgZcrConfig:
    fs: float = 256.0
    hp_hz: float = 100.0          # 原文 150Hz；256Hz 采样下退到 100Hz（Nyquist 128）
    hysteresis_uv: float = 50.0   # 原文 ±50µV ≈ 3×本底噪声 SD
    win_s: float = 1.0            # 原文 1 秒窗
    overlap: float = 0.75         # 原文 75% 重叠 → 步长 0.25s
    zc_threshold: float = 120.0   # 待在我们数据上定；原文三角肌 241（150Hz 高通）
    n_windows: int = 19           # 原文三角肌 19 个连续窗


def _butter_hp_sos(fs: float, fc: float):
    """二阶 Butterworth 高通，返回双二次系数。

    手写而不用 scipy：engine 要能跑在没有 scipy 的设备上，
    而且只需要一个定阶滤波器，没必要引依赖。
    双线性变换 + 预畸变（prewarp）。
    """
    if fc >= fs / 2:
        raise ValueError(f"高通 {fc}Hz 超过 Nyquist {fs/2}Hz，物理上不可实现")
    w = math.tan(math.pi * fc / fs)          # 预畸变
    k = math.sqrt(2.0)                        # Butterworth Q: 1/Q = sqrt(2)
    n = 1.0 + k * w + w * w
    b0, b1, b2 = 1.0 / n, -2.0 / n, 1.0 / n
    a1, a2 = 2.0 * (w * w - 1.0) / n, (1.0 - k * w + w * w) / n
    return (b0, b1, b2, a1, a2)


def highpass(x, fs: float, fc: float):
    """零初态的二阶高通。前若干样本有瞬态，调用方给足前导即可。"""
    b0, b1, b2, a1, a2 = _butter_hp_sos(fs, fc)
    out = [0.0] * len(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):
        y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y
        x2, x1 = x1, v
        y2, y1 = y1, y
    return out


def zero_crossings_hyst(x, hyst: float) -> int:
    """带回差的零交叉计数 —— 原文第 III-A 节的定义。

    只有当信号先升过 +hyst、再降过 -hyst（或反之）才记一次。
    这样本底噪声在零附近的抖动不会被计进来。
    """
    n = 0
    state = 0          # 0 未定，+1 上次在高位，-1 上次在低位
    for v in x:
        if v > hyst:
            if state == -1:
                n += 1
            state = 1
        elif v < -hyst:
            if state == 1:
                n += 1
            state = -1
    return n


def calibrate_hysteresis(sig_uv, fs: float, hp_hz: float, k: float = 3.0,
                         floor_uv: float = 0.0):
    """按本段信号自己的本底噪声定回差，而不是写死 ±50µV。

    为什么必须这么做（实测踩的坑）：
      原文的 ±50µV 来自"我们测到本底噪声 SD 高达 15µV，3SD≈50µV"。
      那是**他们那套电极**的噪声水平，不是一个普适常数。
      SeizeIT2 的 20 个 recording 里，背景零交叉数的 P99 从 0 一直到 200 ——
      有些患者的电极本底噪声高到能把 50µV 回差直接冲穿，于是整夜都在"报警"。
      用固定 50µV 时 sEMG 合取门几乎没降误报（0.48→0.445/h），
      因为误报段的零交叉峰值（中位 110）和发作段（中位 109）看着一样。

    做法保持原文的逻辑不变：回差 = k × 滤波后信号的噪声 SD。
    用中位绝对偏差（MAD）估 SD，避免发作/大体动段把估计拉高。

    ⚠ `floor_uv` 原为 50.0，是个**反方向的 bug**（2026-10-07 实测修掉）：
    上面那段注释讲的是"噪声大的电极会把 50µV 冲穿"，于是加了地板兜底。
    但对**干净**电极，k×sd 远小于 50，地板反过来把信号整个砍掉：
    20 条记录里 **18 条**被钉在 50.0，零交叉序列 58%-99.7% 全是 0，
    其中 5 条（sub-070/073/115/116）P99 都是 0 —— 整夜 sEMG 被清零。
    后果：sEMG 四个特征 AUC 全在 0.45-0.53（等于随机），
    看起来像"sEMG 生理上没信息"，其实是地板造成的。

    这正是 `nightguard-absolute-thresholds-fail` 那条教训的同一个坑：
    **任何绝对量纲的常数在跨记录上都不成立**，包括用来"兜底"的那个。
    地板改为 0.0 —— 完全交给本记录自身的本底噪声定。
    """
    f = highpass(sig_uv, fs, hp_hz)
    a = sorted(abs(v) for v in f)
    if not a:
        return floor_uv
    mad = a[len(a) // 2]
    sd = 1.4826 * mad                  # MAD → SD（高斯假设）
    return max(floor_uv, k * sd)


def zcr_series(sig_uv, cfg: EmgZcrConfig, hyst: float = None):
    """返回 [(窗起始秒, 该窗零交叉数), ...]。sig_uv 必须已是 µV。

    hyst 显式传入时用它（供按 recording 标定），否则用 cfg 里的固定值。
    """
    f = highpass(sig_uv, cfg.fs, cfg.hp_hz)
    if hyst is None:
        hyst = cfg.hysteresis_uv
    w = int(round(cfg.win_s * cfg.fs))
    step = max(1, int(round(w * (1.0 - cfg.overlap))))
    out = []
    for s in range(0, max(0, len(f) - w + 1), step):
        out.append((s / cfg.fs, zero_crossings_hyst(f[s:s + w], hyst)))
    return out


def detect(sig_uv, cfg: EmgZcrConfig = None):
    """连续 n_windows 个窗的零交叉数都超阈 → 一次检出。

    返回 [(报警时刻秒, 触发时的连续窗数)]；报警时刻取第 n_windows 个窗的末尾，
    即原文的"最小延迟"口径。
    """
    cfg = cfg or EmgZcrConfig()
    ser = zcr_series(sig_uv, cfg)
    hits, run = [], 0
    for t, c in ser:
        if c >= cfg.zc_threshold:
            run += 1
            if run == cfg.n_windows:
                hits.append((t + cfg.win_s, run))
        else:
            run = 0
    return hits, ser
