"""陀螺仪辅助证据 —— 把 validation/run_r9_gyr_sweep.py 验过的判据接进生产路径。

这一路在干什么
-----------
`stepwise.evaluate()` 只看加速度和心率。**强直期是等长收缩** ——
肌肉绷紧但身体不位移，加速度计几乎看不见，而姿态扭转让角速度看得见。
所以本模块只做一件事：对 stepwise **已经否决**的候选，
问"陀螺仪是否强烈支持"，若是则放回来。

它是**辅助证据，不是独立候选源**
---------------------------
实测过让陀螺仪独立提候选（validation/run_r6d_gyro_gate.py）：
**0/3 捞回、46 次/10夜**，彻底失败。所以这里的接法是
"只对 stepwise 否决过的事件做翻案"，不新增任何候选 ——
和 LLM 裁决、personal_novelty 同一个位置（见
`production-path-missing-validated-modules` 的教训）。

判据与门槛的来源（全部实跑，不是估的）
--------------------------------
· 判据：事件窗内角速度幅值超**本人背景 P95** 的段占比
  （绝对阈值在 SeizeIT2 上必败，电极/佩戴条件差一个数量级，
   一律转本记录背景分位 —— 见 `nightguard-absolute-thresholds-fail`）
· 门槛 0.70：validation/run_r9_gyr_sweep.py 扫了 0.10–0.95。
  救回 2 次致命级（sub-029 占比 0.73、sub-073 占比 0.97），
  代价 6 个误报桶 = 2.68 次/10夜。
· 过拟合检查：按**患者留一** 17 位，16/17 次在训练集上选出的最优门槛
  落在 0.70±0.10 → 门槛稳定。
· 位置无关：耳后与胸部陀螺仪实测等效（run_r15_chest_gyr.py），
  两者均救回 2 次、同样 6 桶代价。**硬件可以只用胸贴。**

代价是不可约的，别再试着调
----------------------
查过被救回的 2 次发作和带进的 6 个误报在
角速度占比 / 体动峰 / 心率峰三个维度上的分布：
**发作全部落在误报分布内部**（sub-029 那次发作体动峰 26.6
还低于它自己的误报 43.1）。没有任何一维能分离。
也试过"报警后持续性"这条时间货币（run_r14_persist.py）：
sub-015 的 f2b 报警后占比 0.02，比所有误报都低 —— 任何自清门槛都会
误撤真发作。所以这 6 桶只能通过**分级提示**消化，不能靠判据消除。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# 与 validation 侧完全一致的常数（改这里必须同步改 run_r9_gyr_sweep.py）
WIN_S = 2.0        # 包络窗长（秒）
HOP_S = 1.0        # 包络步进（秒）
BG_PCT = 95.0      # 背景分位
FRAC_TH = 0.70     # 占比门槛


@dataclass
class GyroConfig:
    win_s: float = WIN_S
    hop_s: float = HOP_S
    bg_pct: float = BG_PCT
    frac_th: float = FRAC_TH
    # 至少要几轴才算有效。单轴无法构成幅值，判据会失真。
    min_axes: int = 2
    # 背景池最少段数。太少则分位不可信，一律按"无数据"处理（fail-open）。
    min_bg_segs: int = 30


def _percentile(xs: Sequence[float], p: float) -> Optional[float]:
    """线性插值分位。不引 numpy —— 生产路径保持零第三方依赖。"""
    v = sorted(x for x in xs if x is not None and x == x)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return v[int(k)]
    return v[lo] * (hi - k) + v[hi] * (k - lo)


def envelope(axes: Sequence[Sequence[float]], fs: float,
             cfg: GyroConfig = None) -> list:
    """多轴角速度 → 幅值的 RMS 包络序列。

    axes: 每轴一个采样序列（长度可不等，按最短截断）。
    返回每个包络窗的 RMS 值；数据不足返回 []。
    """
    cfg = cfg or GyroConfig()
    good = [list(a) for a in axes if a is not None and len(a) > 0]
    if len(good) < cfg.min_axes:
        return []
    n = min(len(a) for a in good)
    if n < fs * 10:          # 不足 10 秒，判据不成立
        return []
    mag = [math.sqrt(sum(a[i] ** 2 for a in good)) for i in range(n)]
    w, h = int(cfg.win_s * fs), int(cfg.hop_s * fs)
    if w <= 0 or h <= 0 or n < w:
        return []
    out = []
    for s in range(0, n - w + 1, h):
        seg = mag[s:s + w]
        out.append(math.sqrt(sum(x * x for x in seg) / len(seg)))
    return out


def baseline_threshold(bg_axes: Sequence[Sequence[float]], fs: float,
                       cfg: GyroConfig = None) -> Optional[float]:
    """本人背景角速度的 P95。取不到返回 None（调用方须 fail-open）。"""
    cfg = cfg or GyroConfig()
    env = envelope(bg_axes, fs, cfg)
    if len(env) < cfg.min_bg_segs:
        return None
    return _percentile(env, cfg.bg_pct)


def gyro_fraction(event_axes: Sequence[Sequence[float]], fs: float,
                  bg_threshold: Optional[float],
                  cfg: GyroConfig = None) -> Optional[float]:
    """事件窗内超背景 P95 的段占比。无法计算返回 None。"""
    cfg = cfg or GyroConfig()
    if bg_threshold is None:
        return None
    env = envelope(event_axes, fs, cfg)
    if not env:
        return None
    return sum(1 for x in env if x >= bg_threshold) / len(env)


def supports(fraction: Optional[float], cfg: GyroConfig = None) -> bool:
    """陀螺仪是否强烈支持"这是发作"。

    **None → False**：这里不 fail-open，因为本模块只用于**翻案**
    （把 stepwise 否决的放回来）。取不到数据时维持 stepwise 的原判，
    而不是凭空放行 —— fail-open 的位置在 stepwise/watchdog 那一层。
    """
    cfg = cfg or GyroConfig()
    return fraction is not None and fraction >= cfg.frac_th


def rescue(stepwise_keep: bool, fraction: Optional[float],
           cfg: GyroConfig = None) -> dict:
    """对单个候选给出最终去留与原因。

    返回 {keep, tier_hint, reason, fraction}
    · tier_hint="T3"：仅靠陀螺仪单路支撑 → 响铃但标"待确认"
      （不是静默档。实测过静默档会藏住 26/29 次致命级，
       见 `tiered-alarm-hides-missed-seizures`，已废弃该设计。）
    """
    cfg = cfg or GyroConfig()
    if stepwise_keep:
        return {"keep": True, "tier_hint": None,
                "reason": "stepwise 已通过，未用陀螺仪翻案",
                "fraction": fraction}
    if supports(fraction, cfg):
        return {"keep": True, "tier_hint": "T3",
                "reason": (f"stepwise 否决，但角速度超本人背景 P{cfg.bg_pct:.0f} "
                           f"的占比 {fraction:.2f} ≥ {cfg.frac_th:.2f} → 翻案"),
                "fraction": fraction}
    return {"keep": False, "tier_hint": None,
            "reason": ("角速度不支持" if fraction is not None
                       else "无陀螺仪数据，维持 stepwise 原判"),
            "fraction": fraction}
