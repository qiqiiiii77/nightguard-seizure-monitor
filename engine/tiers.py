"""报警分级 —— 分级只改变"怎么响"，绝不改变"是否响"。

这个模块存在的唯一前提
-----------------
2026-10-08 实测过一版"高档响铃 / 中低档静默"的分级
（记忆 `tiered-alarm-hides-missed-seizures`）：

    档位      会叫醒?   发作数   其中致命级
    高风险      是        3         3
    中风险     **否**     7         7
    低风险     **否**    13        11
    未覆盖     **否**     8         8

**26/29 次致命级落在不响铃档，响铃档只覆盖 10%。**
典型是 sub-029 的 189 秒 f2b 判进中风险、sub-073 的 144 秒 f2b 判进低风险
—— 都是会导致 SUDEP 的类型，家长当晚不会被叫醒。
那是用漏报换误报，在安全关键场景里不能当主方案。

**所以本模块的设计约束是硬的：没有静默档。**
`SILENT_TIERS` 恒为空集，`rings()` 对所有合法档位恒返回 True。
分级只决定提示强度/话术，不决定是否响。
任何给本模块加"静默档"的改动，必须先推翻上面那组实测数据。

分级还有什么用
-----------
最终工作点是 6.70 次/10夜 ≈ 每晚 0.67 次（validation/run_r10_final.py）。
产品问题不是"响太多次"，而是"每次都用最高强度叫醒，家长无法分辨轻重"。
分级把低置信那部分降为轻提示，真发作仍 100% 走强提醒。

实测分布（validation/run_r13_tiers.py）
------------------------------------
    档位   响铃?  致命级  非致命  误报桶  次/10夜
    T1      是     18      0       5     2.23
    T2      是      3      0       4     1.79
    T3      是      2      0       6     2.68
    不响铃档 —       0      0       0     0.00   ← 结构上不存在

    23/23 致命级全部落在响铃档内；T1 单独覆盖 18/23 = 78%。

关于 T2 "自动撤回" —— 已验证不可行，别再实现
--------------------------------------
曾设计 T2 为"先响轻提示，若 30s 内证据未持续则自动撤回"。
实测否决（validation/run_r14_persist.py）：报警后 120 秒内
角速度持续占比，发作 sub-015 的 f2b 只有 **0.02**，
比全部 10 个误报桶中的多数都低；sub-058 是 0.19。
扫 0.05–0.50 全部门槛，**任何能撤回误报的门槛都会误撤致命级发作**。
→ T2 保留"响"，去掉"自清"。这是"时间"这条货币在本数据上的否决结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

T1 = "T1"      # 多路证据一致 → 立即强提醒
T2 = "T2"      # 证据较弱（含缺通道降级评估）→ 响铃，轻提示
T3 = "T3"      # 仅单路（陀螺仪）支撑 → 响铃，标"待确认"

ALL_TIERS = (T1, T2, T3)

# 恒空。见模块文档：静默档会藏住致命级发作。
SILENT_TIERS: frozenset = frozenset()

PROMPT = {
    T1: "强提醒：多路证据一致，疑似全身性发作，请立即查看。",
    T2: "提醒：检出疑似发作，部分传感器证据不足，请查看。",
    T3: "待确认：角速度检出疑似发作体征，其余证据不足，请查看。",
}


@dataclass
class TierConfig:
    # 判 T1 需要几路证据支持。实测 ≥2 时 T1 覆盖 18/23 致命级。
    t1_min_support: int = 2
    # 各路的"算支持"门槛，与 validation/run_r13_tiers.py 一致。
    eeg_vote_th: float = 0.8      # ll / zc 是投票比例
    tilt_th: float = 0.5          # ta
    pct_th: float = 0.80          # mot / hr 是分位


def rings(tier: str) -> bool:
    """该档位是否会叫醒家长。恒为 True —— 没有静默档。"""
    if tier not in ALL_TIERS:
        raise ValueError(f"未知档位 {tier!r}；合法值 {ALL_TIERS}")
    return tier not in SILENT_TIERS


def count_support(*, ll=None, zc=None, ta=None,
                  mot_pct=None, hr_pct=None,
                  cfg: TierConfig = None) -> int:
    """几路证据支持。None 一律不计入（缺数据不是支持，也不是反对）。"""
    cfg = cfg or TierConfig()
    n = 0
    for v, th in ((ll, cfg.eeg_vote_th), (zc, cfg.eeg_vote_th),
                  (ta, cfg.tilt_th)):
        if v is not None and v >= th:
            n += 1
    for v in (mot_pct, hr_pct):
        if v is not None and v >= cfg.pct_th:
            n += 1
    return n


def assign(*, support: int, degraded: bool = False,
           gyro_only: bool = False, cfg: TierConfig = None) -> str:
    """定档。

    gyro_only: 该候选是被陀螺仪翻案救回的（stepwise 原本否决）→ T3
    degraded : 该 recording 缺通道、走降级评估 → T2
    """
    cfg = cfg or TierConfig()
    if gyro_only:
        return T3
    if degraded:
        return T2
    return T1 if support >= cfg.t1_min_support else T2


def decide(*, support: int = 0, degraded: bool = False,
           gyro_only: bool = False, cfg: TierConfig = None) -> dict:
    """一步给出档位 + 是否响铃 + 给家长看的话术。"""
    tier = assign(support=support, degraded=degraded,
                  gyro_only=gyro_only, cfg=cfg)
    return {"tier": tier, "rings": rings(tier), "prompt": PROMPT[tier]}


def audit(events: list) -> dict:
    """安全自检：统计各档的致命级数量，并断言不响铃档里没有发作。

    events: [{tier, is_fatal(bool)}, ...]
    任何分级方案上线前必须跑这个 —— 记忆里的硬性判据是
    "必须同时报不响铃档的致命级事件数，>0 则方案只能当次选"。
    """
    per = {t: {"fatal": 0, "other": 0} for t in ALL_TIERS}
    silent_fatal = 0
    for e in events:
        t = e.get("tier")
        if t not in per:
            continue
        per[t]["fatal" if e.get("is_fatal") else "other"] += 1
        if not rings(t) and e.get("is_fatal"):
            silent_fatal += 1
    return {"per_tier": per, "silent_fatal": silent_fatal,
            "safe": silent_fatal == 0}
