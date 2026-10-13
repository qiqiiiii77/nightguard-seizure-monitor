"""事件级分级证据 —— 把验证过的判据搬成生产可调用的函数。

这组判据的来历（都是实测，不是设计意图）
----------------------------------------
嵌套 LOO（六个参数全部只在训练折上选，无任何泄漏）：
**22/22 零丢弃，CRITICAL 0.019/h（一夜 0.15），WARNING 0.037/h**，
18/18 折选中同一组参数，且对代价常量 MISS∈[20,1000]、DEMOTE∈[1,20]
完全不变。

为什么要分级 —— 单级阈值下"零漏报"和"低误报"是矛盾的
-----------------------------------------------------
单级架构的嵌套 LOO 只有 21/22。掉的是 sub-015（体动仅 1.0s）：
它那折的训练集里**没有任何需要 EEG 救援的发作**，于是代价最小化
必然把救援通路关掉，然后在测试折上漏掉它。
这不是调参问题 —— **对罕见失效模式的兜底路，无法在缺少该模式的
折上被验证**，是结构性的。

分级把它解决了：弱证据的默认去向是 WARNING（静默留痕），
**不是丢弃**。代价函数再也关不掉兜底路，因为兜底不需要被"选中"。

重叠区 EEG 佐证（误报再降 44%）
------------------------------
体动 run_s 在 20-30s 这段，发作和伪迹重叠：
  发作 4 次，EEG 投票全部 1.00
  误报 7 个，其中 4 个 EEG ≤0.35（sub-024 那几个翻身）
所以这段要求必须有 EEG 佐证才响铃，否则降 WARNING。
实测误报 9→5 个，仍 22/22，且对 R_HI∈[30,45]、CORR∈[0.5,1.0] 全平 ——
不是卡在悬崖边上的取值。

降级原则（安全关键）
-------------------
任何"算不出"一律往**保留**方向走：所有通路都无数据时判 WARNING，
而不是不报。`posture.py` 本轮修过一个 `posture_gate(None)` 抛
TypeError 的缺陷 —— 安全门抛异常等于静默漏报，这里同样标准。
"""

from __future__ import annotations

from dataclasses import dataclass

CRITICAL = "CRITICAL"
WARNING = "WARNING"
NONE = "NONE"


@dataclass(frozen=True)
class EvidenceConfig:
    """嵌套 LOO 选中的那一组。改这些数之前先重跑 audit_leakage.py。"""
    # 体动持续性：>=run_hi_s 无条件够硬；[run_lo_s, run_hi_s) 需 EEG 佐证。
    run_hi_s: float = 30.0
    run_lo_s: float = 20.0
    # 重叠区的 EEG 佐证门槛。
    corroborate_vote: float = 0.80
    # EEG 救援：体动低于 rescue_max_s 时才允许单靠 EEG 进 WARNING。
    # 限定"主通路看不见"才启用是实测结论 —— 无条件 OR 时误报从
    # 0.148 **涨到** 0.191/h（耳后 EEG 对运动伪迹同样敏感）。
    rescue_max_s: float = 5.0
    rescue_vote: float = 0.80


@dataclass
class EvidenceVerdict:
    level: str                 # CRITICAL | WARNING | NONE
    reason: str                # 给家长/报告看的人话
    run_s: float = None        # 体动连续秒数（None=算不出）
    vote: float = None         # EEG 投票占比（None=算不出）
    degraded: bool = False     # 是否因数据不足而降级保留


def classify(run_s, vote, cfg: EvidenceConfig = None) -> EvidenceVerdict:
    """事件级分级。

    run_s: 体动逐秒超阈的最长连续秒数（`sustain.sustain_evidence` 的
           "run_s" 字段）。None 表示算不出（无体动数据/背景样本不足）。
    vote:  EEG 逐秒异常的滑窗投票占比（`sustain.best_vote_frac`）。
           None 表示算不出。

    两者都算不出 → WARNING + degraded=True。**绝不返回 NONE** ——
    "我看不见"和"我确认没事"是两件事，不能当成一件。
    """
    cfg = cfg or EvidenceConfig()
    has_run = isinstance(run_s, (int, float)) and run_s == run_s
    has_vote = isinstance(vote, (int, float)) and vote == vote

    if not has_run and not has_vote:
        return EvidenceVerdict(WARNING, "数据不足，无法判别，保留待回看",
                               None, None, degraded=True)

    if has_run:
        if run_s >= cfg.run_hi_s:
            return EvidenceVerdict(
                CRITICAL, f"体动持续 {run_s:.0f}s，超过 {cfg.run_hi_s:.0f}s",
                run_s, vote if has_vote else None)
        if run_s >= cfg.run_lo_s:
            # 重叠区：发作与翻身伪迹在体动上分不开，要 EEG 投票佐证。
            if not has_vote:
                # 没有 EEG 可查 → 降级**保留为 CRITICAL**。
                # 这里刻意不降 WARNING：体动已达 20s 是强证据，
                # 不能因为"另一路没数据"就把它压成静默。
                return EvidenceVerdict(
                    CRITICAL,
                    f"体动持续 {run_s:.0f}s，无 EEG 可佐证，按高置信处理",
                    run_s, None, degraded=True)
            if vote >= cfg.corroborate_vote:
                return EvidenceVerdict(
                    CRITICAL,
                    f"体动持续 {run_s:.0f}s + EEG 佐证 {vote:.2f}",
                    run_s, vote)
            return EvidenceVerdict(
                WARNING,
                f"体动持续 {run_s:.0f}s 但 EEG 仅 {vote:.2f}，疑似体动伪迹",
                run_s, vote)

    # 体动不达标（或无体动数据）：只剩 EEG 救援这条路，且只进 WARNING。
    # 为什么不进 CRITICAL：纯 EEG 证据在本数据集上会带进
    # sub-109 那类 vote=1.00 的伪迹，不足以单独叫醒家长。
    low_motion = (not has_run) or run_s < cfg.rescue_max_s
    if low_motion and has_vote and vote >= cfg.rescue_vote:
        return EvidenceVerdict(
            WARNING, f"体动不足但 EEG 异常 {vote:.2f}，保留待回看",
            run_s if has_run else None, vote)

    return EvidenceVerdict(NONE, "证据不足", run_s if has_run else None,
                           vote if has_vote else None)
