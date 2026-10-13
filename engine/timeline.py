"""夜间事件时间轴 —— 赛题功能需求 3 的实现。

赛题原文（`solutions` 需求 3「夜间事件自动记录时间轴」）：
    自动留存每一次可疑事件时间点、持续时长、体征变化特征、
    日/周发作趋势汇总 —— 解决多人照护信息不对称，
    可作复诊调药与病情复盘参考。

为什么单独做一层
-------------
已有两层都**不是**时间轴：
· `gate.EventJournal` 是**逐窗**日志（每 2 秒一行，`start == end`），
  一次 2 分钟的发作会写成几十行。给算法追溯用，不是给家长看的。
· `events.group_windows()` 把窗归并成事件，但只留 `risk_peak` 和级别，
  **没有体征特征、没有趋势汇总、不能落盘**。

本模块把两者接起来：窗 → 事件 → 一条人能读的时间轴 + 日/周趋势。

设计约束（都来自已验证的结论，不是新发明）
----------------------------------
1. **归并窗 300 秒**，直接用 `events.DEFAULT_MERGE_S`，不另造一个。
   理由见 `events.py` 开头：和 `run_llm_adjudicator.py` 跑过 180 个事件的
   那套一致，也落在 `clinical_config.post_ictal_s = 600` 的尺度内。
2. **所有事件都记录，不筛级别**。赛题技术可行性一节明确要求
   非惊厥发作（肌阵挛、失神）"仅作记录完善癫痫日记"——
   不报警但必须留档。所以时间轴收 LOG 级，报警层不收。
3. **时长 ≥2 分钟单独标记**。赛题需求 2 的硬性规则：
   "持续发作超过 2 分钟必须提醒家长"。`clinical.py:16` 已实现报警侧，
   这里补档案侧的标记，便于复诊时一眼看到持续状态风险。
4. **体征特征存"相对本人基线"的量**，不存绝对值。
   绝对阈值在 SeizeIT2 上必败（整个项目的核心教训），
   给医生看的也应该是"比平时高 38bpm"而不是"118bpm"。

隐私边界（赛题 `solutions/README.md` 的红线）
----------------------------------------
提交物不得含患者姓名、电话、病历、影像。本模块只接受
**匿名 ID**（如 `child-01`），落盘字段里没有任何自由文本姓名位，
导出时间用相对秒数 + 可选的夜次编号，不写真实日期。
调用方若硬塞姓名进 `subject_id`，`export_json()` 不会阻止 ——
这一条靠流程保证，代码只做到"不提供姓名字段"。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .events import DEFAULT_MERGE_S, group_windows

# 赛题需求 2：持续 >2 分钟必须提醒（这里用于档案标记）
LONG_SEIZURE_S = 120.0
# 赛题需求 2：发作后 5-10 分钟高风险期。取上界，与
# clinical_config.post_ictal_s = 600 一致。
POST_ICTAL_S = 600.0


@dataclass
class VitalSignature:
    """一次事件的体征变化特征（全部相对本人基线）。"""
    hr_rise_bpm: Optional[float] = None      # 心率比基线高多少
    hr_peak_ratio: Optional[float] = None    # 心率峰 / 基线中位
    motion_peak: Optional[float] = None      # 体动峰值（设备单位）
    gyro_fraction: Optional[float] = None    # 角速度超背景 P95 的占比
    brady_held_s: Optional[float] = None     # 心动过缓持续秒数
    channels_missing: list = field(default_factory=list)   # 本次缺哪些通道

    def to_text(self) -> str:
        """给家长/医生看的一句话。缺值就不提，不编。"""
        bits = []
        if self.hr_rise_bpm is not None:
            bits.append(f"心率比平时高 {self.hr_rise_bpm:.0f}")
        if self.brady_held_s:
            bits.append(f"心动过缓持续 {self.brady_held_s:.0f} 秒")
        if self.motion_peak is not None:
            bits.append(f"体动峰值 {self.motion_peak:.1f}")
        if self.gyro_fraction is not None:
            bits.append(f"身体扭转占比 {self.gyro_fraction:.0%}")
        if self.channels_missing:
            bits.append(f"缺通道：{'/'.join(self.channels_missing)}")
        return "；".join(bits) if bits else "无可用体征特征"


@dataclass
class TimelineEntry:
    """时间轴上的一条。"""
    t0: float
    t1: float
    level: str                     # CRITICAL / WARNING / LOG
    tier: Optional[str] = None     # T1 / T2 / T3（来自 tiers.py）
    rang: bool = False             # 是否真的响了铃
    risk_peak: float = 0.0
    n_windows: int = 0
    vitals: VitalSignature = field(default_factory=VitalSignature)
    note: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t1 - self.t0)

    @property
    def is_long(self) -> bool:
        """≥2 分钟 —— 赛题要求必须提醒的持续状态风险。"""
        return self.duration_s >= LONG_SEIZURE_S

    @property
    def post_ictal_until(self) -> float:
        """发作后高风险期的截止时刻（赛题需求 2：5-10 分钟）。"""
        return self.t1 + POST_ICTAL_S

    def to_row(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.duration_s, 1)
        d["is_long"] = self.is_long
        d["post_ictal_until"] = self.post_ictal_until
        d["vitals_text"] = self.vitals.to_text()
        return d


class NightTimeline:
    """一夜的事件时间轴。

    用法：
        tl = NightTimeline(subject_id="child-01", night=3)
        tl.build_from_windows(journal.events)      # gate.EventJournal.events
        tl.to_text()                               # 给家长看
        tl.export_json(path)                       # 落盘归档
    """

    def __init__(self, subject_id: str, night: Optional[int] = None,
                 merge_s: float = DEFAULT_MERGE_S):
        self.subject_id = subject_id        # 匿名 ID，不要放姓名
        self.night = night                  # 第几夜（相对编号，不写真实日期）
        self.merge_s = merge_s
        self.entries: list = []

    # ---------------------------------------------------------- 构建

    def build_from_windows(self, windows, tier_fn=None, vital_fn=None):
        """把逐窗日志归并成时间轴。

        windows : `gate.EventJournal.events` 那种 dict 列表，
                  需含 start/end/level/risk
        tier_fn : 可选，`entry -> (tier, rings)`，接 tiers.decide() 的结果
        vital_fn: 可选，`CandidateEvent -> VitalSignature`
        """
        # 时间轴收**所有**级别（含 LOG）—— 赛题要求非惊厥发作也要留档
        rows = [w for w in windows if w.get("level") in
                ("CRITICAL", "WARNING", "LOG")]
        # `gate.EventJournal` 写的是 start/end，而 `group_windows` 默认按 "t"
        # 取时间 —— 不传 time_key 会一条都匹配不上（第一版就是这么空的）。
        for ev in group_windows(rows, merge_s=self.merge_s,
                                time_key="start"):
            # `CandidateEvent.t1` 是**末窗的起点**。时长要算到末窗的终点，
            # 否则每次事件都短一个窗（2 秒），跨过 120 秒门槛时会误判。
            ends = [w.get("end") for w in ev.windows
                    if isinstance(w.get("end"), (int, float))]
            t1 = max([ev.t1] + ends) if ends else ev.t1
            e = TimelineEntry(
                t0=ev.t0, t1=t1, level=ev.top_level,
                risk_peak=ev.risk_peak, n_windows=ev.n_windows,
                vitals=(vital_fn(ev) if vital_fn else VitalSignature()),
            )
            if tier_fn:
                got = tier_fn(e)
                if got:
                    e.tier, e.rang = got[0], bool(got[1])
            if e.is_long:
                e.note = "持续 ≥2 分钟，疑似持续状态 —— 赛题要求必须提醒"
            self.entries.append(e)
        self.entries.sort(key=lambda x: x.t0)
        return self.entries

    def add(self, entry: TimelineEntry):
        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.t0)

    # ---------------------------------------------------------- 汇总

    def summary(self) -> dict:
        """一夜汇总。"""
        n_long = sum(1 for e in self.entries if e.is_long)
        rang = sum(1 for e in self.entries if e.rang)
        per_tier = {}
        for e in self.entries:
            if e.tier:
                per_tier[e.tier] = per_tier.get(e.tier, 0) + 1
        durs = [e.duration_s for e in self.entries]
        return {
            "subject_id": self.subject_id,
            "night": self.night,
            "n_events": len(self.entries),
            "n_critical": sum(1 for e in self.entries
                              if e.level == "CRITICAL"),
            "n_warning": sum(1 for e in self.entries
                             if e.level == "WARNING"),
            "n_log_only": sum(1 for e in self.entries if e.level == "LOG"),
            "n_long_ge_2min": n_long,
            "n_rang": rang,
            "per_tier": per_tier,
            "total_duration_s": round(sum(durs), 1),
            "longest_s": round(max(durs), 1) if durs else 0.0,
        }

    # ---------------------------------------------------------- 输出

    def to_text(self) -> str:
        """给家长看的时间轴。时间用 h:mm:ss 相对值，不写真实日期。"""
        def hms(s):
            s = int(s)
            return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

        head = f"【{self.subject_id}"
        if self.night is not None:
            head += f" 第 {self.night} 夜"
        head += "】夜间事件时间轴"
        out = [head, "=" * len(head) * 2]
        if not self.entries:
            out.append("本夜无可疑事件。")
            return "\n".join(out)
        for i, e in enumerate(self.entries, 1):
            mark = " ⚠ 持续≥2分钟" if e.is_long else ""
            bell = "已响铃" if e.rang else "未响铃"
            tier = f" {e.tier}" if e.tier else ""
            out.append(
                f"{i}. {hms(e.t0)} - {hms(e.t1)}（{e.duration_s:.0f} 秒）"
                f" {e.level}{tier} {bell}{mark}")
            out.append(f"   体征：{e.vitals.to_text()}")
            if e.note:
                out.append(f"   备注：{e.note}")
        s = self.summary()
        out += ["", f"合计 {s['n_events']} 次事件"
                    f"（响铃 {s['n_rang']} 次、仅记录 {s['n_log_only']} 次）"
                    f"，最长 {s['longest_s']:.0f} 秒"]
        if s["n_long_ge_2min"]:
            out.append(f"其中 {s['n_long_ge_2min']} 次持续 ≥2 分钟，"
                       f"复诊时请重点说明。")
        return "\n".join(out)

    def export_json(self, path: Optional[str] = None) -> dict:
        """落盘归档。字段里**没有姓名位** —— 见模块文档的隐私边界。"""
        doc = {"summary": self.summary(),
               "entries": [e.to_row() for e in self.entries]}
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        return doc


    def audit_consistency(self) -> dict:
        """报出"仅记录级却响了铃"这类不一致，**不自动改判**。

        为什么只报不改
        -----------
        这里有一条真实的需求冲突，必须让人来裁决，代码不能替人决定：

        · **赛题技术可行性一节**：肌阵挛、失神等非惊厥发作
          "通常无心率血氧变化、伤害较小，仅作记录完善癫痫日记" → 不报警。
        · **本项目 2026-10-08 实测**：把事件判进不响铃档，
          结果 **26/29 次致命级发作落在不响铃档**，响铃档只覆盖 10%。
          所以 `tiers.py` 的 `SILENT_TIERS` 恒空、`rings()` 恒 True。

        两者不矛盾的前提是：**LOG 级必须真的是非惊厥发作**。
        一旦检出层把一次致命级发作误判成 LOG，"仅记录"就等于漏报。
        所以本方法只负责把这种组合**暴露出来**：
        若某条 LOG 事件时长 ≥2 分钟（赛题认定必须提醒的尺度），
        它几乎不可能是无害的肌阵挛 —— 列为 `suspicious`，要求人工复核。

        自动静默这条路我们不走，理由就是上面那组 26/29。
        """
        rang_log = [e for e in self.entries
                    if e.level == "LOG" and e.rang]
        # LOG 级但持续 ≥2 分钟 —— 赛题自己的硬性规则说这种必须提醒，
        # 和"仅记录"冲突，必须人工看
        suspicious = [e for e in self.entries
                      if e.level == "LOG" and e.is_long]
        return {
            "log_but_rang": len(rang_log),
            "log_but_long": len(suspicious),
            "suspicious_t0": [e.t0 for e in suspicious],
            "consistent": not suspicious,
            "note": ("LOG 级事件持续 ≥2 分钟，与赛题「非惊厥仅记录」冲突，"
                     "需人工复核是否误判"
                     if suspicious else "无冲突"),
        }


def weekly_trend(nights: list) -> dict:
    """日/周趋势汇总 —— 赛题需求 3 的后半句。

    nights: [NightTimeline, ...]（或它们的 summary() dict）
    给复诊调药用：每夜事件数、响铃数、最长时长的逐夜序列 + 均值。
    """
    sums = [n.summary() if isinstance(n, NightTimeline) else n
            for n in nights]
    if not sums:
        return {"n_nights": 0}
    ev = [s["n_events"] for s in sums]
    rg = [s["n_rang"] for s in sums]
    lg = [s["n_long_ge_2min"] for s in sums]
    return {
        "n_nights": len(sums),
        "events_per_night": ev,
        "rang_per_night": rg,
        "long_per_night": lg,
        "mean_events": round(sum(ev) / len(ev), 2),
        "mean_rang": round(sum(rg) / len(rg), 2),
        "total_long_ge_2min": sum(lg),
        "worst_night": max(range(len(ev)), key=lambda i: ev[i]) + 1,
    }
