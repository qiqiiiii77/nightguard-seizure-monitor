"""候选事件聚合 —— 把逐窗流式判定攒成"一次事件"。

为什么需要这一层
----------------
`gate.py` 是逐窗的：`EventJournal.record()` 每窗写一行，`start == end`，
没有任何跨窗归并。但验证侧的分级判据是**事件级**的：
"体动连续超阈 ≥20 秒"这种话，只在一段连续时间上才有意义，
单个 2 秒窗上说不出来。

更要紧的是报警语义。家长关心的是"今晚响了几次"，不是"几个窗口越阈"。
一次强直-阵挛持续 1-2 分钟，逐窗算能产出几十个 CRITICAL 窗；
不归并的话误报率的分母和分子都没法和文献对齐
（NightWatch / SzCORE 都按事件计数）。

合并窗取 300 秒，沿用 `validation/run_llm_adjudicator.py:36` 的 MERGE ——
那套逻辑已经在验证里跑过 180 个事件，不另造一个。
依据也和临床一致：`clinical_config.post_ictal_s = 600` 把发作后 10 分钟
视为同一事件的余波，300 秒合并窗落在这个尺度内。

与节流的分工（别混）
-------------------
  · 节流（`gate.py:944-1030`）管"**此刻**要不要响铃"，是实时的，
    被节流的窗仍然进日志、仍然参与判定。
  · 本模块管"**回头看**这些窗属于几次事件"，是离线/准离线的，
    给分级证据和报告用。
两者都用时间窗，但一个作用于通知、一个作用于计数，不要互相替代。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 同一事件内两个报警窗的最大间隔。超过就算两次独立事件。
DEFAULT_MERGE_S = 300.0


@dataclass
class CandidateEvent:
    """一簇连续报警窗归并成的候选事件。"""
    t0: float                       # 首个窗的时间
    t1: float                       # 末个窗的时间
    windows: list = field(default_factory=list)   # 原始窗（保留给证据层）
    risk_peak: float = 0.0
    levels: list = field(default_factory=list)    # 各窗判定的级别

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t1 - self.t0)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def top_level(self) -> str:
        """簇内最高级别。LOG < WARNING < CRITICAL。"""
        rank = {"LOG": 0, "WARNING": 1, "CRITICAL": 2}
        best, best_r = "LOG", -1
        for lv in self.levels:
            r = rank.get(lv, 0)
            if r > best_r:
                best, best_r = lv, r
        return best


def group_windows(windows, merge_s: float = DEFAULT_MERGE_S,
                  time_key="t", risk_key="risk", level_key="level"):
    """把报警窗按时间间隔切成候选事件。

    windows: 可迭代的 dict，至少含时间字段。**不要求已排序** ——
             流式来源偶尔乱序（补包、时钟回拨），这里统一排一次。
    返回 [CandidateEvent]，按时间升序。

    时间字段缺失或非数值的窗会被跳过：没有时间就无法归并，
    而硬塞一个默认时间会把不相关的窗粘成一簇（比漏掉更糟）。
    """
    rows = []
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        t = w.get(time_key)
        if not isinstance(t, (int, float)) or t != t:   # t != t 滤 NaN
            continue
        rows.append((float(t), w))
    if not rows:
        return []
    rows.sort(key=lambda x: x[0])

    out, cur = [], [rows[0]]
    for prev, nxt in zip(rows, rows[1:]):
        if nxt[0] - prev[0] > merge_s:
            out.append(_build(cur, risk_key, level_key))
            cur = []
        cur.append(nxt)
    out.append(_build(cur, risk_key, level_key))
    return out


def _build(rows, risk_key, level_key) -> CandidateEvent:
    ws = [w for _, w in rows]
    risks = [w.get(risk_key) for w in ws]
    risks = [r for r in risks if isinstance(r, (int, float))]
    return CandidateEvent(
        t0=rows[0][0],
        t1=rows[-1][0],
        windows=ws,
        risk_peak=max(risks) if risks else 0.0,
        levels=[w.get(level_key) for w in ws if w.get(level_key)],
    )


def count_events(times, merge_s: float = DEFAULT_MERGE_S) -> int:
    """只数事件个数（不建对象）。等价于 run_llm_adjudicator.merge_count。"""
    ts = sorted(float(t) for t in (times or [])
                if isinstance(t, (int, float)) and t == t)
    if not ts:
        return 0
    n = 1
    for a, b in zip(ts, ts[1:]):
        if b - a > merge_s:
            n += 1
    return n
