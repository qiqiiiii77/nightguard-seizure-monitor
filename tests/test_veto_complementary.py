#!/usr/bin/env python3
"""第九棒：否决逻辑的互补通道约束。

这组测试的来历：`probe_motion_burst` 的原版只看心率一路就否决
（`hr - base_hr <= 3.0` → veto）。在 SeizeIT2 上实测，这是一条
**结构性漏报通道**，而既有 104 个测试一个都没覆盖这个函数。

锚点是真实数据，不是构造的极端值：
  sub-083 是 15 人中唯一发作时心率不升的（基线 131.3 → 发作 129.6 bpm），
  他那次真发作的体动是本人基线的 48.6 倍。原版逻辑会否决它。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.gate import probe_motion_burst  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {extra}")


class FakeBaseline:
    def __init__(self, hr=None, motion=None):
        self._c = {"hr": hr, "motion": motion}

    def center(self, k):
        return self._c.get(k)


print("=" * 60)
print("否决逻辑：单通道证据不得独自否决")
print("=" * 60)

# --- 回归测试：sub-083 那一次真发作 --------------------------------
# 心率几乎不动（-1.6bpm），但体动 48.6 倍基线。原版逻辑在这里 veto。
v, why = probe_motion_burst(
    {"hr": 129.6, "motion": 48.6, "rhythm": 0.39},
    [], FakeBaseline(hr=131.3, motion=1.0))
check("sub-083 真发作（心率不升 + 体动48.6×）不被否决",
      v != "veto", f"→ 实得 {v}: {why}")
check("sub-083 这一次应当被加重（体动极端 + 有节律）",
      v == "confirm", f"→ 实得 {v}")

# --- 单通道极端一律免疫 --------------------------------------------
v, _ = probe_motion_burst({"hr": 70.0, "motion": 40.0, "rhythm": 0.05},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("仅体动极端（40×）心率全平 → 不否决", v != "veto", f"→ {v}")

v, _ = probe_motion_burst({"hr": 120.0, "motion": 1.0, "rhythm": 0.05},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("仅心率极端（+50bpm）体动全平 → 不否决", v != "veto", f"→ {v}")

v, _ = probe_motion_burst({"hr": 70.0, "motion": 1.0, "rhythm": 0.60},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("仅节律性极端（0.60）→ 不否决", v != "veto", f"→ {v}")

# --- 真正该否决的：所有通道一致平淡 --------------------------------
v, why = probe_motion_burst({"hr": 71.0, "motion": 2.0, "rhythm": 0.10},
                            [], FakeBaseline(hr=70.0, motion=1.0))
check("翻身（心率+1、体动2×、无节律）→ 否决", v == "veto", f"→ {v}")
check("否决必须给出可追溯理由", bool(why) and "平淡" in (why or ""))

# --- 证据不足时不许否决 --------------------------------------------
v, _ = probe_motion_burst({"hr": None, "motion": 2.0, "rhythm": None},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("只有体动一路可看 → 不否决（证据不足偏向报警）", v != "veto", f"→ {v}")

v, _ = probe_motion_burst({"hr": None, "motion": None, "rhythm": None},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("全部通道缺失 → 不否决", v != "veto", f"→ {v}")

v, _ = probe_motion_burst({"hr": 71.0, "motion": 2.0, "rhythm": 0.10},
                          [], FakeBaseline(hr=None, motion=None))
check("无个人基线 → 不否决（没有比较基准）", v != "veto", f"→ {v}")

# --- 边界：心率略升就不许否决 --------------------------------------
v, _ = probe_motion_burst({"hr": 74.0, "motion": 2.0, "rhythm": 0.10},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("心率已抬升 +4bpm（>3 阈值）→ 不否决", v != "veto", f"→ {v}")

v, _ = probe_motion_burst({"hr": 71.0, "motion": 4.0, "rhythm": 0.10},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("体动 4×基线（>3 阈值）→ 不否决", v != "veto", f"→ {v}")

v, _ = probe_motion_burst({"hr": 71.0, "motion": 2.0, "rhythm": 0.30},
                          [], FakeBaseline(hr=70.0, motion=1.0))
check("存在中等节律性（0.30）→ 不否决", v != "veto", f"→ {v}")

print()
print("=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
