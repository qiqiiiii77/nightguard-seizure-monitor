"""逐秒持续性 + 体位门的单元测试。

重点测两类行为：
  1. 正例/反例能分开（发作式"连成片" vs 伪迹式"短突发"）
  2. **降级路径**：任何数据不足的情形都必须放行，不能漏报。
     这是安全关键 —— 门的默认值错了会静默漏掉发作。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.posture import PostureConfig, posture_gate, tilt_deg  # noqa: E402
from engine.sustain import (SustainConfig, best_vote_frac,  # noqa: E402
                            longest_run_s, sustain_evidence, sustain_keep)

PASS = FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ {msg}")


print("longest_run_s / best_vote_frac")
ok(longest_run_s([True] * 14, 0.5) == 7.0, "14 段 ×0.5s = 7s 连续")
ok(longest_run_s([True, False, True], 0.5) == 0.5, "被打断只算最长那段")
ok(longest_run_s([], 0.5) == 0, "空序列 → 0")
ok(longest_run_s([None, True, True], 0.5) == 1.0, "None 视为中断")
ok(best_vote_frac([True] * 20, SustainConfig()) == 1.0, "全异常 → 投票 1.0")
_v = best_vote_frac([True] * 8 + [False] * 2, SustainConfig())
ok(abs(_v - 0.8) < 1e-9, "8/10 异常 → 投票 0.8（SeizeIT2 规则）")
ok(best_vote_frac([True, True], SustainConfig()) == 1.0,
   "段数不足一个投票窗 → 用全部段占比，不直接判否")
ok(best_vote_frac([None, None], SustainConfig()) is None,
   "全是 None → 返回 None（交由上层降级）")

print("\nsustain_evidence：背景分位门 + 持续性")
bg = list(range(100))                      # 背景 0..99，P90 = 89
ev = sustain_evidence([95] * 20, bg, SustainConfig())
ok(ev is not None and ev["thr"] == 89, "门槛取本记录背景 P90")
ok(ev["run_s"] == 10.0, "20 段全超阈 → 连续 10s")
ok(ev["vote"] == 1.0, "20 段全超阈 → 投票 1.0")
ev2 = sustain_evidence([95, 1, 95, 1, 95, 1, 95, 1], bg, SustainConfig())
ok(ev2["run_s"] == 0.5, "断续高值 → 连不成片")
ok(sustain_evidence([95] * 20, list(range(5))) is None,
   "背景样本 <20 → None（建不出分位）")
ok(sustain_evidence([95], bg) is None, "事件段 <3 → None")

print("\nsustain_keep：降级必须放行（安全关键）")
ok(sustain_keep(None) is True, "证据为 None → 放行，不漏报")
ok(sustain_keep({"run_s": 7.0, "vote": 0.0}) is True, "连续达标 → 放行")
ok(sustain_keep({"run_s": 0.0, "vote": 0.8}) is True, "投票达标 → 放行")
ok(sustain_keep({"run_s": 0.0, "vote": 0.1}) is False, "两者皆不达标 → 否决")
ok(sustain_keep({"run_s": 0.0, "vote": None}) is False,
   "投票无值且连续不够 → 否决")

print("\n体位门：躺 vs 立")
ok(abs(tilt_deg((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) - 0.0) < 1e-6,
   "重力垂直于长轴 → 倾角 0°（躺）")
ok(abs(tilt_deg((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) - 90.0) < 1e-6,
   "重力沿长轴 → 倾角 90°（立）")
cfg = PostureConfig(horiz_max_deg=45.0, horiz_frac_min=0.5)
ok(posture_gate([0.0] * 10, cfg)[0] is True, "整窗躺 → 放行")
ok(posture_gate([80.0] * 10, cfg)[0] is False, "整窗立 → 否决")
ok(posture_gate([0.0] * 6 + [80.0] * 4, cfg)[0] is True,
   "60% 的窗在躺 → 放行（阵挛会污染个别窗）")

print("\n体位门：降级必须放行")
ok(posture_gate([], cfg)[0] is True, "无倾角数据 → 放行")
ok(posture_gate([None, None], cfg)[0] is True, "全 None → 放行")
ok(posture_gate(None, cfg)[0] is True, "传 None → 放行")

print("\n" + "=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
