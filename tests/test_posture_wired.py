"""体位门**在生产路径上真的通电**的回归测试。

为什么要这个测试
----------------
`engine/posture.py` 一直写得很完整，但从未接到生产路径：
`gate.py` / `signals.py` 里一个 `tilt` 字段都没有，它只活在
`validation/run_*_arch.py` 这些离线脚本里。于是竞品靠它降 5 倍误报
（Lazeron 2022），我们却关着这条路，还据此得出"传感器不够"的结论。

这和证据层死掉的方式一模一样（读 `features["sustain_run_s"]`，
而流式路径没人写该字段 → 永远 None → 零降级）。

所以本测试不测 `posture_gate()` 这个函数本身（那是 test_sustain_posture.py
的事），而是测**经过 NightGate.step() 这条真实路径**后级别有没有变。
断言点选 `step()` 的返回值和 `posture_demotions` 计数器 ——
只有接线真的在环上才会变。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.gate import GateConfig, NightGate      # noqa: E402
from engine.signals import wearable_signals        # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run(tilt, enabled=True, n_quiet=40, n_big=8):
    """喂安静窗建基线，再喂必然 CRITICAL 的大窗，返回(最后结果, gate)。"""
    g = NightGate(GateConfig(posture_enabled=enabled),
                  signal_fn=wearable_signals)
    for i in range(n_quiet):
        g.step({"t": i * 2, "motion": 0.5, "motion_psd": 0.1,
                "hr": 70, "tilt": tilt})
    r = None
    for k in range(n_big):
        r = g.step({"t": 100 + k * 2, "motion": 30.0, "motion_psd": 0.9,
                    "hr": 150, "tilt": tilt})
    return r, g


print("体位门接线（经 NightGate.step 真实路径）")

r, g = run(tilt=10.0)
check(r.level == "CRITICAL" and g.posture_demotions == 0,
      f"躺着(10°) 保持 CRITICAL，不降级（实得 {r.level}, "
      f"降级 {g.posture_demotions}）")

r, g = run(tilt=89.0)
check(r.level == "WARNING" and r.notify is False and g.posture_demotions > 0,
      f"立着(89°) 降为静默 WARNING（实得 {r.level}, notify={r.notify}, "
      f"降级 {g.posture_demotions}）")
check("体位门" in (r.note or ""), f"降级原因写进 note（实得 {r.note!r}）")

# 半躺：靠枕头坐起来一点仍可能发作，不能漏
r, g = run(tilt=40.0)
check(r.level == "CRITICAL",
      f"半躺(40°, <45° 门限) 仍算躺，保持 CRITICAL（实得 {r.level}）")

print("\n降级通道安全边界")

# 缺字段 → 必须放行。看不见不等于没事。
r, g = run(tilt=None)
check(r.level == "CRITICAL" and g.posture_demotions == 0,
      f"tilt 缺失 → 放行，不开漏报通道（实得 {r.level}）")

# 开关默认关 = 行为不变
r_off, g_off = run(tilt=89.0, enabled=False)
check(r_off.level == "CRITICAL" and g_off.posture_demotions == 0,
      f"posture_enabled=False 时立着也不降级（实得 {r_off.level}）")
check(GateConfig().posture_enabled is False,
      "GateConfig 默认 posture_enabled=False（默认行为不变）")

# 体位门只能降到 WARNING，绝不能把事件丢成 LOG/NONE
r, g = run(tilt=89.0)
check(r.level != "LOG" and r.level != "NONE",
      f"体位门不把事件丢掉，只降到 WARNING（实得 {r.level}）")

print("\n" + ("失败 %d 项" % len(FAIL) if FAIL else "全部通过"))
sys.exit(1 if FAIL else 0)
