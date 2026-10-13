#!/usr/bin/env python3
"""第十七棒：陀螺仪辅助证据 + 报警分级的语义约束。

这组测试锁住的**不是**误报率或灵敏度（那是 validation 脚本的事），
而是三条用实跑换来的**语义边界**。每一条都曾经被试错过：

1. 陀螺仪只做**翻案**，不新增候选
   —— 独立提候选实测 0/3 捞回、46 次/10夜（run_r6d_gyro_gate.py）
2. **没有静默档**
   —— 静默档实测藏住 26/29 次致命级（2026-10-08）
3. T2 **没有自动撤回**
   —— 任何自清门槛都会误撤真发作（run_r14_persist.py）

一旦有人把这三条里任何一条"顺手"改掉，漏报通道就又开了一条。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.gyro_evidence import (GyroConfig, baseline_threshold,  # noqa: E402
                                  envelope, gyro_fraction, rescue, supports)
from engine.schema import Channels, Window                        # noqa: E402
from engine.tiers import (ALL_TIERS, SILENT_TIERS, T1, T2, T3,    # noqa: E402
                          assign, audit, count_support, decide, rings)

PASS, FAIL = 0, 0
FS = 25.0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def _const(v, secs, fs=FS):
    return [v] * int(secs * fs)


def _noise(amp, secs, fs=FS, seed=1):
    out, x = [], seed
    for _ in range(int(secs * fs)):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(amp * (x / (1 << 31) - 0.5))
    return out


print("== 包络计算 ==")
check("单轴构不成幅值，返回空（不许悄悄按单轴算）",
      envelope([_const(1.0, 30)], FS) == []
      and envelope([_const(1.0, 30)] * 2, FS) != [])
check("不足 10 秒判据不成立", envelope([_const(1.0, 5)] * 3, FS) == [])
_env = envelope([_const(1.0, 30)] * 3, FS)
check("三轴恒定 1.0 → 每段 RMS = sqrt(3)",
      _env and all(abs(v - math.sqrt(3)) < 1e-9 for v in _env),
      f"→ 首段 {_env[0] if _env else None}")
check("30s / 窗2s / 步进1s → 29 段", len(_env) == 29, f"→ {len(_env)}")

print()
print("== 背景分位（绝对阈值在 SeizeIT2 上必败，一律转本人背景分位）==")
check("背景池太短 → None",
      baseline_threshold([_const(1.0, 15)] * 3, FS) is None)
check("背景池足够 → 有阈值",
      baseline_threshold([_noise(1.0, 120)] * 3, FS) is not None)
_th_hi = baseline_threshold([_noise(0.1, 300)] * 3, FS)
check("事件远高于背景 → 占比 ≈ 1",
      (gyro_fraction([_const(5.0, 60)] * 3, FS, _th_hi) or 0) > 0.99)
_th_same = baseline_threshold([_noise(1.0, 600)] * 3, FS)
_f_same = gyro_fraction([_noise(1.0, 120, seed=99)] * 3, FS, _th_same)
check("事件与背景同量级 → 占比低（P95 定义决定约 5%）",
      _f_same is not None and _f_same < 0.30, f"→ {_f_same}")
check("无背景阈值 → None",
      gyro_fraction([_const(5.0, 60)] * 3, FS, None) is None)

print()
print("== 翻案语义（约束 1：只翻案，不新增候选）==")
check("门槛 0.70（来自 run_r9_gyr_sweep.py 扫描 0.10-0.95）",
      supports(0.70) and supports(0.97) and not supports(0.69))
check("实测救回的两次致命级都能翻案（占比 0.73 / 0.97）",
      all(rescue(False, f)["keep"] and rescue(False, f)["tier_hint"] == T3
          for f in (0.73, 0.97)))
check("无陀螺仪数据 → 维持 stepwise 原判，不凭空放行",
      rescue(False, None)["keep"] is False
      and rescue(True, None)["keep"] is True)
_r = rescue(True, 0.01)
check("stepwise 已通过的，陀螺仪不参与也不改档",
      _r["keep"] is True and _r["tier_hint"] is None)

print()
print("== 分级（约束 2：没有静默档）==")
check("SILENT_TIERS 恒空，所有档位都响铃",
      SILENT_TIERS == frozenset() and all(rings(t) for t in ALL_TIERS))
_raised = False
try:
    rings("T9")
except ValueError:
    _raised = True
check("未知档位必须报错，不能默默当成不响", _raised)
check("定档规则：support>=2→T1、<2→T2、降级→T2、仅陀螺仪→T3",
      assign(support=2) == T1 and assign(support=1) == T2
      and assign(support=3, degraded=True) == T2
      and assign(support=3, gyro_only=True) == T3)
check("两标记并存时以「仅陀螺仪」为准（置信更低）",
      assign(support=0, degraded=True, gyro_only=True) == T3)
check("缺数据不是支持也不是反对",
      count_support(ll=None, zc=None, ta=None) == 0
      and count_support(ll=0.9, mot_pct=0.85) == 2
      and count_support(ll=0.79, mot_pct=0.79) == 0)
check("任何档位 decide() 都响铃（约束 3：T2 无自动撤回）",
      all(decide(**kw)["rings"] is True
          for kw in ({"support": 2}, {"support": 0},
                     {"support": 0, "gyro_only": True},
                     {"support": 0, "degraded": True})))

print()
print("== 安全审计 ==")
_a = audit([{"tier": T1, "is_fatal": True},
            {"tier": T3, "is_fatal": True},
            {"tier": T2, "is_fatal": False}])
check("审计能逐档统计致命级且判定安全",
      _a["safe"] and _a["silent_fatal"] == 0
      and _a["per_tier"][T1]["fatal"] == 1
      and _a["per_tier"][T3]["fatal"] == 1
      and _a["per_tier"][T2]["other"] == 1)
_m = audit([{"tier": T1, "is_fatal": True}] * 18
           + [{"tier": T2, "is_fatal": True}] * 3
           + [{"tier": T3, "is_fatal": True}] * 2)
check("复现 run_r13_tiers.py 实测分布 T1=18/T2=3/T3=2，合计 23",
      sum(v["fatal"] for v in _m["per_tier"].values()) == 23 and _m["safe"])

print()
print("== schema 的 gyro 通道（21/23 → 23/23 的那一路）==")
_caps = Channels.seizeit2_gyro()
check("seizeit2_gyro() 声明 hr + motion + gyro",
      _caps.has("gyro") and _caps.has("motion") and _caps.has("hr")
      and "gyro" in _caps.present_fields())
check("旧 seizeit2() 保持原样，不悄悄多出一路信号",
      not Channels.seizeit2().has("gyro")
      and "gyro" in Channels.seizeit2().absent_fields())
_w = Window.build(t=1.0, caps=_caps, hr=80.0, motion=0.4, gyro=1.2)
check("Window 能携带 gyro 值", _w.get("gyro") == 1.2
      and _w.status("gyro") == "present")
_w2 = Window.build(t=1.0, caps=Channels.seizeit2(), hr=80.0, gyro=1.2)
check("设备没声明陀螺仪时，传了值也不采信",
      _w2.get("gyro") is None and _w2.status("gyro") == "absent")
_w3 = Window.build(t=1.0, caps=_caps, hr=80.0, motion=0.4, gyro=None)
check("声明了但本窗取不到 → DROPOUT（看不见≠没事）",
      _w3.status("gyro") == "dropout")

print()
print("=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
