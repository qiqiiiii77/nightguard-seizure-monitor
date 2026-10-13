#!/usr/bin/env python3
"""响铃塌缩的语义约束：压"一个事件响几声"，绝不压"哪些事件会响"。

来历
----
用户的验收标准是**响铃次数**：灵敏度 100% 且误响铃 ≤2 次/10 夜。
事件级误报率不约束响铃数 —— 已知最坏案例 9 个误报事件 = 2846 次响铃。
实测既有最好架构事件级 22/22 @ 0.049/h，响铃级却是 63 次/10 夜。

ring_collapse 让一个事件只出一声，响铃级降到 4.0 次/10 夜且灵敏度不变。

这组测试锁的是它**不会重犯 alarm_episode_enabled 的错**：
那个封顶按时间段跨事件累计，于是"发作前的误报吃掉额度"，
真发作被整段静默（实测压掉 6 次真发作，召回假性显示 16/22）。
塌缩按**事件**计数、事件结束即清零，所以压不到下一个事件。
下面第 3、4 条就是专门钉这一点的回归测试。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.gate import GateConfig, NightGate  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {extra}")


def three_way(features, baseline, history):
    """三路满格 → risk ≈ 0.79，稳定在 CRITICAL 之上。"""
    return [{"key": "motion_burst", "doubt": 1.0, "detail": {}, "note": "测试用"},
            {"key": "motion_rhythm", "doubt": 1.0, "detail": {}, "note": None},
            {"key": "hr_surge", "doubt": 1.0, "detail": {}, "note": None}]


def quiet(features, baseline, history):
    return []


def cfg(**kw):
    base = dict(clinical_enabled=False, critical_threshold=0.70,
                warning_threshold=0.40, confirm_windows=1,
                min_baseline_windows=1, alarm_throttle_s=0.0,
                alarm_min_gap_s=0.0, alarm_episode_enabled=False)
    base.update(kw)
    return GateConfig(**base)


def drive(ng, spec, step=30, t0=0.0):
    """spec: [(是否高危, 窗数), ...] → 返回 [(t, result)]。"""
    out, t = [], t0
    for hot, n in spec:
        for _ in range(n):
            ng.signal_fn = three_way if hot else quiet
            r = ng.step({"t": t, "motion": 5.0 if hot else 0.0,
                         "hr": 100.0 if hot else 60.0})
            out.append((t, r))
            t += step
    return out


print("=" * 60)
print("响铃塌缩：一个事件只出一声")
print("=" * 60)

# --- 1. 同一事件内只响一次 ----------------------------------------
ng = NightGate(config=cfg(ring_collapse=True), signal_fn=three_way)
out = drive(ng, [(True, 10)])
crit = [r for _t, r in out if r.level == "CRITICAL"]
rung = [r for r in crit if r.notify]
check("长事件只响 1 声", len(rung) == 1, f"→ 响了 {len(rung)} 声")
check("判定级别不受影响（CRITICAL 照常产生）", len(crit) >= 8,
      f"→ CRITICAL {len(crit)}/10")
check("第一声必须是响的（绝不节流首次报警）",
      crit[0].notify is True)

# 对照：关掉塌缩应该响很多声
ng2 = NightGate(config=cfg(ring_collapse=False), signal_fn=three_way)
out2 = drive(ng2, [(True, 10)])
rung2 = [r for _t, r in out2 if r.level == "CRITICAL" and r.notify]
check("关掉塌缩则响多声（证明是塌缩起的作用）", len(rung2) > len(rung),
      f"→ 开启 {len(rung)} 声 vs 关闭 {len(rung2)} 声")

# --- 2. 静默只改 notify，不改 level、不丢日志 ----------------------
silenced = [r for r in crit if not r.notify]
check("被塌缩静默的窗仍是 CRITICAL（不许降级）",
      all(r.level == "CRITICAL" for r in silenced))
check("静默的窗有说明（可追溯）",
      all(r.note and "静默" in r.note for r in silenced),
      f"→ {silenced[0].note if silenced else '无'}")
check("塌缩计数器可观测", ng.collapsed == len(silenced),
      f"→ collapsed={ng.collapsed} vs 静默 {len(silenced)}")

# --- 3. 回归：塌缩不得压掉**下一个**事件的第一声 ------------------
# 这正是 alarm_episode_enabled 翻车的地方：误报响满额度 → 紧跟的
# 真发作被静默。塌缩按事件计数，必须压不到第二个事件。
ng3 = NightGate(config=cfg(ring_collapse=True), signal_fn=three_way)
out3 = drive(ng3, [(True, 6), (False, 6), (True, 6)])
ring_ts = [t for t, r in out3 if r.level == "CRITICAL" and r.notify]
check("两个事件各响一声（第二个事件没被第一个吃掉）",
      len(ring_ts) == 2, f"→ 响铃时刻 {ring_ts}")
check("第二声落在第二个事件内（不是第一个事件的余波）",
      len(ring_ts) == 2 and ring_ts[1] >= 12 * 30,
      f"→ {ring_ts}")

# --- 4. 回归：连续多个误报事件之后，真发作仍然响 ------------------
# 模拟"患者躁动整夜 → 一连串误报事件 → 真发作紧随"。
ng4 = NightGate(config=cfg(ring_collapse=True), signal_fn=three_way)
spec = []
for _ in range(5):                       # 5 个"误报"事件
    spec += [(True, 4), (False, 5)]
spec += [(True, 4)]                      # 第 6 个 = "真发作"
out4 = drive(ng4, spec)
ring_ts4 = [t for t, r in out4 if r.level == "CRITICAL" and r.notify]
check("6 个事件响 6 声（每事件一声，误报不吃真发作的额度）",
      len(ring_ts4) == 6, f"→ 响了 {len(ring_ts4)} 声：{ring_ts4}")
last_event_t0 = sum(n for _h, n in spec[:-1]) * 30
check("**最后那次（真发作）一定响了**",
      any(t >= last_event_t0 for t in ring_ts4),
      f"→ 末事件起点 {last_event_t0}，响铃 {ring_ts4}")

# --- 5. 升级仍可击穿塌缩 ------------------------------------------
# 事态升级必须出声，否则"从轻微到全身抽动"的过程被捂住。
def ramp():
    """风险逐窗抬高：一路 → 两路 → 三路。"""
    seq = [[{"key": "motion_burst", "doubt": 1.0, "detail": {}, "note": None}],
           [{"key": "motion_burst", "doubt": 1.0, "detail": {}, "note": None},
            {"key": "motion_rhythm", "doubt": 1.0, "detail": {}, "note": None}],
           [{"key": "motion_burst", "doubt": 1.0, "detail": {}, "note": None},
            {"key": "motion_rhythm", "doubt": 1.0, "detail": {}, "note": None},
            {"key": "hr_surge", "doubt": 1.0, "detail": {}, "note": None}]]
    box = {"i": 0}

    def fn(features, baseline, history):
        s = seq[min(box["i"], len(seq) - 1)]
        box["i"] += 1
        return s
    return fn


ng5 = NightGate(config=cfg(ring_collapse=True, critical_threshold=0.35,
                           alarm_breakthrough_margin=0.12),
                signal_fn=ramp())
out5 = []
for i in range(6):
    out5.append((i * 30, ng5.step({"t": i * 30, "motion": 5.0, "hr": 100.0})))
rung5 = [t for t, r in out5 if r.level == "CRITICAL" and r.notify]
check("风险升级时可击穿塌缩（事态升级必须出声）", len(rung5) >= 2,
      f"→ 响铃时刻 {rung5}")

# --- 6. 默认配置：塌缩开、发作期封顶关 -----------------------------
d = GateConfig()
check("默认 ring_collapse=True", d.ring_collapse is True)
check("默认 alarm_episode_enabled=False（它会静默真发作）",
      d.alarm_episode_enabled is False)

# --- 7. 回归：collapse_clear_s 必须保持 0 --------------------------
# 实测它一开就掉 6 次发作（见 GateConfig.collapse_clear_s 注释）。
# 机制与 alarm_episode_enabled 相同：事件边界一旦跨越安静期，
# 误报就能占满额度、把紧跟的真发作当成"同一事件的延续"静默掉。
check("默认 collapse_clear_s=0（非 0 会掉 6 次发作，实测）",
      GateConfig().collapse_clear_s == 0.0,
      f"→ {GateConfig().collapse_clear_s}")

# 直接钉住失败机制本身：开了 clear_s，误报之后的真发作就被吃掉。
ng7 = NightGate(config=cfg(ring_collapse=True, collapse_clear_s=300.0),
                signal_fn=three_way)
# "误报"事件 → 短暂安静(不足 300s) → "真发作"
out7 = drive(ng7, [(True, 4), (False, 4), (True, 4)], step=30)
ring_ts7 = [t for t, r in out7 if r.level == "CRITICAL" and r.notify]
check("（反面证据）clear_s=300 时第二个事件被吃掉 —— 正是不能开的原因",
      len(ring_ts7) == 1,
      f"→ 响铃 {ring_ts7}（只有 1 声 = 真发作被静默）")
# 同样的输入，clear_s=0 时两声都在
ng8 = NightGate(config=cfg(ring_collapse=True, collapse_clear_s=0.0),
                signal_fn=three_way)
out8 = drive(ng8, [(True, 4), (False, 4), (True, 4)], step=30)
ring_ts8 = [t for t, r in out8 if r.level == "CRITICAL" and r.notify]
check("同样输入下 clear_s=0 两个事件都响（对照）", len(ring_ts8) == 2,
      f"→ 响铃 {ring_ts8}")

print()
print("=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
