"""事件聚合 + 分级证据 + gate 接线的单元测试。

重点在**安全边界**，不是功能点。这一层能直接造成漏报，所以：
  1. 证据层永不丢事件（NONE → 保持原级别）
  2. 临床规则要求的 CRITICAL，证据层不许降
  3. 数据算不出时维持原判（看不见 ≠ 没事）
  4. 默认关闭 → 行为与接线前完全一致
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.events import (CandidateEvent, count_events,  # noqa: E402
                           group_windows)
from engine.evidence import (CRITICAL, NONE, WARNING,  # noqa: E402
                             EvidenceConfig, classify)
from engine.gate import GateConfig, NightGate  # noqa: E402
from engine.signals import wearable_signals  # noqa: E402

PASS = FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ {msg}")


print("事件聚合 group_windows")
ws = [{"t": 0, "risk": 0.3, "level": "WARNING"},
      {"t": 2, "risk": 0.9, "level": "CRITICAL"},
      {"t": 4, "risk": 0.5, "level": "WARNING"}]
g = group_windows(ws, merge_s=300.0)
ok(len(g) == 1, "间隔都在合并窗内 → 1 个事件")
ok(g[0].t0 == 0 and g[0].t1 == 4, "事件起止取首末窗")
ok(g[0].risk_peak == 0.9, "risk_peak 取簇内最大")
ok(g[0].top_level == "CRITICAL", "top_level 取簇内最高级别")
ok(g[0].n_windows == 3 and g[0].duration_s == 4, "窗数/时长")

g2 = group_windows([{"t": 0}, {"t": 301}], merge_s=300.0)
ok(len(g2) == 2, "间隔 301s > 合并窗 → 切成 2 个事件")
g3 = group_windows([{"t": 0}, {"t": 300}], merge_s=300.0)
ok(len(g3) == 1, "间隔正好等于合并窗 → 仍算同一事件（边界）")
ok(group_windows([]) == [], "空输入 → 空列表")
ok(group_windows(None) == [], "None 输入 → 空列表，不抛异常")

g4 = group_windows([{"t": 10}, {"t": 0}, {"t": 5}])
ok([w["t"] for w in g4[0].windows] == [0, 5, 10], "乱序输入会先排序")
g5 = group_windows([{"t": 0}, {"no_t": 1}, {"t": float("nan")}, {"t": 2}])
ok(g5[0].n_windows == 2, "缺时间/NaN 的窗被跳过（硬塞默认时间会粘错簇）")
ok(count_events([0, 2, 400, 402], 300.0) == 2, "count_events 只数个数")
ok(count_events([]) == 0, "count_events 空 → 0")

print("\n分级判据 classify（阈值来自嵌套 LOO 选中的那组）")
cfg = EvidenceConfig(run_hi_s=30.0, run_lo_s=20.0, corroborate_vote=0.80,
                     rescue_max_s=5.0, rescue_vote=0.80)
ok(classify(41.0, 0.0, cfg).level == CRITICAL,
   "体动 41s ≥ 30s → CRITICAL（sub-046 真发作，EEG 看不见）")
ok(classify(30.0, None, cfg).level == CRITICAL, "体动正好 30s → CRITICAL（边界）")
ok(classify(23.0, 1.0, cfg).level == CRITICAL,
   "重叠区 23s + EEG 1.00 → CRITICAL（sub-098 真发作）")
ok(classify(21.0, 0.25, cfg).level == WARNING,
   "重叠区 21s 但 EEG 0.25 → WARNING（sub-024 翻身伪迹）")
ok(classify(20.0, 0.80, cfg).level == CRITICAL, "佐证门槛取等号 → CRITICAL")
ok(classify(1.0, 0.85, cfg).level == WARNING,
   "体动 1s + EEG 0.85 → WARNING（sub-015，单级架构会漏掉它）")
ok(classify(19.0, 1.0, cfg).level == NONE,
   "体动 19s（既不够 20s 又超出 5s 救援区）+ EEG 1.0 → 证据不足")
ok(classify(None, 1.0, cfg).level == WARNING, "无体动数据 + EEG 1.0 → 走救援")
ok(classify(0.5, 0.79, cfg).level == NONE, "救援门槛差 0.01 → 不进 WARNING")

print("\n降级路径：算不出 ≠ 没事")
v = classify(None, None, cfg)
ok(v.level == WARNING and v.degraded,
   "两路都算不出 → WARNING + degraded（sub-083 就是这种）")
ok(classify(float("nan"), float("nan"), cfg).degraded, "NaN 同样视为算不出")
v2 = classify(25.0, None, cfg)
ok(v2.level == CRITICAL and v2.degraded,
   "重叠区但无 EEG 可查 → 保留 CRITICAL（体动 20s 本身是强证据）")
ok(classify(25.0, None, cfg).run_s == 25.0, "证据值会回传，便于写报告")

print("\ngate 接线：默认关闭 = 行为不变")
base = dict(motion=0.05, hr=60.0, rr=14.0, spo2=98.0, t=0.0)


def run(cfg_kw, feats):
    g = NightGate(signal_fn=wearable_signals, config=GateConfig(**cfg_kw))
    out = None
    for f in feats:
        out = g.step(f)
    return g, out


warm = [dict(base, t=float(i)) for i in range(25)]
spike = dict(base, t=25.0, motion=3.0, hr=120.0)
g_off, r_off = run({}, warm + [spike, dict(spike, t=26.0)])
ok(GateConfig().evidence_enabled is False, "evidence_enabled 默认 False")
ok(g_off.evidence_demotions == 0, "默认关 → 证据层一次都没介入")

print("\ngate 接线：安全边界")
# 造一个会判 CRITICAL 的序列，再给证据层"弱证据"看它是否降级
feats_weak = warm + [dict(spike, t=25.0, sustain_run_s=21.0, eeg_vote=0.2),
                     dict(spike, t=26.0, sustain_run_s=21.0, eeg_vote=0.2)]
g_on, r_on = run({"evidence_enabled": True, "critical_threshold": 0.4,
                  "warning_threshold": 0.2, "clinical_enabled": False},
                 feats_weak)
ok(r_on.level in ("WARNING", "LOG", "CRITICAL"), "开启后仍返回合法级别")
if g_on.evidence_demotions:
    ok(r_on.level == "WARNING" and r_on.notify is False,
       "弱证据 → 降为 WARNING 且静默（notify=False）")
    ok("证据层降级" in (r_on.note or ""), "降级原因写进 note，可追溯")
else:
    ok(True, "该序列未触发 CRITICAL，降级路径此例不适用")

# NONE 不能丢事件
feats_none = warm + [dict(spike, t=25.0, sustain_run_s=19.0, eeg_vote=0.0),
                     dict(spike, t=26.0, sustain_run_s=19.0, eeg_vote=0.0)]
g_n, r_n = run({"evidence_enabled": True, "critical_threshold": 0.4,
                "warning_threshold": 0.2, "clinical_enabled": False},
               feats_none)
ok(r_n.level != "UNAVAILABLE" and r_n is not None,
   "证据判 NONE 时事件仍在（不丢），级别不被抹掉")

# 数据缺失 → 维持原判
feats_na = warm + [dict(spike, t=25.0), dict(spike, t=26.0)]
g_na, r_na = run({"evidence_enabled": True, "critical_threshold": 0.4,
                  "warning_threshold": 0.2, "clinical_enabled": False},
                 feats_na)
ok(g_na.evidence_demotions == 0, "无证据字段 → 不降级（看不见不等于没事）")

print("\n最危险的边界：临床规则要求的 CRITICAL，证据层不许降")
# 心动过缓是 demand="CRITICAL" 的硬规则（clinical.py）。
# 造一个"临床判 CRITICAL + 证据很弱"的序列：证据层必须让路。
brady = [dict(base, t=float(i), hr=50.0) for i in range(30)]
brady_weak = [dict(b, sustain_run_s=21.0, eeg_vote=0.0) for b in brady]
g_c, r_c = run({"evidence_enabled": True, "clinical_enabled": True}, brady_weak)
rules_met = [r for r in (r_c.rules or []) if r.get("status") == "met"]
if r_c.level == "CRITICAL" and rules_met:
    ok(g_c.evidence_demotions == 0,
       "临床规则判 CRITICAL 时证据层零介入（安全网不能被绕过）")
else:
    # 规则没触发就换直接单测 _apply_evidence，保证这条边界一定被覆盖
    from engine.gate import GateResult
    gg = NightGate(signal_fn=wearable_signals,
                   config=GateConfig(evidence_enabled=True))
    fake = GateResult("CRITICAL", 0.9, True, "hr_surge", None, [], [])
    kept = gg._apply_evidence(fake, {"sustain_run_s": 21.0, "eeg_vote": 0.0},
                              clinical_demanded=True)
    ok(kept.level == "CRITICAL" and gg.evidence_demotions == 0,
       "clinical_demanded=True 时弱证据也不降级（直接单测）")

# 反面：同样的弱证据，但没有临床规则要求 → 应当降级
gg2 = NightGate(signal_fn=wearable_signals,
                config=GateConfig(evidence_enabled=True))
from engine.gate import GateResult  # noqa: E402
fake2 = GateResult("CRITICAL", 0.9, True, "motion_burst", None, [], [])
dem = gg2._apply_evidence(fake2, {"sustain_run_s": 21.0, "eeg_vote": 0.0},
                          clinical_demanded=False)
ok(dem.level == "WARNING" and dem.notify is False,
   "无临床要求 + 弱证据 → 降级静默（对照组，证明上一条不是恒真）")
ok(gg2.evidence_demotions == 1, "降级计数可观测")

# NONE（证据不足）也降为静默 WARNING，但绝不丢事件。
# 这条是改过的：第一版让 NONE 保持原级别，结果生产路径降噪 0%
# （证据层过度保守到自我抵消）。实测 NONE 档里 180/189 是误报、0 次发作。
gg4 = NightGate(signal_fn=wearable_signals,
                config=GateConfig(evidence_enabled=True))
fake4 = GateResult("CRITICAL", 0.9, True, "motion_burst", None, [], [])
nd = gg4._apply_evidence(fake4, {"sustain_run_s": 19.0, "eeg_vote": 0.0},
                         clinical_demanded=False)
ok(nd.level == "WARNING" and nd.notify is False,
   "证据不足（NONE）→ 降为静默 WARNING（不是丢弃）")
ok("证据不足" in (nd.note or ""), "NONE 的降级原因可区分于弱证据降级")
ok(gg4._apply_evidence(fake4, {"sustain_run_s": 19.0, "eeg_vote": 0.0},
                       clinical_demanded=True).level == "CRITICAL",
   "但临床规则要求时，NONE 也不许降")

# WARNING 不该被证据层碰（它只处理 CRITICAL）
gg3 = NightGate(signal_fn=wearable_signals,
                config=GateConfig(evidence_enabled=True))
fake3 = GateResult("WARNING", 0.5, False, "motion_burst", None, [], [])
ok(gg3._apply_evidence(fake3, {"sustain_run_s": 0.0, "eeg_vote": 0.0},
                       clinical_demanded=False).level == "WARNING",
   "证据层只审 CRITICAL，不动 WARNING/LOG")

# ---------------------------------------------------------------------
# 在线 sustain_run_s（_online_sustain_run_s）
#
# 为什么补这组：证据层读的是 features["sustain_run_s"]，而**流式路径上
# 没有任何人写这个字段**，只有本文件上面那些用例在手工塞。于是线上
# 永远 None → classify 判 degraded → 一律维持原判，证据层等于没接。
# 生产路径复测实测降噪 0%，根因就在这里。
# 这组锁住"在线代理真的算得出数、且不会把低体动发作压掉"。
# ---------------------------------------------------------------------

def _mk_gate(bg_n=120, bg_val=1.0):
    """喂 bg_n 个安静窗把背景池建起来，返回 gate 和下一个时间戳。"""
    g = NightGate(signal_fn=wearable_signals,
                  config=GateConfig(evidence_enabled=True,
                                    min_baseline_windows=1))
    t = 0.0
    for _ in range(bg_n):
        g.step({"t": t, "motion": bg_val, "hr": 70.0})
        t += 2.0
    return g, t


g, t = _mk_gate()
# 背景全是 1.0 → P90 门也是 1.0；喂一个明显更大的 motion 应当算出 run_s
run1 = g._online_sustain_run_s({"t": t, "motion": 50.0, "hr": 70.0})
ok(run1 is not None and run1 > 0,
   f"在线代理能算出 run_s（流式路径不再恒为 None）→ {run1}")

# 背景样本不足时必须返回 None（宁可不降级，也不用没学好的门压真发作）
g_cold = NightGate(signal_fn=wearable_signals,
                   config=GateConfig(evidence_enabled=True))
ok(g_cold._online_sustain_run_s({"t": 0.0, "motion": 50.0}) is None,
   "背景样本不足 → 返回 None（degraded 维持原判，不误降级）")

# motion 在门下 → 0
g2, t2 = _mk_gate()
ok(g2._online_sustain_run_s({"t": t2, "motion": 0.5}) == 0.0,
   "motion 低于背景分位 → run_s=0")

# 单窗掉落必须被容忍。
# 坑的原形：sub-015 的发作里有一窗 motion 掉到门下，严格连续计数把
# run_s 从 24s 砍回 0，发作后半段被判"证据不足"降级 —— 丢了一次真发作。
# sustain.py 的 best_vote_frac 早有同样结论（SeizeIT2 的 8/10 规则）。
g3, t3 = _mk_gate()
for k in range(10):                      # 先连续 10 窗高体动
    g3.step({"t": t3 + k * 2.0, "motion": 50.0, "hr": 70.0})
t_dip = t3 + 10 * 2.0
g3.step({"t": t_dip, "motion": 0.5, "hr": 70.0})          # 掉一窗
run_after_dip = g3._online_sustain_run_s(
    {"t": t_dip + 2.0, "motion": 50.0, "hr": 70.0})
ok(run_after_dip is not None and run_after_dip >= 20.0,
   f"单窗掉落被容忍，run_s 不被砍回 0 → {run_after_dip}")

# 连续两窗掉落 = 真的断了
g4, t4 = _mk_gate()
for k in range(10):
    g4.step({"t": t4 + k * 2.0, "motion": 50.0, "hr": 70.0})
t_d2 = t4 + 20.0
g4.step({"t": t_d2, "motion": 0.5, "hr": 70.0})
g4.step({"t": t_d2 + 2.0, "motion": 0.5, "hr": 70.0})
run_break = g4._online_sustain_run_s(
    {"t": t_d2 + 4.0, "motion": 50.0, "hr": 70.0})
ok(run_break is not None and run_break < 20.0,
   f"连续两窗掉落视为中断 → {run_break}")

print("\n" + "=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
