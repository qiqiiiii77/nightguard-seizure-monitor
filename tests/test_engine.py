"""
test_engine.py — 引擎单元测试（移植 Tell tests 的关键性质 + 新增生理场景）

跑法：python3 tests/test_engine.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.gate import (GateConfig, PatientBaseline, NightGate,
                         fuse_signals, noisy_or, gate_decision)
from engine.signals import wearable_signals, eeg_signals
from sim.simulator import generate_night, windowize

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  √ {name}")
    else:
        FAIL += 1
        print(f"  × {name}  {detail}")


print("== 1. noisy-OR 融合性质（Tell 核心主张的移植验证）==")
cfg = GateConfig()
# 单个强信号不被七个弱信号稀释
strong = [{"key": "motion_rhythm", "doubt": 1.0, "detail": {}}]
weak7 = [{"key": f"k{i}", "doubt": 0.05, "detail": {}} for i in range(7)]
r_strong = fuse_signals(strong, cfg)["risk"]
r_mixed = fuse_signals(strong + weak7, cfg)["risk"]
check("单个强信号独立拉高风险", r_strong > 0.35, f"risk={r_strong}")
check("七个弱信号几乎不稀释强信号", r_mixed >= r_strong * 0.9, f"{r_mixed} vs {r_strong}")
# 多个弱信号可以叠加
r_weak = fuse_signals(weak7, cfg)["risk"]
r_one = fuse_signals(weak7[:1], cfg)["risk"]
check("多个弱信号叠加抬升风险", r_weak > r_one, f"{r_weak} > {r_one}")

print("== 2. 个人基线自适应 ==")
bl = PatientBaseline(min_windows=5)
for i in range(10):   # 这个孩子平时体动就大（1.8）、心率高（95）
    bl.observe({"motion": 1.8 + i * 0.01, "hr": 95 + i * 0.1,
                "breathing": 14, "spo2": 97, "breath_reg": 0.9})
check("基线就绪", bl.ready)
check("基线学到本人常态体动", abs(bl.center("motion") - 1.845) < 0.05, str(bl.center("motion")))
check("基线学到本人常态心率", abs(bl.center("hr") - 95.45) < 0.5, str(bl.center("hr")))

print("== 3. 门控分级 + 二次确认 ==")
scored = {"risk": 0.85, "components": [{"key": "motion_rhythm", "doubt": 1.0, "influence": 0.45,
                                        "contribution": 0.45, "detail": {}}], "evidence": ["节律性运动"]}
r1 = gate_decision(scored, cfg, streak=0, escalating=False)
check("首窗高风险只给 WARNING（不吵醒）", r1.level == "WARNING" and not r1.confirmed, r1.level)
r2 = gate_decision(scored, cfg, streak=1, escalating=False)
check("连续第2窗升级 CRITICAL", r2.level == "CRITICAL" and r2.confirmed, r2.level)
r3 = gate_decision(scored, cfg, streak=0, escalating=True)
check("连续抬升允许直接 CRITICAL（快通道）", r3.level == "CRITICAL", r3.level)
low = {"risk": 0.3, "components": [], "evidence": []}
check("低风险只 LOG 不打扰", gate_decision(low, cfg, 0, False).level == "LOG")
check("WARNING 带二次确认策略（probe 非空）", bool(r1.probe))

print("== 4. 整夜模拟：检出发作 ==")
night = generate_night(seed=7, minutes=90,
                       seizures=[{"onset_min": 50, "dur_s": 70, "intensity": 1.0}])
wins = windowize(night)
ng = NightGate(signal_fn=wearable_signals)
results = [ng.step(w) for w in wins]
seiz_wins = [i for i, w in enumerate(wins) if w["label"] == 1]
first_sz = min(seiz_wins)
after = results[first_sz:first_sz + 12]
levels = [r.level for r in after]
check("发作期出现 WARNING 以上", any(l in ("WARNING", "CRITICAL") for l in levels), str(levels))
check("发作期最终确认 CRITICAL", any(l == "CRITICAL" for l in levels), str(levels))
# 检出延迟（发作起点到第一个 WARNING+）
import bisect
first_alert = next((i for i in range(first_sz, first_sz + 12) if results[i].level != "LOG"), None)
if first_alert is not None:
    latency_s = (wins[first_alert]["t"] - wins[first_sz]["t"])
    check("检出延迟 < 60s", latency_s < 60, f"{latency_s}s")
# 证据链可解释
crit = next((r for r in results if r.level == "CRITICAL"), None)
check("CRITICAL 带证据链", crit is not None and len(crit.evidence) > 0, str(crit.evidence if crit else None))

print("== 5. 误报陷门（正常生理事件不应吵醒家长）==")
def runs(levels, want):
    """连续同级别的窗口算 1 次（一次翻身不该记成 8 次报警）"""
    cnt, prev = 0, None
    for l in levels:
        if l == want and prev != want:
            cnt += 1
        prev = l
    return cnt

for seed, name in [(11, "正常夜+翻身心跳波动"), (23, "正常夜+REM"), (31, "正常夜+呼吸暂停")]:
    night = generate_night(seed=seed, minutes=90, seizures=[], traps=True)
    wins = windowize(night)
    ng = NightGate(signal_fn=wearable_signals)
    results = [ng.step(w) for w in wins]
    levels = [r.level for r in results]
    check(f"{name}: 零 CRITICAL 误报", runs(levels, "CRITICAL") == 0,
          f"critical_runs={runs(levels, 'CRITICAL')}")
    check(f"{name}: WARNING 事件 ≤ 3", runs(levels, "WARNING") <= 3,
          f"warning_runs={runs(levels, 'WARNING')}")

print("== 6. EEG 信号配置可用（模态无关性）==")
ng_eeg = NightGate(signal_fn=eeg_signals)
for i in range(30):
    ng_eeg.step({"t": i * 4, "amp": 18 + i * 0.1, "rhythm": 0.2,
                 "spread": 0.05, "power": 40})
r = ng_eeg.step({"t": 130, "amp": 95.0, "rhythm": 0.72, "spread": 0.6, "power": 210})
check("EEG 发作样窗口触发 WARNING+", r.level in ("WARNING", "CRITICAL"),
      f"level={r.level} risk={r.risk}")
seq = [ng_eeg.step({"t": 134 + i * 4, "amp": 99.0, "rhythm": 0.75,
                    "spread": 0.65, "power": 220}) for i in range(6)]
check("EEG 持续发作窗口升级 CRITICAL", any(x.level == "CRITICAL" for x in seq),
      str([x.level for x in seq]))

print("== 7. 事件日志 ==")
# 正常夜：零事件（不产生任何打扰）
night_norm = generate_night(seed=11, minutes=90, seizures=[])
ng_norm = NightGate(signal_fn=wearable_signals)
for w in windowize(night_norm):
    ng_norm.step(w)
check("正常夜零事件", ng_norm.journal.summary()["total"] == 0, str(ng_norm.journal.summary()))
# 发作夜：有可追溯事件
night_sz = generate_night(seed=7, minutes=90,
                          seizures=[{"onset_min": 50, "dur_s": 70}])
ng_sz = NightGate(signal_fn=wearable_signals)
for w in windowize(night_sz):
    ng_sz.step(w)
js = ng_sz.journal.summary()
check("发作夜事件日志有记录", js["total"] > 0 and js["critical"] > 0, str(js))
ev = ng_sz.journal.events[0]
check("日志含证据链与驱动信号", bool(ev["evidence"]) and ev["driver"] is not None, str(ev)[:120])

print("== 8. 时间基持续性判据（sustain_seconds，与步进无关）==")
cfg_t = GateConfig(warning_threshold=0.50, critical_threshold=0.70,
                   confirm_windows=1, sustain_seconds=30)
ng_t = NightGate(config=cfg_t, signal_fn=eeg_signals)
for i in range(30):          # 60 秒正常窗（2s 步进）建基线
    ng_t.step({"t": i * 2, "amp": 18, "rhythm": 0.2, "spread": 0.05, "power": 40})
spike = ng_t.step({"t": 62, "amp": 95.0, "rhythm": 0.72, "spread": 0.6, "power": 210})
check("孤立尖峰（<30s）不报警", spike.level == "LOG",
      f"level={spike.level} risk={spike.risk}")
seq_t = [ng_t.step({"t": 64 + i * 2, "amp": 99.0, "rhythm": 0.75,
                    "spread": 0.65, "power": 220}) for i in range(16)]
check("持续 30s+ 的高风险升级 CRITICAL", any(x.level == "CRITICAL" for x in seq_t),
      str([x.level for x in seq_t]))
# 无 t 的输入自动退回窗口基，不报错
cfg_w = GateConfig(sustain_windows=3, sustain_seconds=30)
ng_w = NightGate(config=cfg_w, signal_fn=eeg_signals)
for i in range(25):
    ng_w.step({"amp": 18, "rhythm": 0.2, "spread": 0.05, "power": 40})
r_w = ng_w.step({"amp": 95.0, "rhythm": 0.72, "spread": 0.6, "power": 210})
check("无时间戳输入退回窗口基且不崩溃", r_w.level in ("LOG", "WARNING", "CRITICAL"), r_w.level)

print("== 9. 看护失效看门狗（信号缺失不得静默）==")
# 需求是「大发作绝对不可以漏报」。信号全失联时风险分天然很低，
# 旧行为是静默判 LOG —— 家长无法区分"一切正常"和"设备瞎了"。
# 实测发作期有中位 29.7% 的窗口拿不到心率（抽动毁掉 ECG），所以这条必须守住。
cfg_wd = GateConfig(watchdog_seconds=45)
ng_wd = NightGate(config=cfg_wd, signal_fn=wearable_signals)
for i in range(25):          # 先正常监护，建立基线
    ng_wd.step({"t": i * 2, "motion": 0.5, "hr": 70, "breathing": 16, "spo2": 97})
# 信号开始全失联
blind = [ng_wd.step({"t": 50 + i * 2}) for i in range(30)]
check("短暂失联（<45s）不报看护失效",
      all(b.level != "UNAVAILABLE" for b in blind[:20]),
      str([b.level for b in blind[:20]]))
check("持续失联超 45s 报 UNAVAILABLE",
      any(b.level == "UNAVAILABLE" for b in blind),
      str([b.level for b in blind[-5:]]))
u = next(b for b in blind if b.level == "UNAVAILABLE")
check("看护失效带可执行提示与证据",
      bool(u.probe) and bool(u.evidence) and u.driver == "signal_lost",
      f"probe={u.probe} ev={u.evidence}")
check("看护失效不伪装成发作风险", u.risk == 0.0 and not u.confirmed,
      f"risk={u.risk} confirmed={u.confirmed}")
# 信号恢复后立刻退出该状态
back = ng_wd.step({"t": 200, "motion": 0.5, "hr": 70, "breathing": 16, "spo2": 97})
check("信号恢复后退出看护失效", back.level != "UNAVAILABLE", back.level)
# EEG 配置不该被误判为失联（字段名不同）
ng_eeg_wd = NightGate(config=GateConfig(watchdog_seconds=45), signal_fn=eeg_signals)
eeg_lv = [ng_eeg_wd.step({"t": i * 2, "amp": 18, "rhythm": 0.2,
                          "spread": 0.05, "power": 40}).level for i in range(40)]
check("EEG 模态不被看门狗误伤", "UNAVAILABLE" not in eeg_lv,
      str(set(eeg_lv)))

# ------------------------------------------------------------------ #
# 二次确认（probe）的执行 —— 第七棒新增
# ------------------------------------------------------------------ #
# 这几条是用一次真实漏报换来的，别改松：sub-059 一次 59 秒全面性发作，
# 初版 probe_hr_surge 把发作**自己的**抽动当成"此前的运动"，连否 6 窗漏掉。

def _feed(ng, wins):
    return [ng.step(w) for w in wins]

# ① 运动后心率升高（先动、现已安静）→ 应当否决
# 阈值取与验证脚本一致的口径（warning = θ-0.2，θ=0.6 → 推荐工作点），
# 否则 probe 根本不会被触发——它只在 WARNING 上运行。
cfg_p = GateConfig(watchdog_seconds=0, warning_threshold=0.40,
                   critical_threshold=0.60)
ng_p = NightGate(config=cfg_p, signal_fn=wearable_signals)
# 建基线：安静
_feed(ng_p, [{"t": i * 2, "motion": 0.5, "hr": 70, "breathing": 16, "spo2": 97}
             for i in range(40)])
# 一段体动（翻身）
_feed(ng_p, [{"t": 80 + i * 2, "motion": 4.0, "hr": 78, "breathing": 17, "spo2": 97}
             for i in range(8)])
# 体动停下、先安静 24 秒（清空 20 秒近窗），这才叫"运动后"而不是"运动中"
_feed(ng_p, [{"t": 96 + i * 2, "motion": 0.5, "hr": 72, "breathing": 16, "spo2": 97}
             for i in range(12)])
# 此刻心率骤升（伴轻度呼吸增快，否则单信号封顶 0.35 到不了 WARNING——
# 这是「不可依赖单一信号判断」的红线决定的，见 GateConfig.influence），
# 但远窗里仍有刚才那段体动 → 应被判为运动后心率反应
quiet_hr = _feed(ng_p, [{"t": 120 + i * 2, "motion": 0.5, "hr": 130,
                         "breathing": 22, "spo2": 97} for i in range(5)])
vetoed = [r for r in quiet_hr if r.probe_verdict == "veto"]
check("运动后心率升高被二次确认否决", bool(vetoed),
      str([(r.level, r.driver, r.probe_verdict) for r in quiet_hr]))
check("被否决的窗口仍写入事件日志（可追溯）",
      any(e.get("probe_verdict") == "veto" for e in ng_p.journal.events),
      f"journal={len(ng_p.journal.events)} 条")

# ② 体动与心率同步持续抬升（发作形态）→ 绝不否决
ng_s = NightGate(config=cfg_p, signal_fn=wearable_signals)
_feed(ng_s, [{"t": i * 2, "motion": 0.5, "hr": 70, "breathing": 16, "spo2": 97}
             for i in range(40)])
# 体动一路爬升 + 心率同步上冲 —— 这正是 sub-059 的形态
sync = _feed(ng_s, [{"t": 80 + i * 2, "motion": 2.0 + i * 1.5, "hr": 100 + i * 6,
                     "breathing": 24, "spo2": 92} for i in range(12)])
check("体动与心率同步抬升时绝不否决（回归：sub-059 漏报）",
      not any(r.probe_verdict == "veto" for r in sync),
      str([(r.driver, r.probe_verdict) for r in sync]))
check("同步抬升能升到 CRITICAL",
      any(r.level == "CRITICAL" for r in sync),
      str([r.level for r in sync]))

# ③ veto 只能降 WARNING，永远不能降 CRITICAL
crit = [r for r in sync if r.level == "CRITICAL"]
check("CRITICAL 不被二次确认降级",
      all(r.probe_verdict != "veto" for r in crit), f"{len(crit)} 个 CRITICAL")

# ④ 开关能关掉（用于消融对照）
ng_off = NightGate(config=GateConfig(watchdog_seconds=0, warning_threshold=0.40,
                                     critical_threshold=0.60,
                                     probe_enabled=False),
                   signal_fn=wearable_signals)
_feed(ng_off, [{"t": i * 2, "motion": 0.5, "hr": 70, "breathing": 16, "spo2": 97}
               for i in range(40)])
_feed(ng_off, [{"t": 80 + i * 2, "motion": 4.0, "hr": 78, "breathing": 17,
                "spo2": 97} for i in range(8)])
off_hr = _feed(ng_off, [{"t": 96 + i * 2, "motion": 0.5, "hr": 112,
                         "breathing": 17, "spo2": 96} for i in range(6)])
check("probe_enabled=False 时不执行二次确认",
      all(r.probe_verdict is None for r in off_hr),
      str([r.probe_verdict for r in off_hr]))

# ───────────────────────────────────────────────────────────────────
# ⑤ 三个闩锁的回归（第八棒 diag_fa_autopsy / diag_baseline_latch 实测）
# ───────────────────────────────────────────────────────────────────

# 5.1 基线不被报警级别锁死。
# 场景复刻 sub-073：入夜头几分钟心率偏低（入睡过渡），随后整夜稳定在高位。
# 旧实现下基线被钉在过渡期的低值上，正常心率被当成骤升，81% 的窗报 CRITICAL。
ng_l = NightGate(config=GateConfig(watchdog_seconds=0, warning_threshold=0.30,
                                   critical_threshold=0.50),
                 signal_fn=wearable_signals)
# 入睡过渡：心率 58（全夜最低）
_feed(ng_l, [{"t": i * 2, "motion": 0.3, "hr": 58} for i in range(30)])
# 随后整夜稳定 104 —— 逐步爬升，避免被"心率跳变"判据挡住（那是正确行为）
_feed(ng_l, [{"t": 60 + i * 2, "motion": 0.3, "hr": 58 + i} for i in range(46)])
stable = _feed(ng_l, [{"t": 152 + i * 2, "motion": 0.3, "hr": 104}
                      for i in range(200)])
base_hr = ng_l.baseline.center("hr")
check("基线跟得上整夜稳定心率（不被级别锁死）",
      base_hr is not None and base_hr > 85,
      f"基线={None if base_hr is None else round(base_hr, 1)}（期望 >85，旧实现约 58）")
crit_frac = sum(1 for r in stable if r.level == "CRITICAL") / len(stable)
check("稳定正常心率不会整夜报 CRITICAL",
      crit_frac < 0.10, f"CRITICAL 占比 {crit_frac:.0%}（旧实现 81%）")

# 5.2 发作本身仍不进基线（修闩锁不能把安全性修掉）
ng_s = NightGate(config=GateConfig(watchdog_seconds=0), signal_fn=wearable_signals)
_feed(ng_s, [{"t": i * 2, "motion": 0.3, "hr": 70} for i in range(40)])
pool_before = len(ng_s.baseline.values["hr"])
# 发作：体动剧烈 + 心率急变
_feed(ng_s, [{"t": 80 + i * 2, "motion": 8.0, "hr": 170} for i in range(20)])
pool_after = len(ng_s.baseline.values["hr"])
check("发作期窗口不进基线",
      pool_after - pool_before <= 2,
      f"池增长 {pool_after - pool_before} 个（期望 ≤2）")

# 5.3 发作后高危期不自我续期。
# 旧实现：高危期内的 CRITICAL 会刷新锚点，而高危期本身更容易产生 CRITICAL
# → 自锁。实测一次发作后锁死 18 分钟（1152 秒）。
from engine.clinical_config import ClinicalConfig                    # noqa: E402
cfg_pi = GateConfig(watchdog_seconds=0, warning_threshold=0.30,
                    critical_threshold=0.50, clinical_enabled=True,
                    clinical=ClinicalConfig.for_hop(2.0))
ng_pi = NightGate(config=cfg_pi, signal_fn=wearable_signals)
_feed(ng_pi, [{"t": i * 2, "motion": 0.3, "hr": 70} for i in range(40)])
_feed(ng_pi, [{"t": 80 + i * 2, "motion": 8.0, "hr": 175} for i in range(15)])
anchor = ng_pi.last_critical_t
# 高危期内再喂一段异常：锚点不得被推后
_feed(ng_pi, [{"t": 110 + i * 2, "motion": 7.0, "hr": 170} for i in range(30)])
check("高危期内的 CRITICAL 不刷新锚点（不自我续期）",
      anchor is None or ng_pi.last_critical_t == anchor,
      f"锚点 {anchor} → {ng_pi.last_critical_t}")

# 5.4 全局 ready 标志不得当作单通道可用性（两个潜伏 bug 的回归）
# 只有 hr+motion 的设备（SeizeIT2 就是），breathing/spo2 整夜为 None。
ng_ch = NightGate(config=GateConfig(watchdog_seconds=0), signal_fn=wearable_signals)
try:
    _feed(ng_ch, [{"t": i * 2, "motion": 0.3, "hr": 70} for i in range(40)])
    # ready 已为真，但 breathing/spo2 两路没有任何样本
    r_ch = ng_ch.step({"t": 80, "motion": 0.3, "hr": 70, "breathing": 16})
    r_ch2 = ng_ch.step({"t": 82, "motion": 0.3, "hr": 70, "spo2": 97})
    ok_ch = True
    err_ch = f"breathing 窗={r_ch.level} spo2 窗={r_ch2.level}"
except (UnboundLocalError, TypeError) as e:
    ok_ch, err_ch = False, f"{type(e).__name__}: {e}"
check("ready=True 但单通道无样本时不崩", ok_ch, err_ch)

# 5.5 两个闩锁的**耦合**形态（实测：单独还原任一个都复现不了环路）
# 消融开关只为实验存在，这里断言它们确实接在环上 —— 否则消融结论是假的。
cfg_both = GateConfig(watchdog_seconds=0, warning_threshold=0.30,
                      critical_threshold=0.50, clinical_enabled=True,
                      clinical=ClinicalConfig.for_hop(2.0),
                      ablate_baseline_latch=True, ablate_anchor_guard=True)
ng_both = NightGate(config=cfg_both, signal_fn=wearable_signals)
_feed(ng_both, [{"t": i * 2, "motion": 0.3, "hr": 58} for i in range(30)])
_feed(ng_both, [{"t": 60 + i * 2, "motion": 0.3, "hr": 58 + i} for i in range(46)])
_feed(ng_both, [{"t": 152 + i * 2, "motion": 0.3, "hr": 104} for i in range(200)])
pool_latched = len(ng_both.baseline.values["hr"])

ng_fixed = NightGate(config=GateConfig(
    watchdog_seconds=0, warning_threshold=0.30, critical_threshold=0.50,
    clinical_enabled=True, clinical=ClinicalConfig.for_hop(2.0)),
    signal_fn=wearable_signals)
_feed(ng_fixed, [{"t": i * 2, "motion": 0.3, "hr": 58} for i in range(30)])
_feed(ng_fixed, [{"t": 60 + i * 2, "motion": 0.3, "hr": 58 + i} for i in range(46)])
_feed(ng_fixed, [{"t": 152 + i * 2, "motion": 0.3, "hr": 104} for i in range(200)])
pool_fixed = len(ng_fixed.baseline.values["hr"])
check("消融开关确实复现闩锁（池被饿死）",
      pool_latched < pool_fixed,
      f"还原后池={pool_latched}，修复后池={pool_fixed}（期望前者更小）")
check("生产默认不带消融开关",
      GateConfig().ablate_baseline_latch is False
      and GateConfig().ablate_anchor_guard is False,
      "默认值必须为 False")

print(f"\n{'=' * 40}\n通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
