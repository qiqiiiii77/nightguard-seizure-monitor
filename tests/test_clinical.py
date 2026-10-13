"""
test_clinical.py — 临床规则层测试（第八棒）

每条医生判据一组测试：规则该命中时必须命中，不该命中时必须不命中，
缺通道时必须报 PARTIAL 而**不是**静默当作不满足。

跑法：python3 tests/test_clinical.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.gate import GateConfig, NightGate, apply_clinical, GateResult
from engine.clinical import (evaluate_rules, coverage_report, MET, PARTIAL, UNMET,
                             rule_hr_surge_sustained, rule_bradycardia, rule_apnea,
                             rule_desat_confirm, rule_prolonged_event)
from engine.clinical_config import ClinicalConfig
from engine.schema import Window, Channels, PRESENT, DROPOUT, ABSENT
from engine.signals import wearable_signals

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  √ {name}")
    else:
        FAIL += 1
        print(f"  × {name}  {detail}")


class FakeBaseline:
    """固定基线，让规则测试不依赖基线建立过程。"""
    def __init__(self, **centers):
        self.ready = True
        self._c = centers

    def center(self, k):
        return self._c.get(k)

    def spread(self, k):
        return {"hr": 2.0, "spo2": 0.8, "breathing": 1.0, "motion": 0.2}.get(k, 1.0)


def hist(points):
    """[(t, {field: val}), ...] → history 列表"""
    return [dict(t=t, **vals) for t, vals in points]


cfg = ClinicalConfig(window_hop_s=2.0)

# ------------------------------------------------------------------ #
print("== R1 大发作主判据：心率升≥30 且持续≥45 秒 ==")

base = FakeBaseline(hr=70.0)

# 升幅够但只持续 10 秒 → PARTIAL（不该报）
h = hist([(t, {"hr": 105.0}) for t in range(1000, 1010, 2)])
o = rule_hr_surge_sustained({"t": 1010.0, "hr": 105.0}, base, h, cfg)
check("升幅达标但仅持续10秒 → partial", o.status == PARTIAL, o.status)

# 升幅够且持续 50 秒 → MET
h = hist([(t, {"hr": 105.0}) for t in range(960, 1010, 2)])
o = rule_hr_surge_sustained({"t": 1010.0, "hr": 105.0}, base, h, cfg)
check("升幅达标且持续50秒 → met", o.status == MET, f"{o.status} held={o.detail.get('held_s')}")
check("met 时要求升级", o.demand == "ESCALATE", str(o.demand))
check("说明里带医生判据数字", o.note and "45" in o.note, str(o.note))

# 升幅不足（+15）即使持续很久也不命中
h = hist([(t, {"hr": 85.0}) for t in range(900, 1010, 2)])
o = rule_hr_surge_sustained({"t": 1010.0, "hr": 85.0}, base, h, cfg)
check("升幅不足(+15)持续再久也不命中", o.status == UNMET, o.status)

# 中途断一次 → 持续时长从断点后重算，不该命中
pts = [(t, {"hr": 105.0}) for t in range(960, 1010, 2)]
pts[len(pts) // 2] = (pts[len(pts) // 2][0], {"hr": 72.0})   # 中间掉回基线
o = rule_hr_surge_sustained({"t": 1010.0, "hr": 105.0}, base, hist(pts), cfg)
check("持续期间断过一次 → 不算连续45秒", o.status == PARTIAL,
      f"{o.status} held={o.detail.get('held_s')}")

# 缺心率通道 → PARTIAL + missing，不是 UNMET
w = Window.build(t=1010.0, caps=Channels(motion=True), motion=0.3)
o = rule_hr_surge_sustained(w.as_features(), base, [], cfg)
check("无心率通道 → partial 且声明 missing",
      o.status == PARTIAL and "hr" in o.missing, f"{o.status} {o.missing}")

# ------------------------------------------------------------------ #
print("== R2 心动过缓：持续低于 60bpm 达 10 秒（第七棒完全没有）==")

h = hist([(t, {"hr": 52.0}) for t in range(994, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 52.0}, FakeBaseline(hr=78.0), h, cfg)
check("心率52持续16秒 → met", o.status == MET, f"{o.status} held={o.detail.get('held_s')}")

o = rule_bradycardia({"t": 1010.0, "hr": 52.0}, FakeBaseline(hr=78.0), [], cfg)
check("心率低但刚出现(<10秒) → 未达判据", o.status != MET, o.status)

# 回归（实测踩坑，diag_brady3.py）：判据是"绝对60 AND 比基线低15"的合取。
# 这批孩子里 sub-024/059/069/031 睡眠心率中位就在 64-68bpm，
# 深睡掉到 58 是正常生理，不是 SUDEP 前兆。单用绝对 60 本底 0.80/h。
h = hist([(t, {"hr": 58.0}) for t in range(960, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 58.0}, FakeBaseline(hr=64.0), h, cfg)
check("低基线患儿(64bpm)深睡掉到58 → 不报（正常生理）", o.status == UNMET,
      f"{o.status} drop={o.detail.get('drop')}")

# 同一个孩子真的掉到 46（比基线低 18）→ 必须报
h = hist([(t, {"hr": 46.0}) for t in range(960, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 46.0}, FakeBaseline(hr=64.0), h, cfg)
check("同一患儿掉到46（低18）→ 报 CRITICAL",
      o.status == MET and o.demand == "CRITICAL", f"{o.status} {o.demand}")

# 高基线患儿(130bpm)掉到 58：绝对门限和落差都满足 → 必须报
h = hist([(t, {"hr": 58.0}) for t in range(960, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 58.0}, FakeBaseline(hr=130.0), h, cfg)
check("高基线患儿(130bpm)掉到58 → 报（巨幅下降）", o.status == MET,
      f"{o.status} drop={o.detail.get('drop')}")

# 基线未建立时退回绝对门限，不得静默
class NotReady:
    ready = False
    def center(self, k): return None
    def spread(self, k): return None

h = hist([(t, {"hr": 52.0}) for t in range(960, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 52.0}, NotReady(), h, cfg)
check("基线未建立时退回绝对门限（不静默）", o.status == MET, o.status)

# 回归（实测踩坑，diag_brady2.py）：缺值必须中断连续性。
# 原实现只遍历非空样本，于是 t=1210 和 t=62788 两个孤立低心率
# 被算成"连续 61578 秒心动过缓"。发作期肌电干扰恰好让 ECG 大量丢点，
# 这个 bug 在最关键的时刻最严重。
h = hist([(1210.0, {"hr": 52.0})]) + hist([(62788.0, {"hr": 52.0})])
o = rule_bradycardia({"t": 62790.0, "hr": 52.0}, FakeBaseline(hr=78.0), h, cfg)
check("整夜缺值不得算成连续心动过缓（回归：61578秒假段）",
      o.status != MET, f"{o.status} held={o.detail.get('held_s')}")

# 段内大半是缺值（20秒里只有2个真实低值）→ 不算持续
h = hist([(1000.0, {"hr": 52.0}), (1018.0, {"hr": 52.0})])
o = rule_bradycardia({"t": 1020.0, "hr": 52.0}, FakeBaseline(hr=78.0), h, cfg)
check("段内大量缺值 → 不算持续", o.status != MET,
      f"{o.status} held={o.detail.get('held_s')}")

# 偶发单点丢失（间隔4秒）仍应算连续 —— 不能过度严格
h = hist([(t, {"hr": 52.0}) for t in range(990, 1010, 4)])
o = rule_bradycardia({"t": 1010.0, "hr": 52.0}, FakeBaseline(hr=78.0), h, cfg)
check("偶发丢点(间隔4s)仍算连续", o.status == MET,
      f"{o.status} held={o.detail.get('held_s')}")

# 陈旧数据不得当作"正在持续"：最近样本离现在 100 秒
h = hist([(t, {"hr": 52.0}) for t in range(900, 920, 2)])
o = rule_bradycardia({"t": 1020.0, "hr": 52.0}, FakeBaseline(hr=78.0), h, cfg)
check("陈旧低心率段不得当作正在持续", o.status != MET,
      f"{o.status} held={o.detail.get('held_s')}")

# 回归（实测踩坑）：Dravet 患儿睡眠心率可达中位 130bpm（SeizeIT2 sub-046）。
# 曾用"基线×0.75"做相对门限 → 门限 97bpm → 整夜正常心率全判为心动过缓，
# 规则每夜命中 500+ 次，误报 1.20→3.80/h。现在只用绝对 60。
h = hist([(t, {"hr": 128.0}) for t in range(960, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 128.0}, FakeBaseline(hr=130.0), h, cfg)
check("高基线患儿(130bpm)的正常心率不得判为心动过缓", o.status == UNMET,
      f"{o.status} thr={o.detail.get('threshold')}")

h = hist([(t, {"hr": 75.0}) for t in range(1000, 1010, 2)])
o = rule_bradycardia({"t": 1010.0, "hr": 75.0}, FakeBaseline(hr=78.0), h, cfg)
check("心率正常 → unmet", o.status == UNMET, o.status)

# 回归：心率下降在第七棒是零分的，确认融合层确实不管这个方向
sig = wearable_signals({"t": 1010.0, "hr": 45.0, "motion": 0.1},
                       FakeBaseline(hr=78.0, motion=0.3, spo2=97.0), [])
hr_doubt = next((s["doubt"] for s in sig if s["key"] == "hr_surge"), 0)
check("融合层对心率骤降确实给 0 分（故需规则兜底）", hr_doubt == 0, str(hr_doubt))

# ------------------------------------------------------------------ #
print("== R3 呼吸暂停：呼吸趋零持续≥10 秒 ==")

b = FakeBaseline(breathing=18.0)
h = hist([(t, {"breathing": 2.0}) for t in range(988, 1010, 2)])
o = rule_apnea({"t": 1010.0, "breathing": 2.0}, b, h, cfg)
check("呼吸停止24秒 → met", o.status == MET, f"{o.status} held={o.detail.get('held_s')}")
check("10-30秒暂停要求升级而非直接唤醒", o.demand == "ESCALATE", str(o.demand))

h = hist([(t, {"breathing": 2.0}) for t in range(960, 1010, 2)])
o = rule_apnea({"t": 1010.0, "breathing": 2.0}, b, h, cfg)
check("呼吸停止超30秒 → 直接 CRITICAL", o.demand == "CRITICAL", str(o.demand))

o = rule_apnea({"t": 1010.0, "breathing": 17.0}, b, [], cfg)
check("呼吸正常 → unmet", o.status == UNMET, o.status)

w = Window.build(t=1010.0, caps=Channels(hr=True, motion=True), hr=80.0, motion=0.2)
o = rule_apnea(w.as_features(), b, [], cfg)
check("无呼吸通道 → partial 且点明医生列为刚需",
      o.status == PARTIAL and "breathing" in o.missing, f"{o.status} {o.missing}")

# ------------------------------------------------------------------ #
print("== R4 血氧：延迟但高准确 → 做升级确认器 ==")

b = FakeBaseline(spo2=97.0)
# 此前有异常（risk 抬头）+ 血氧掉 5 点 → 确认缺氧，直接 CRITICAL
h = hist([(t, {"risk": 0.5}) for t in range(960, 1010, 2)])
o = rule_desat_confirm({"t": 1010.0, "spo2": 92.0}, b, h, cfg)
check("既往有异常 + 血氧降5点 → CRITICAL",
      o.status == MET and o.demand == "CRITICAL", f"{o.status} {o.demand}")

# 孤立的小幅血氧下降（无既往异常）→ 不升级，只待观察
h = hist([(t, {"risk": 0.05}) for t in range(960, 1010, 2)])
o = rule_desat_confirm({"t": 1010.0, "spo2": 92.0}, b, h, cfg)
check("孤立降5点无既往异常 → partial 待观察", o.status == PARTIAL, f"{o.status} {o.demand}")

# 孤立但降幅极大（-10）→ 仍要升级
o = rule_desat_confirm({"t": 1010.0, "spo2": 87.0}, b, h, cfg)
check("孤立但降10点 → 仍升级", o.status == MET and o.demand == "ESCALATE",
      f"{o.status} {o.demand}")

# 延迟性：血氧比心率晚 60 秒才掉，仍要能关联上（context 窗 90 秒）
h = hist([(t, {"risk": 0.6}) for t in range(940, 960, 2)])   # 50-70 秒前的异常
o = rule_desat_confirm({"t": 1010.0, "spo2": 92.0}, b, h, cfg)
check("血氧滞后60秒仍能关联既往异常（延迟性）",
      o.status == MET and o.demand == "CRITICAL", f"{o.status} {o.demand}")

w = Window.build(t=1010.0, caps=Channels(hr=True), hr=80.0)
o = rule_desat_confirm(w.as_features(), b, h, cfg)
check("无血氧通道 → partial 且声明 missing",
      o.status == PARTIAL and "spo2" in o.missing, f"{o.status} {o.missing}")

# ------------------------------------------------------------------ #
print("== R5 癫痫持续状态：异常持续超 2 分钟必报 ==")

h = hist([(t, {"risk": 0.5}) for t in range(880, 1010, 2)])   # 130 秒
o = rule_prolonged_event({"t": 1010.0}, FakeBaseline(), h, cfg)
check("异常持续130秒 → met + CRITICAL",
      o.status == MET and o.demand == "CRITICAL", f"{o.status} {o.demand}")
check("说明里给出分钟数", o.note and "分钟" in o.note, str(o.note))

h = hist([(t, {"risk": 0.5}) for t in range(980, 1010, 2)])   # 30 秒
o = rule_prolonged_event({"t": 1010.0}, FakeBaseline(), h, cfg)
check("异常仅30秒 → 不触发持续状态", o.status != MET, o.status)

# ------------------------------------------------------------------ #
print("== 分层语义：只升不降 ==")

base_res = GateResult("CRITICAL", 0.9, True, "hr_surge", None, [], [])
from engine.clinical import RuleOutcome
out = apply_clinical(base_res, [RuleOutcome("x", UNMET, {}, demand=None)], GateConfig())
check("规则全不命中也不能把 CRITICAL 降下来", out.level == "CRITICAL", out.level)

log_res = GateResult("LOG", 0.1, False, None, None, [], [])
out = apply_clinical(log_res, [RuleOutcome("apnea", MET, {}, note="呼吸停了",
                                           demand="CRITICAL")], GateConfig())
check("LOG 遇 CRITICAL 诉求 → 升到 CRITICAL", out.level == "CRITICAL", out.level)
check("升级后标记已确认", out.confirmed is True, str(out.confirmed))
check("家长可见说明写进 evidence", "呼吸停了" in out.evidence, str(out.evidence))

warn_res = GateResult("WARNING", 0.6, False, "hr_surge", None, [], [])
out = apply_clinical(warn_res, [RuleOutcome("hr_surge_sustained", MET, {},
                                            note="持续45秒", demand="ESCALATE")],
                     GateConfig())
check("WARNING + ESCALATE → CRITICAL", out.level == "CRITICAL", out.level)

log_res = GateResult("LOG", 0.2, False, None, None, [], [])
out = apply_clinical(log_res, [RuleOutcome("post_ictal_window", MET, {},
                                           note="高危期", demand="ESCALATE")],
                     GateConfig())
check("LOG + ESCALATE → 只到 WARNING（不跳级唤醒）", out.level == "WARNING", out.level)

# 发作后高危期只报告状态，不要求升级 —— 否则 600 秒内每窗都被钉成 WARNING。
outs = evaluate_rules({"t": 1000.0, "hr": 80.0, "_last_critical_t": 900.0},
                      FakeBaseline(hr=78.0), [], cfg)
pi = [o for o in outs if o.key == "post_ictal_window"][0]
check("高危期判据命中但不索要升级（靠下调门限实现高敏）",
      pi.status == MET and pi.demand is None, f"{pi.status}/{pi.demand}")

# ------------------------------------------------------------------ #
print("== 分层不得反噬下层：基线学习与 streak 用风险分层的级别 ==")

# 实测教训（diag_sens_drop.py）：高危期把 300 窗抬成 WARNING，
# 而 baseline.observe 只在 level=="LOG" 时执行 → 基线整十分钟停学，
# sub-110 紧接着的真发作风险分 0.610→0.176，灵敏度 88%→82%。
clin_t = ClinicalConfig.for_hop(2.0)
ng = NightGate(config=GateConfig(clinical_enabled=True, clinical=clin_t),
               signal_fn=wearable_signals)
ng.last_critical_t = 1000.0          # 伪造"刚确认过一次发作"
for i in range(40):                  # 高危期内喂 40 窗平静数据
    t = 1002.0 + i * 2
    w = Window.build(t=t, caps=Channels.wearable_full(), hr=78.0, motion=0.02,
                     breathing=16.0, spo2=98.0)
    ng.step(w.as_features())
ready_after = ng.baseline.ready
check("发作后高危期内基线仍在学习（不被临床层冻结）", ready_after is True,
      f"ready={ready_after}")
check("高危期内 streak 不被临床层顶满", ng.streak == 0, str(ng.streak))

# ------------------------------------------------------------------ #
print("== 判据覆盖表：缺通道必须显式声明 ==")

w = Window.build(t=1010.0, caps=Channels.seizeit2(), hr=80.0, motion=0.3)
outs = evaluate_rules(w.as_features(), FakeBaseline(hr=78.0), [], cfg)
cov = coverage_report(outs)
check("SeizeIT2(无血氧无呼吸) 覆盖表点出两路缺失",
      set(cov["blocked_by_missing_channels"]) >= {"spo2", "breathing"},
      str(cov["blocked_by_missing_channels"]))
check("覆盖表给出可判条数", cov["judgeable"] < cov["total"],
      f"{cov['judgeable']}/{cov['total']}")

# ------------------------------------------------------------------ #
print("== 三态通道语义 ==")

caps = Channels.wearable_full()
w = Window.build(t=1.0, caps=caps, hr=80.0, motion=0.2, spo2=None,
                 breathing=16.0, temp=36.8)
check("有值 → present", w.status("hr") == PRESENT, w.status("hr"))
check("通道在但本窗无值 → dropout", w.status("spo2") == DROPOUT, w.status("spo2"))

w2 = Window.build(t=1.0, caps=Channels.seizeit2(), hr=80.0, motion=0.2, spo2=95.0)
check("通道不存在 → absent", w2.status("spo2") == ABSENT, w2.status("spo2"))
check("通道不存在时即便传了值也不采信", w2.get("spo2") is None, str(w2.get("spo2")))
check("declared_missing 列出设备没有的通道",
      "spo2" in w2.declared_missing(), str(w2.declared_missing()))

w3 = Window.build(t=1.0, caps=Channels.seizeit2())
check("该有的全没取到 → is_blind", w3.is_blind() is True, str(w3.is_blind()))
w4 = Window.build(t=1.0, caps=Channels.seizeit2(), hr=80.0)
check("还有一路就不算盲", w4.is_blind() is False, str(w4.is_blind()))

# ------------------------------------------------------------------ #
print("== 端到端：NightGate 带临床层不崩且能兜住心动过缓 ==")

ng = NightGate(config=GateConfig(clinical_enabled=True), signal_fn=wearable_signals)
caps = Channels.wearable_full()
res = None
for i in range(40):                      # 先建基线（正常夜）
    f = Window.build(t=i * 2.0, caps=caps, hr=78.0, motion=0.2,
                     spo2=97.0, breathing=17.0, temp=36.8).as_features()
    res = ng.step(f)
check("正常段不报警", res.level == "LOG", res.level)
check("正常段覆盖表存在（四路齐全）",
      res.coverage is not None and not res.coverage["blocked_by_missing_channels"],
      str(res.coverage))

for i in range(40, 50):                  # 心率掉到 50 —— 第七棒会完全静默
    f = Window.build(t=i * 2.0, caps=caps, hr=50.0, motion=0.1,
                     spo2=97.0, breathing=17.0, temp=36.8).as_features()
    res = ng.step(f)
check("心动过缓被临床层兜住（原本静默漏报）", res.level != "LOG", res.level)
check("命中的是 bradycardia 规则", "bradycardia" in ng.clinical_hits,
      str(ng.clinical_hits))

# 关掉临床层 → 回到第七棒行为，确认这条确实是新增能力而非原本就有
ng2 = NightGate(config=GateConfig(clinical_enabled=False), signal_fn=wearable_signals)
res2 = None
for i in range(50):
    hr = 78.0 if i < 40 else 50.0
    f = Window.build(t=i * 2.0, caps=caps, hr=hr, motion=0.1 if i >= 40 else 0.2,
                     spo2=97.0, breathing=17.0, temp=36.8).as_features()
    res2 = ng2.step(f)
check("关掉临床层后心动过缓确实静默（证明是新增能力）", res2.level == "LOG", res2.level)

print("\n" + "=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
