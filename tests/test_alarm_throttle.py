#!/usr/bin/env python3
"""第九棒：响铃节流的语义约束。

来历：418 次误报里 62.7% 彼此间隔 <5 分钟（成簇），而真发作是孤立事件。
节流把同一次生理事件的余波合并掉，实测 3.05 → 1.90/h 且灵敏度零代价。

这组测试锁住的**不是**误报率（那是验证脚本的事），而是语义边界：
节流只许改 notify，绝不许改 level、不许少写日志、不许影响基线学习。
一旦有人把节流"顺手"做成降级，漏报通道就又开了一条。
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


def always_critical(features, baseline, history):
    """把每个窗都推成满风险，方便只测节流本身。

    必须给多路信号：单路 motion_burst 的 influence 上限是 0.40，
    noisy-OR 后到不了 critical_threshold，一路信号永远测不出 CRITICAL。
    三路满格 → 1-(0.6·0.55·0.65) ≈ 0.79。
    """
    return [{"key": "motion_burst", "doubt": 1.0, "detail": {}, "note": "测试用"},
            {"key": "motion_rhythm", "doubt": 1.0, "detail": {}, "note": None},
            {"key": "hr_surge", "doubt": 1.0, "detail": {}, "note": None}]


def run(throttle_s, n=12, step=30, **kw):
    # confirm_windows=1：不让"多窗确认"把前几窗吃掉，本测试只关心节流。
    cfg = GateConfig(alarm_throttle_s=throttle_s, clinical_enabled=False,
                     critical_threshold=0.70, warning_threshold=0.40,
                     confirm_windows=1, min_baseline_windows=1, **kw)
    ng = NightGate(config=cfg, signal_fn=always_critical)
    out = []
    for i in range(n):
        r = ng.step({"t": i * step, "motion": 5.0, "hr": 100.0})
        out.append(r)
    return ng, out


print("=" * 60)
print("响铃节流：只压通知，不压判定")
print("=" * 60)

ng, out = run(120.0, n=12, step=30)
crit = [r for r in out if r.level == "CRITICAL"]
rung = [r for r in crit if r.notify]
check("判定级别不受节流影响（CRITICAL 照常产生）", len(crit) >= 8,
      f"→ CRITICAL {len(crit)}/12")
check("30s 步进 + 120s 节流 → 响铃次数显著少于判定次数",
      0 < len(rung) < len(crit), f"→ 响铃 {len(rung)} / 判定 {len(crit)}")
check("第一次 CRITICAL 必须响铃（绝不能把首次报警节流掉）",
      crit[0].notify is True)
check("被节流的结果带 throttled_for_s（可追溯）",
      all(r.throttled_for_s is not None for r in crit if not r.notify))
check("节流计数器可观测", ng.throttled == len(crit) - len(rung),
      f"→ {ng.throttled} vs {len(crit) - len(rung)}")

# 响铃间隔必须 >= 节流窗
times = [i * 30 for i, r in enumerate(out) if r.level == "CRITICAL" and r.notify]
gaps = [b - a for a, b in zip(times, times[1:])]
check("相邻两次响铃间隔 >= 120s", all(g >= 120 for g in gaps), f"→ {gaps}")

# --- 日志不得因节流而丢 -------------------------------------------
ng_t, out_t = run(120.0, n=12, step=30)
ng_n, out_n = run(0.0, n=12, step=30)
check("节流不减少日志条数（降级≠丢弃）",
      len(ng_t.journal.events) == len(ng_n.journal.events),
      f"→ 节流 {len(ng_t.journal.events)} vs 关闭 {len(ng_n.journal.events)}")

# --- 关闭开关可完全复原旧行为 -------------------------------------
# 要连 ring_collapse 一起关：它是**独立**机制（管"一个事件响几声"），
# 不受 alarm_throttle_s 开关连坐，否则 throttle=0 时仍只响一声。
# 见 tests/test_ring_collapse.py。
ng_n0, out_n0 = run(0.0, n=12, step=30, ring_collapse=False)
check("alarm_throttle_s=0 且 ring_collapse=False → 全部响铃（可一键回退对照）",
      all(r.notify for r in out_n0 if r.level == "CRITICAL"))
check("两个开关都关时 throttled 计数为 0", ng_n0.throttled == 0,
      f"→ throttled={ng_n0.throttled}")
# 只关节流、留着塌缩：仍然只响一声（锁住"互相独立"这个语义）
check("只关节流时塌缩仍生效（两机制互相独立）",
      sum(1 for r in out_n if r.level == "CRITICAL" and r.notify) == 1,
      f"→ 响了 {sum(1 for r in out_n if r.level == 'CRITICAL' and r.notify)} 声")

# --- 一次持续事件只响一次（不应期的核心语义）----------------------
# 20 窗全程维持 CRITICAL = 生理上**一次**长事件，就该只响一次。
# 纯时间节流会每 120s 放行一次，把一次事件记成 5~6 次 —— 那是错的。
_, out_long = run(120.0, n=20, step=30)
rung_long = [r for r in out_long if r.level == "CRITICAL" and r.notify]
check("全程不降级的一次长事件只响一次", len(rung_long) == 1,
      f"→ 响了 {len(rung_long)} 次")


def run_two_events(throttle_s, gap_windows, quiet_windows=4):
    """两段真 CRITICAL 事件，中间插入安静窗（级别降下来）。
    这是"新事件"的定义：必须先离开 CRITICAL。"""
    calls = {"i": 0}

    def sig(features, baseline, history):
        quiet = features.get("_quiet")
        d = 0.0 if quiet else 1.0
        return [{"key": "motion_burst", "doubt": d, "detail": {}, "note": None},
                {"key": "motion_rhythm", "doubt": d, "detail": {}, "note": None},
                {"key": "hr_surge", "doubt": d, "detail": {}, "note": None}]

    cfg = GateConfig(alarm_throttle_s=throttle_s, clinical_enabled=False,
                     critical_threshold=0.70, warning_threshold=0.40,
                     confirm_windows=1, min_baseline_windows=1)
    ng = NightGate(config=cfg, signal_fn=sig)
    out = []
    plan = ([False] * 3 + [True] * quiet_windows + [False] * 3)
    for i, quiet in enumerate(plan):
        r = ng.step({"t": i * gap_windows, "motion": 1.0 if quiet else 5.0,
                     "hr": 70.0 if quiet else 100.0, "_quiet": quiet})
        out.append(r)
    return out


# 安静段够长（4 窗 × 60s = 240s > 120s 节流窗）→ 第二段是新事件，必须响
out2 = run_two_events(120.0, gap_windows=60, quiet_windows=4)
rung2 = [i for i, r in enumerate(out2)
         if r.level == "CRITICAL" and r.notify]
check("中间降级且冷却够久 → 第二次事件重新响铃", len(rung2) == 2,
      f"→ 响铃窗序号 {rung2}")

# 短暂降级后的余波**会**再响一次 —— 这是明确的取舍，不是漏洞。
#
# 要压住这种余波，就得让 event_peak_risk 跨冷却窗保留；实测那样做
# 会把"紧跟体动簇之后的真发作"一起压掉：灵敏度 17/17 → 15/17，
# 换来响铃窗 87.9 → 71.6/h。红线是灵敏度，不拿它换安静。
# 所以这里断言的是"宁可多响一次"，并把代价记在案上。
out3 = run_two_events(120.0, gap_windows=30, quiet_windows=1)
rung3 = [r for r in out3 if r.level == "CRITICAL" and r.notify]
check("短暂降级后的余波允许再响（宁可多响不可漏报）", len(rung3) == 2,
      f"→ 响了 {len(rung3)} 次")

# ---------------------------------------------------------------------
# 响铃风暴：这组是事后补的回归锁，对应生产路径复测时才暴露的三个坑。
#
# 为什么原来这个文件没抓住：上面的用例全都用"干净的两段式信号"
# （3 窗吵 → N 窗静 → 3 窗吵），而真实信号是在阈值附近**反复抖动**的。
# 抖动会让 fresh / 击穿 反复重新成立，于是 120s 节流形同不存在：
# 实测 sub-024 一夜响铃 2846 次（233/h），而按事件合并只显示 9 个
# 误报事件 —— **事件级指标把响铃风暴完全藏住了**。
# 所以这里直接断言"响铃次数"，并且用抖动信号而不是干净信号。
# ---------------------------------------------------------------------


def run_flapping(n=300, period=2, throttle=120.0, min_gap=30.0,
                 clinical=False, episode=False):
    """信号在 CRITICAL 阈值上下每 `period` 窗抖一次，持续 n 窗。

    窗间隔 2s（和生产路径一致），所以 n=300 ≈ 10 分钟。
    """
    def sig(features, baseline, history):
        d = 1.0 if features.get("_hot") else 0.0
        return [{"key": "motion_burst", "doubt": d, "detail": {}, "note": None},
                {"key": "motion_rhythm", "doubt": d, "detail": {}, "note": None},
                {"key": "hr_surge", "doubt": d, "detail": {}, "note": None}]

    cfg = GateConfig(alarm_throttle_s=throttle, alarm_min_gap_s=min_gap,
                     clinical_enabled=clinical, critical_threshold=0.70,
                     warning_threshold=0.40, confirm_windows=1,
                     min_baseline_windows=1,
                     alarm_episode_enabled=episode)
    ng = NightGate(config=cfg, signal_fn=sig)
    rings = 0
    for i in range(n):
        hot = (i // period) % 2 == 0
        r = ng.step({"t": i * 2.0, "motion": 5.0 if hot else 1.0,
                     "hr": 100.0 if hot else 70.0, "_hot": hot})
        if r.level == "CRITICAL" and r.notify:
            rings += 1
    run_flapping.last_gate = ng          # 给可观测性用例看计数器
    return rings


# 10 分钟抖动信号，响铃下限 30s → 理论上限 ~20 次。
# 松一点给 25，留出首响和边界。改前这里是 150 次（每个 hot 窗都响）。
rings_flap = run_flapping()
check("抖动信号不产生响铃风暴（下限生效）", rings_flap <= 25,
      f"→ 10 分钟抖动响了 {rings_flap} 次")

# 关掉下限 → 回到风暴，证明上面那条确实是下限的功劳，不是别的东西顺手压住的
#
# 这组对照必须关掉发作期封顶（episode=False，即两臂都关）：
# 封顶本身就能把抖动压到 2 次，开着它两臂都是 2，
# 对照就测不出"下限"的独立贡献了。封顶的效果另有专门用例（见文件末尾）。
rings_nogap = run_flapping(min_gap=0.0)
check("关掉响铃下限时抖动确实会风暴（对照）", rings_nogap > rings_flap * 2,
      f"→ 无下限 {rings_nogap} 次 vs 有下限 {rings_flap} 次")

# ---- 发作期封顶（episode cap）---------------------------------------
# 这是上面那层之外的**总数**约束：下限管间隔，封顶管一次事件响几声。
# 生产实测的问题是 1198 次响铃只合并成 12 个事件（每事件 ~100 声），
# 下限只能把间隔拉开、压不住总数。
rings_cap = run_flapping(episode=True)
check("发作期封顶把抖动压到上限内",
      rings_cap <= GateConfig().episode_max_rings,
      f"→ 封顶 {GateConfig().episode_max_rings} 声，实响 {rings_cap} 次")
check("封顶比单靠下限更安静（对照）", rings_cap < rings_flap,
      f"→ 封顶 {rings_cap} 次 vs 仅下限 {rings_flap} 次")

# 封顶**不能**吃掉"安静够久之后的新事件"——那是新发作，必须能响。
# 构造：先抖一阵（吃掉封顶额度），再长时间安静（超过 episode_clear_s），
# 然后再来一次 CRITICAL。第三段必须响。


def run_cap_then_new_event():
    def sig(features, baseline, history):
        d = 1.0 if features.get("_hot") else 0.0
        return [{"key": "motion_burst", "doubt": d, "detail": {}, "note": None},
                {"key": "motion_rhythm", "doubt": d, "detail": {}, "note": None},
                {"key": "hr_surge", "doubt": d, "detail": {}, "note": None}]

    cfg = GateConfig(alarm_throttle_s=120.0, alarm_min_gap_s=30.0,
                     clinical_enabled=False, critical_threshold=0.70,
                     warning_threshold=0.40, confirm_windows=1,
                     min_baseline_windows=1, alarm_episode_enabled=True)
    ng = NightGate(config=cfg, signal_fn=sig)
    t = 0.0
    phase1 = phase3 = 0
    # 第一段：60 窗热信号，吃满封顶
    for _ in range(60):
        r = ng.step({"t": t, "motion": 5.0, "hr": 100.0, "_hot": True})
        if r.level == "CRITICAL" and r.notify:
            phase1 += 1
        t += 2.0
    # 第二段：安静，时长 > episode_clear_s(600s) → 发作期结束
    for _ in range(400):
        ng.step({"t": t, "motion": 1.0, "hr": 70.0, "_hot": False})
        t += 2.0
    # 第三段：新的 CRITICAL，必须响
    for _ in range(30):
        r = ng.step({"t": t, "motion": 5.0, "hr": 100.0, "_hot": True})
        if r.level == "CRITICAL" and r.notify:
            phase3 += 1
        t += 2.0
    return phase1, phase3, ng


p1, p3, ng_cap = run_cap_then_new_event()
check("封顶后：安静够久的新事件仍然响铃（不丢发作）", p3 >= 1,
      f"→ 第一段响 {p1} 次（封顶 2），安静 800s 后新事件响 {p3} 次")
check("封顶后新事件另起一个发作期（计数可观测）", ng_cap.episodes >= 2,
      f"→ 发作期 {ng_cap.episodes} 个（第一段 + 安静后的新事件）")

# episode_capped 要在**封顶真正起作用**的场景里查：
# 上面 run_cap_then_new_event 里下限先挡住了那些响铃（capped 永远轮不到），
# 所以关掉下限、只留封顶，这时被压掉的就全记在 episode_capped 上。
rings_caponly = run_flapping(min_gap=0.0, episode=True)
ng_caponly = run_flapping.last_gate
check("仅靠封顶也能压住风暴（下限关闭）",
      rings_caponly <= GateConfig().episode_max_rings,
      f"→ 无下限纯封顶响 {rings_caponly} 次")
check("被封顶压掉的声数有计数（可观测）", ng_caponly.episode_capped > 0,
      f"→ 压掉 {ng_caponly.episode_capped} 声，发作期 {ng_caponly.episodes} 个")


# ---- 封顶最危险的失效模式：发作前的误报吃掉额度 --------------------
# 实测踩过（run_production_target.py 第一版）：灵敏度 22/22 → 15/22。
# 机制：患者躁动 → 误报响满额度 → 真发作紧接着来 → 被封顶静默；
# 而患者一直不安静，发作期永不过期，于是整次发作都在静默里。
#
# 现在靠"升级豁免"挡住：风险突破**发作期峰值** + margin 一定放行。
# 这条用例就是钉住这个行为，防止以后有人把豁免去掉。
def run_fa_then_seizure():
    """低强度躁动响满额度后，紧跟一次更强的事件（模拟真发作）。"""
    state = {"risk": 0.0}

    def sig(features, baseline, history):
        d = features.get("_d", 0.0)
        return [{"key": "motion_burst", "doubt": d, "detail": {}, "note": None},
                {"key": "motion_rhythm", "doubt": d, "detail": {}, "note": None},
                {"key": "hr_surge", "doubt": d, "detail": {}, "note": None}]

    cfg = GateConfig(alarm_throttle_s=120.0, alarm_min_gap_s=30.0,
                     clinical_enabled=False, critical_threshold=0.40,
                     warning_threshold=0.20, confirm_windows=1,
                     min_baseline_windows=1, alarm_episode_enabled=True,
                     episode_max_rings=2, episode_clear_s=600.0)
    ng = NightGate(config=cfg, signal_fn=sig)
    t = 0.0
    fa = 0
    # 第一段：中等强度躁动 5 分钟，吃满封顶额度
    for _ in range(150):
        r = ng.step({"t": t, "motion": 3.0, "hr": 90.0, "_d": 0.55})
        if r.level == "CRITICAL" and r.notify:
            fa += 1
        t += 2.0
    # 第二段：**不给安静期**，直接升级成强信号（真发作紧跟误报）
    sz = 0
    for _ in range(40):
        r = ng.step({"t": t, "motion": 9.0, "hr": 150.0, "_d": 1.0})
        if r.level == "CRITICAL" and r.notify:
            sz += 1
        t += 2.0
    return fa, sz, ng


fa_n, sz_n, ng_fa = run_fa_then_seizure()
check("发作前误报吃满额度后，升级的真发作仍然响铃", sz_n >= 1,
      f"→ 躁动段响 {fa_n} 声（封顶 2），紧跟的升级事件响 {sz_n} 声"
      f"（必须 ≥1，否则就是漏报）")
check("升级豁免没把封顶架空（躁动段仍被压住）", fa_n <= 4,
      f"→ 躁动段只响 {fa_n} 声")


def run_sustained_rule(n=300, min_gap=30.0):
    """临床规则**持续成立**数分钟（hr 一直高），看会不会每窗都击穿。

    坑的原形：clinical_met 原先是电平触发（本窗 met 就击穿），
    而 hr_surge_sustained 这类判据按设计会连续成立好几分钟，
    于是 120s 节流被逐窗绕过。
    """
    def sig(features, baseline, history):
        return [{"key": "motion_burst", "doubt": 1.0, "detail": {}, "note": None},
                {"key": "motion_rhythm", "doubt": 1.0, "detail": {}, "note": None},
                {"key": "hr_surge", "doubt": 1.0, "detail": {}, "note": None}]

    cfg = GateConfig(alarm_throttle_s=120.0, alarm_min_gap_s=min_gap,
                     clinical_enabled=True, critical_threshold=0.70,
                     warning_threshold=0.40, confirm_windows=1,
                     min_baseline_windows=1)
    ng = NightGate(config=cfg, signal_fn=sig)
    rings = 0
    for i in range(n):
        # 心率全程高位 → hr_surge_sustained 一旦成立就持续成立
        r = ng.step({"t": i * 2.0, "motion": 5.0, "hr": 150.0})
        if r.level == "CRITICAL" and r.notify:
            rings += 1
    return rings


rings_sust = run_sustained_rule()
check("临床规则持续成立不逐窗击穿（边沿触发）", rings_sust <= 25,
      f"→ 10 分钟持续成立响了 {rings_sust} 次")

# 下限只"推迟"不"取消"：被挡住的响铃必须补响，否则就是开漏报通道。
# 构造：一次孤立事件，响第一次后立刻又有更强证据（撞上下限），
# 等下限到期必须补响，不能静默吞掉。


def run_pending():
    seq = {"i": 0}

    def sig(features, baseline, history):
        d = features.get("_d", 0.0)
        return [{"key": "motion_burst", "doubt": d, "detail": {}, "note": None},
                {"key": "motion_rhythm", "doubt": d, "detail": {}, "note": None},
                {"key": "hr_surge", "doubt": d, "detail": {}, "note": None}]

    cfg = GateConfig(alarm_throttle_s=120.0, alarm_min_gap_s=30.0,
                     clinical_enabled=False, critical_threshold=0.70,
                     warning_threshold=0.40, confirm_windows=1,
                     min_baseline_windows=1)
    ng = NightGate(config=cfg, signal_fn=sig)
    out = []
    # 0-2 窗热（首响）→ 3 窗静 → 4 窗热（撞下限，挂起）→ 之后一直热
    plan = [1.0] * 3 + [0.0] + [1.0] * 30
    for i, d in enumerate(plan):
        r = ng.step({"t": i * 2.0, "motion": 5.0 if d else 1.0,
                     "hr": 100.0 if d else 70.0, "_d": d})
        out.append((i * 2.0, r))
    return ng, out


ng_p, out_p = run_pending()
ring_ts = [t for t, r in out_p if r.level == "CRITICAL" and r.notify]
# 首响在 t=4 附近；下限 30s → 补响应落在 t>=34 左右，且必须发生
check("被下限挡住的响铃会补响（推迟不取消）",
      len(ring_ts) >= 2 and ring_ts[1] - ring_ts[0] >= 30.0,
      f"→ 响铃时刻 {ring_ts[:4]}")
check("补响有计数（可观测）", ng_p.pending_rings_fired >= 1,
      f"→ pending_rings_fired={ng_p.pending_rings_fired}")

print()
print("=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
