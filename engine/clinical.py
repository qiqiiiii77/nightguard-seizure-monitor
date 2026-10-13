"""
clinical.py — 医生访谈判据的直接编码(第八棒,框架层)

来源:`~/Desktop/夜间癫痫发作监测/医学背景的一些知识.docx`
(田医生访谈,Dravet 综合征夜间监护)

为什么要单独一层
================
第七棒的引擎是**扁平投票**:八路信号各出一个 doubt,noisy-OR 合成一个风险分,
过阈值就报警。但医生的判断模型不是这样,它是**有结构的**:

  · 心率升 30-60bpm **且持续 >45 秒** → 这是一条带时长的合取判据,
    不是"心率升高"这一个瞬时分数
  · 血氧下降**延迟 50-60 秒才出现,但准确率极高** → 它不该和心率同权投票,
    它是**升级确认器**:别的信号先报,血氧一掉就从 WARNING 跳 CRITICAL
  · 发作**超过 2 分钟必须提醒** → 这是独立于风险分的硬性规则
  · 心率**持续低于 60** → SUDEP 前兆,和"心率升高"是**反方向**的独立通道
  · 发作后 **5-10 分钟**高危期 → 时间尺度比第七棒的 12 秒长两个数量级

所以这一层不改融合逻辑,而是在它**之上**叠加显式的临床规则:
每条规则有名字、有出处、有"满足/部分满足/不满足"三态结论,
并能说出"因为缺血氧通道,这条判据只能部分判定"。

这同时解决了可解释性:给家长和医生看的不是"风险 0.73",
而是"心率比平时高 38,已持续 52 秒(医生判据:升30-60持续45秒)"。

设计约束
========
1. **纯 numpy/stdlib**,不引入 NeuroKit2/YASA 等依赖 —— 这个引擎要能跑在
   腕表/树莓派上,现有代码零依赖是资产,不要为了用库而破坏它。
   (NeuroKit2 的价值在离线特征提取和验证侧,不在实时判定侧。)
2. 规则只读 history 和 baseline,**不改**它们 —— 副作用全留在 NightGate。
3. 任何规则缺少必需通道时返回 PARTIAL + 原因,**绝不**静默当作不满足。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .schema import Window, features_window, caps_of, ABSENT

MET = "met"             # 判据满足
PARTIAL = "partial"     # 条件部分满足,或缺通道无法完整判定
UNMET = "unmet"         # 判据不满足


@dataclass
class RuleOutcome:
    key: str
    status: str                      # MET | PARTIAL | UNMET
    detail: Dict[str, Any]
    note: Optional[str] = None       # 给人看的一句话(中文,直接可上报告)
    missing: tuple = ()              # 因缺哪些通道而无法完整判定
    # 规则对分级的**硬性**诉求。None = 不干预,由风险分决定。
    demand: Optional[str] = None     # "CRITICAL" | "ESCALATE" | None
    source: str = ""                 # 医学出处,保证可追溯


# --------------------------------------------------------------------- #
# 工具:在 history 上按秒回溯
# --------------------------------------------------------------------- #

def _span(history, t, lo_s, hi_s, key):
    """取 [t-hi_s, t-lo_s] 秒区间内 key 的非空值列表(含时间戳)。"""
    if not isinstance(t, (int, float)):
        return []
    out = []
    for h in history:
        ht = h.get("t")
        if not isinstance(ht, (int, float)):
            continue
        age = t - ht
        if lo_s <= age <= hi_s:
            v = h.get(key)
            if isinstance(v, (int, float)) and v == v:
                out.append((ht, float(v)))
    return out


def _sustained_for(history, t, key, predicate, max_look_s=300.0,
                   max_gap_s=6.0):
    """从当前时刻往回数,predicate **连续**成立了多少秒。

    返回 (秒数, 样本数)。两种情况会中断连续性:
      1. 出现一个 predicate 不成立的样本 —— "持续"的本义
      2. 相邻两个样本之间间隔超过 max_gap_s —— **缺值也中断连续性**

    第 2 条是实测补的(validation/diag_brady2.py)。原实现只遍历非空样本,
    于是 t=1210 的一个低心率和 t=62788 的另一个低心率被算成
    "连续 61578 秒的心动过缓" —— 中间整夜的缺值被无声跳过了。
    强直-阵挛的肌电干扰本就会毁掉 ECG,缺值恰恰集中在发作期,
    所以这个 bug 在最关键的时刻最严重。

    max_gap_s 默认 6 秒 ≈ 3 个 2 秒窗:容忍偶发单点丢失,
    但不容忍"段内大半是空的"。真实的生理持续必须有连续采样支撑;
    拿不到数据时的正确回答是"我看不见"(看门狗的职责),不是"它持续着"。
    """
    pts = _span(history, t, 0.0, max_look_s, key)
    if not pts:
        return 0.0, 0
    pts.sort(key=lambda p: -p[0])        # 由近及远
    newest = pts[0][0]
    # 锚定当前时刻:最近一个样本离现在太远,说明"现在"本身就没数据,
    # 不能拿一段陈旧的低心率当作"正在持续"。
    if isinstance(t, (int, float)) and (t - newest) > max_gap_s:
        return 0.0, 0
    oldest_ok, n, prev_t = newest, 0, None
    for ht, v in pts:
        if prev_t is not None and (prev_t - ht) > max_gap_s:
            break                        # 采样缺口过大 → 连续性断裂
        if not predicate(v):
            break
        oldest_ok, n, prev_t = ht, n + 1, ht
    return max(0.0, newest - oldest_ok), n


# --------------------------------------------------------------------- #
# R1  大发作主判据:心率升 30-60bpm 且持续 >45 秒
# --------------------------------------------------------------------- #

def rule_hr_surge_sustained(features, baseline, history, cfg) -> RuleOutcome:
    """医生原话:「心率增高30至60次/分钟且持续时间超过45秒作为大发作的报警阈值」。

    第七棒只有瞬时升幅门限,持续性靠 risk 中位数间接体现,
    而且扫描只到 16 秒 —— 等于从未在这条判据上测过。
    """
    src = "医生访谈:强直阵挛期心率比基线瞬时增高30-60bpm;建议升高30-60且持续>45s报警"
    t = features.get("t")
    hr = features.get("hr")
    w = features_window(features)
    if w is not None and w.status("hr") == ABSENT:
        return RuleOutcome("hr_surge_sustained", PARTIAL, {},
                           note="本设备无心率通道,大发作主判据无法判定",
                           missing=("hr",), source=src)
    base = baseline.center("hr") if baseline.ready else None
    if hr is None or base is None:
        return RuleOutcome("hr_surge_sustained", PARTIAL,
                           {"reason": "心率缺值或基线未建立"},
                           note="心率暂不可得或个人基线未建立,主判据待定",
                           source=src)

    rise_now = hr - base
    thr = cfg.hr_rise_bpm                        # 默认 30:医生给的下界
    held_s, n = _sustained_for(history, t, "hr",
                               lambda v: (v - base) >= thr,
                               max_look_s=cfg.hr_sustain_lookback_s,
                               max_gap_s=cfg.max_gap_s)
    # 当前窗也算进持续时长(history 里尚未写入本窗)
    if rise_now >= thr:
        held_s += cfg.window_hop_s

    detail = {"hr": round(hr, 1), "baseline": round(base, 1),
              "rise": round(rise_now, 1), "held_s": round(held_s, 1),
              "need_rise": thr, "need_s": cfg.hr_sustain_s, "samples": n}

    if rise_now >= thr and held_s >= cfg.hr_sustain_s:
        return RuleOutcome("hr_surge_sustained", MET, detail,
                           note=(f"心率 {hr:.0f}(平时 {base:.0f},升 {rise_now:+.0f}),"
                                 f"已持续 {held_s:.0f} 秒 —— 符合大发作判据"
                                 f"(升≥{thr}且持续≥{cfg.hr_sustain_s}秒)"),
                           demand="ESCALATE", source=src)
    if rise_now >= thr:
        return RuleOutcome("hr_surge_sustained", PARTIAL, detail,
                           note=(f"心率升 {rise_now:+.0f} 达判据幅度,但只持续 "
                                 f"{held_s:.0f} 秒(需 {cfg.hr_sustain_s} 秒)"),
                           source=src)
    return RuleOutcome("hr_surge_sustained", UNMET, detail, source=src)


# --------------------------------------------------------------------- #
# R2  心动过缓 / SUDEP 前兆:心率持续低于 60
# --------------------------------------------------------------------- #

def rule_bradycardia(features, baseline, history, cfg) -> RuleOutcome:
    """医生原话:「若心率持续异常(如持续低于60次/分钟),建议设定5至10秒的异常时间
    作为报警触发条件」;「夜间猝死最凶险的体征是心跳呼吸骤停,部分患儿会先出现
    呼吸停止,随后心率逐渐减慢」。

    第七棒**完全没有**这条:signals.py 只算 hr_surge(升),
    心率往下掉时所有信号都是 0 分 —— 一条彻底的漏报通道,
    而且漏的恰好是最致命的那种事件。
    """
    src = "医生访谈:心率持续低于60bpm达5-10秒即应报警;猝死前常见呼吸先停、心率渐慢"
    t = features.get("t")
    hr = features.get("hr")
    w = features_window(features)
    if w is not None and w.status("hr") == ABSENT:
        return RuleOutcome("bradycardia", PARTIAL, {},
                           note="本设备无心率通道,心动过缓判据无法判定",
                           missing=("hr",), source=src)
    if hr is None:
        return RuleOutcome("bradycardia", PARTIAL, {"reason": "心率缺值"},
                           source=src)

    # 绝对门限(医生给的 60)与个体门限(基线下方)取更宽松的那个 ——
    # 基线本来就低的孩子(实测 sub-059 中位 65bpm)用绝对 60 会迟钝,
    # 基线高的孩子用绝对 60 会漏;两者取或。
    # 判据形式:**绝对门限 AND 相对基线落差**(合取,不是取大/取小)。
    #
    # 三轮实测才定下来(validation/diag_brady{,2,3}.py,SeizeIT2 15 人):
    #
    # 1) 单用相对门限(基线×0.75)→ 灾难。Dravet 患儿基础心率本就快,
    #    sub-046 睡眠心率中位 **130bpm**,门限被抬到 97 → 整夜误报,
    #    规则命中 500+ 次,误报 1.20→3.80/h。
    #
    # 2) 单用医生给的绝对 60 → 仍然吵(0.80/h 本底)。根因是医学的:
    #    60bpm 是**人群参考值**,而这批孩子里好几个睡眠心率中位就在 64-68
    #    (sub-024 67、sub-059 64、sub-069 67、sub-031 68)。
    #    对他们来说深睡掉到 58 是正常生理,不是 SUDEP 前兆。
    #
    # 3) 合取 → 本底归零。要求"既低于人群参考值 60,又比**自己平时**
    #    低了 margin 以上"。基线 64 的孩子要掉到 ≤49 才算(真·心动过缓);
    #    基线 130 的孩子掉到 60 已是 -70 的巨幅下降,自然满足。
    #
    # margin 实测(背景 5 小时):0→0.80/h　5→0.60/h　10→0.40/h　**15→0**
    thr = cfg.brady_bpm
    margin = cfg.brady_margin_bpm
    base = baseline.center("hr") if baseline.ready else None
    if base is None:
        # 基线未建立时只能用绝对门限 —— 标注在 detail 里,便于复盘
        def _low(v):
            return v <= thr
    else:
        def _low(v, b=base):
            return v <= thr and (b - v) >= margin

    held_s, n = _sustained_for(history, t, "hr", _low,
                               max_look_s=cfg.brady_lookback_s,
                               max_gap_s=cfg.max_gap_s)
    if _low(hr):
        held_s += cfg.window_hop_s

    detail = {"hr": round(hr, 1), "abs_thr": thr,
              "baseline": round(base, 1) if base is not None else None,
              "margin_need": margin,
              "drop": round(base - hr, 1) if base is not None else None,
              "held_s": round(held_s, 1), "need_s": cfg.brady_sustain_s,
              "samples": n}

    if _low(hr) and held_s >= cfg.brady_sustain_s:
        drop_txt = (f",比平时低 {base - hr:.0f}" if base is not None else "")
        return RuleOutcome("bradycardia", MET, detail,
                           note=(f"心率降至 {hr:.0f}(低于 {thr:.0f}{drop_txt}),"
                                 f"已持续 {held_s:.0f} 秒 —— 心动过缓,SUDEP 高危体征"),
                           demand="CRITICAL", source=src)
    if _low(hr):
        return RuleOutcome("bradycardia", PARTIAL, detail,
                           note=(f"心率 {hr:.0f} 已达心动过缓,持续 {held_s:.0f} 秒"
                                 f"(需 {cfg.brady_sustain_s} 秒)"),
                           source=src)
    return RuleOutcome("bradycardia", UNMET, detail, source=src)


# --------------------------------------------------------------------- #
# R3  呼吸暂停:10-30 秒(发作期),常规憋气可达 30-60 秒
# --------------------------------------------------------------------- #

def rule_apnea(features, baseline, history, cfg) -> RuleOutcome:
    """医生原话:「强直阵挛发作期间,患儿常因喉肌痉挛出现呼吸暂停,通常持续10至30秒」;
    「缺氧超过6分钟就有脑损伤,能及时提示呼吸暂停,家长就可以及时……」

    第七棒有 breathing_irreg(呼吸率偏离/规整度),但**没有"暂停时长"**这个判据。
    呼吸暂停是"呼吸率趋零并持续 N 秒",和"呼吸率偏离基线"是两回事。
    """
    src = "医生访谈:发作期喉肌痉挛致呼吸暂停,通常10-30秒;呼吸监测是刚需"
    t = features.get("t")
    br = features.get("breathing")
    w = features_window(features)
    if w is not None and w.status("breathing") == ABSENT:
        return RuleOutcome("apnea", PARTIAL, {},
                           note="本设备无呼吸通道,呼吸暂停判据无法判定(医生列为刚需)",
                           missing=("breathing",), source=src)
    if br is None:
        return RuleOutcome("apnea", PARTIAL, {"reason": "呼吸率缺值"}, source=src)

    stop_thr = cfg.apnea_rate          # 呼吸率低到这个值视为"停"
    held_s, n = _sustained_for(history, t, "breathing", lambda v: v <= stop_thr,
                               max_look_s=cfg.apnea_lookback_s,
                               max_gap_s=cfg.max_gap_s)
    if br <= stop_thr:
        held_s += cfg.window_hop_s

    detail = {"breathing": round(br, 1), "stop_thr": stop_thr,
              "held_s": round(held_s, 1), "need_s": cfg.apnea_s, "samples": n}

    if br <= stop_thr and held_s >= cfg.apnea_s:
        # 暂停越久越危险:医生提到缺氧 6 分钟即脑损伤
        demand = "CRITICAL" if held_s >= cfg.apnea_critical_s else "ESCALATE"
        return RuleOutcome("apnea", MET, detail,
                           note=(f"呼吸近乎停止已 {held_s:.0f} 秒"
                                 f"(判据 ≥{cfg.apnea_s} 秒)"),
                           demand=demand, source=src)
    if br <= stop_thr:
        return RuleOutcome("apnea", PARTIAL, detail,
                           note=f"呼吸微弱 {held_s:.0f} 秒(需 {cfg.apnea_s} 秒)",
                           source=src)
    return RuleOutcome("apnea", UNMET, detail, source=src)


# --------------------------------------------------------------------- #
# R4  血氧下降 —— 延迟但高准确:做**升级确认器**,不做并列投票
# --------------------------------------------------------------------- #

def rule_desat_confirm(features, baseline, history, cfg) -> RuleOutcome:
    """医生/家长原话:「发作50秒-60秒,血氧才会出现变化,所以这个指标有延迟,
    但它的准确率非常高。比如翻身哭闹也会心率有波动……但缺氧指标的出现让报警的
    提示升级,或许是个可行的思路。」

    这正是第七棒做错的地方:spo2_desat 和 hr_surge 在 noisy-OR 里同权并列,
    于是"高准确率"这个性质被摊薄成了一票。它应该是:
    **其他信号先报 → 血氧一掉 → 直接升级 CRITICAL。**

    注意它的延迟性:不要求血氧与心率同时异常,而是允许血氧**滞后** ——
    所以这条规则在"最近 N 秒内曾有风险抬头"的前提下才生效。
    """
    src = "家长/医生经验:发作50-60秒后血氧才变化,延迟但准确率极高,宜用于报警升级"
    sp = features.get("spo2")
    w = features_window(features)
    if w is not None and w.status("spo2") == ABSENT:
        return RuleOutcome("desat_confirm", PARTIAL, {},
                           note="本设备无血氧通道,无法用缺氧升级报警",
                           missing=("spo2",), source=src)
    if sp is None:
        return RuleOutcome("desat_confirm", PARTIAL, {"reason": "血氧缺值"}, source=src)

    base = baseline.center("spo2") if baseline.ready else None
    ref = base if base is not None else 96.0
    drop = ref - sp

    t = features.get("t")
    prior = _span(history, t, 0.0, cfg.desat_context_s, "risk")
    prior_peak = max((v for _, v in prior), default=0.0)

    detail = {"spo2": round(sp, 1), "ref": round(ref, 1), "drop": round(drop, 1),
              "need_drop": cfg.desat_drop, "prior_risk_peak": round(prior_peak, 3),
              "context_s": cfg.desat_context_s}

    if drop >= cfg.desat_drop:
        if prior_peak >= cfg.desat_context_risk:
            return RuleOutcome("desat_confirm", MET, detail,
                               note=(f"血氧降至 {sp:.0f}%(平时 {ref:.0f}%,降 {drop:.0f} 点),"
                                     f"且此前 {cfg.desat_context_s:.0f} 秒内已有异常 —— "
                                     f"缺氧确认,升级唤醒"),
                               demand="CRITICAL", source=src)
        # 孤立的血氧下降:可能是探头接触不良,但降幅大时仍不可忽视
        if drop >= cfg.desat_drop_alone:
            return RuleOutcome("desat_confirm", MET, detail,
                               note=f"血氧显著下降 {drop:.0f} 点(孤立出现,先按高风险处理)",
                               demand="ESCALATE", source=src)
        return RuleOutcome("desat_confirm", PARTIAL, detail,
                           note=f"血氧下降 {drop:.0f} 点,但此前无其他异常,待观察",
                           source=src)
    return RuleOutcome("desat_confirm", UNMET, detail, source=src)


# --------------------------------------------------------------------- #
# R5  发作时长:超过 2 分钟必须提醒(致死性风险)
# --------------------------------------------------------------------- #

def rule_prolonged_event(features, baseline, history, cfg) -> RuleOutcome:
    """医生原话:「若患儿持续发作超过2分钟,设备必须提醒家长,以防致死性风险。」

    这是**硬性规则**,与风险分高低无关:只要异常状态持续够久就必须报。
    第七棒的 warning_escalate_windows 是按"窗口数"算的(默认 5 窗 = 10 秒),
    与 2 分钟差一个数量级,而且窗口数会随采样步进漂移。
    """
    src = "医生访谈:持续发作超过2分钟设备必须提醒,以防致死性风险(癫痫持续状态)"
    t = features.get("t")
    held_s, n = _sustained_for(history, t, "risk",
                               lambda v: v >= cfg.prolonged_risk,
                               max_look_s=cfg.prolonged_lookback_s,
                               max_gap_s=cfg.max_gap_s)
    detail = {"held_s": round(held_s, 1), "need_s": cfg.prolonged_s,
              "risk_floor": cfg.prolonged_risk, "samples": n}
    if held_s >= cfg.prolonged_s:
        return RuleOutcome("prolonged_event", MET, detail,
                           note=(f"异常状态已持续 {held_s / 60:.1f} 分钟"
                                 f"(判据 {cfg.prolonged_s / 60:.0f} 分钟)—— "
                                 f"疑似癫痫持续状态,必须提醒"),
                           demand="CRITICAL", source=src)
    if held_s >= cfg.prolonged_s * 0.5:
        return RuleOutcome("prolonged_event", PARTIAL, detail,
                           note=f"异常已持续 {held_s:.0f} 秒,接近持续状态判据",
                           source=src)
    return RuleOutcome("prolonged_event", UNMET, detail, source=src)


# --------------------------------------------------------------------- #
# R6  发作后高危期:5-10 分钟内保持高敏
# --------------------------------------------------------------------- #

def rule_post_ictal_window(features, baseline, history, cfg) -> RuleOutcome:
    """医生原话:「单次发作结束后,仍需保持5至10分钟的高风险监测期,
    重点预防呼吸心跳骤停及发作后呼吸抑制。」

    第七棒的 post_ictal_shift 只回看 6 个窗(12 秒),而且只是加一票。
    真正需要的是:确认过一次 CRITICAL 后,**接下来 5-10 分钟降低报警门限**,
    因为这段时间正是 SUDEP 的窗口。这条由 NightGate 消费(改门限),
    规则本身只负责判断"现在是否处于高危期"。

    这条规则**只报告状态,不要求升级**(demand=None)。
    实现高敏的手段是 NightGate 里的门限下调,不是把每一窗都抬成 WARNING。

    这个区别是实测打出来的(validation/diag_sens_drop.py):
    我最初让它返回 demand="ESCALATE",后果是发作后 600 秒内
    **每一窗**(约 300 窗)都被钉成 WARNING。两个灾难:
      1. 家长侧:每次发作后十分钟持续 WARNING,等于十分钟的噪声。
      2. 更隐蔽:NightGate 只在 level == "LOG" 时调 baseline.observe(),
         于是高危期内基线**完全停止学习**。sub-110 的基线因此冻结,
         紧接着 50 秒后的那次真发作风险分从 0.610 掉到 0.176,
         0.6@45 工作点的灵敏度 88% → 82% —— 一条号称"只升不降"的规则
         通过冻结基线这条暗路把灵敏度弄丢了。

    教训:分层架构里"只升不降"必须包括**不得改变下层的学习输入**,
    否则安全机制会从侧门反噬灵敏度。
    """
    src = "医生访谈:单次发作后需保持5-10分钟高风险监测期,防呼吸心跳骤停与发作后抑制"
    t = features.get("t")
    last = features.get("_last_critical_t")
    if not isinstance(t, (int, float)) or not isinstance(last, (int, float)):
        return RuleOutcome("post_ictal_window", UNMET, {}, source=src)
    since = t - last
    detail = {"since_s": round(since, 1), "window_s": cfg.post_ictal_s}
    if 0 <= since <= cfg.post_ictal_s:
        return RuleOutcome("post_ictal_window", MET, detail,
                           note=(f"距上次确认发作 {since / 60:.1f} 分钟,"
                                 f"处于发作后高危期(报警门限已下调)"),
                           demand=None, source=src)
    return RuleOutcome("post_ictal_window", UNMET, detail, source=src)


RULES = (
    rule_hr_surge_sustained,
    rule_bradycardia,
    rule_apnea,
    rule_desat_confirm,
    rule_prolonged_event,
    rule_post_ictal_window,
)


def evaluate_rules(features, baseline, history, cfg) -> List[RuleOutcome]:
    """跑全部临床规则。任一规则抛异常都不能拖垮主链路 ——
    它退化为 PARTIAL 并记下原因,报警照常由风险分决定。"""
    out = []
    for fn in RULES:
        try:
            out.append(fn(features, baseline, history, cfg))
        except Exception as e:                       # noqa: BLE001
            out.append(RuleOutcome(fn.__name__, PARTIAL,
                                   {"error": f"{type(e).__name__}: {e}"},
                                   note="规则执行异常,已跳过(不影响风险分)"))
    return out


def coverage_report(outcomes: List[RuleOutcome]) -> Dict[str, Any]:
    """判据覆盖表:哪些医生判据真的在判、哪些因缺通道判不了。

    这张表要直接进交付文档 —— 它是"我看不全"的诚实声明,
    比一个漂亮的灵敏度数字更该给评委看。
    """
    rows, missing = [], set()
    for o in outcomes:
        rows.append({"rule": o.key, "status": o.status,
                     "missing": list(o.missing), "source": o.source})
        missing.update(o.missing)
    judged = sum(1 for o in outcomes if o.status in (MET, UNMET))
    return {"rules": rows, "judgeable": judged, "total": len(outcomes),
            "blocked_by_missing_channels": sorted(missing)}
