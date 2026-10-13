"""
gate.py — NightGuard 风险门控引擎（Tell gate.js 的 Python 移植 + 生理信号适配）

移植自 AssemblyAI 黑客松 Tell 项目 public/js/gate.js（8信号融合 + 个人基线 +
noisy-OR + 门控决策），核心思想一一对应：

  Tell（语音确定度）                NightGuard（夜间发作风险）
  ─────────────────────────────    ─────────────────────────────
  8 个语音犹豫信号                  8 个生理异常信号（可插拔）
  患者本通话内语音基线              患儿本夜生理基线（中位数 + MAD）
  noisy-OR 融合（强信号不被稀释）   同一套 noisy-OR
  不确定 → 拒绝写入 + 分级追问      不确定 → 拒绝报警 + 分级提醒 + 二次确认
  证据链 evidence 输出              每次报警输出可解释证据链

纯函数、无 I/O：同一份代码跑在模拟器、真实数据检验和设备端。
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from typing import Optional

from .clinical import evaluate_rules, coverage_report, MET, PARTIAL
from .clinical_config import ClinicalConfig
from .evidence import CRITICAL as EV_CRITICAL
from .evidence import WARNING as EV_WARNING
from .evidence import EvidenceConfig, classify
from .posture import PostureConfig, posture_gate


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def r3(v: float) -> float:
    return round(v, 3)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    mid = n >> 1
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def quantile(xs, q):
    """线性插值分位数（纯标准库，与 numpy 默认 'linear' 口径一致）。"""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def mad(xs):
    """中位数绝对偏差——一个离群窗口带不歪基线（与 gate.js PatientBaseline.mad 相同）。"""
    m = median(xs)
    if m is None:
        return None
    return median([abs(x - m) for x in xs]) or None


# ------------------------------------------------------------------ #
# 配置
# ------------------------------------------------------------------ #

@dataclass
class GateConfig:
    # 风险分超过该值且通过门控 → CRITICAL（立即唤醒家长）
    critical_threshold: float = 0.80
    # 风险分超过该值 → WARNING（推送通知 + 二次确认）
    warning_threshold: float = 0.55
    # 二次确认：WARNING 后需连续 N 个窗口维持高风险才升级为 CRITICAL
    confirm_windows: int = 2
    # 持续性判据（关键）：判定级别时必须用**最近一段时间内风险的中位数**，
    # 而不是瞬时风险。正常人睡一晚的 EEG/体动/心率波动大多是短暂的
    # （睡眠期转换、微觉醒、翻身、电极伪差，数秒即过），
    # 发作则是持续数十秒的平台。实测（CHB-MIT 23 名患儿 / 166 次发作，
    # 阈值扫描 1656 个工作点）：把持续性判据从"瞬时"改为"30 秒中位数"后，
    # 在同等灵敏度下误报降至朴素做法的 1/1.7–1/1.5（中低灵敏度可用区间）。
    # sustain_windows=1 时退化为原行为（保持兼容）。
    sustain_windows: int = 1
    # 时间基持续性（秒）：设置后覆盖 sustain_windows。
    # 为什么改时间基：窗口数语义依赖采样步进（2s/窗时 5 窗=10s、
    # 步进变了含义就变），"持续 30 秒"才是与设备/数据集无关的生理量。
    # 用法：GateConfig(sustain_seconds=30) —— 取最近 30 秒内风险的中位数。
    # 需要 features 里带绝对时间 t；无 t 时自动退回按窗口数（sustain_windows）。
    sustain_seconds: Optional[float] = None
    # 持续 WARNING：风险不到临界线但连续超过 N 窗仍不消退 → 也升级
    # （对应临床上"持续异常状态本身就是危险"）
    warning_escalate_windows: int = 5
    # 覆盖整个窗口序列的强信号连续出现也直接升级（不等奖满）
    escalate_windows: int = 3

    # 建立个人基线所需的最少正常窗口数（对应 gate.js minBaselineTurns）
    min_baseline_windows: int = 20

    # ---- 看护失效看门狗（第七棒新增）----------------------------------
    # 需求是「大发作绝对不可以漏报」。但"信号缺失"会制造一条**静默漏报通道**：
    # 心率质量不过关 → hr=None → 不参与融合 → 风险自然很低 → 判级 LOG →
    # 什么都不报。对家长来说，这和"一切正常"完全无法区分。
    #
    # 实测这不是边缘情况（validation/run_blindspot_audit.py，SeizeIT2 A 组 15 例）：
    #   · 整夜拿不到心率的窗口占比：中位 12.8%，最差患者 64.4%
    #   · **发作期落在盲区的窗口占比：中位 29.7%，最差 88.6%**
    # 后一个数才是要命的：强直-阵挛本身的肌电干扰会毁掉 ECG，
    # 于是"最需要看清的时刻"恰恰是最看不清的时刻。
    #
    # 所以把"看不见"本身做成一个级别：连续缺失超过 watchdog_seconds 就报
    # UNAVAILABLE——它不是发作报警，是「现在看不见孩子」的设备告警。
    # 宁可让家长知道设备瞎了，也不能让他以为一切正常。
    watchdog_seconds: float = 45.0

    # ---- 二次确认是否真的执行（第七棒新增）----------------------------
    # PROBE_BY_SIGNAL 一直只是给家长看的说明文字。打开这个开关后，
    # PROBE_EXEC 里登记的几条会**真的去回溯 history 查证**，
    # 查到良性原因就把 WARNING 降为 LOG（仍留日志）。
    # 置 False 可一键回到"只显示不执行"的旧行为，便于对照测量增量。
    probe_enabled: bool = True
    # ---- 临床规则层（第八棒新增）--------------------------------------
    # 医生访谈给的判据（心率升30持续45s / 心动过缓 / 呼吸暂停 /
    # 血氧升级确认 / 发作>2分钟 / 发作后高危期）编码在 clinical.py。
    #
    # 为什么要叠一层而不是改 influence：那些判据是**带时长的合取式**和
    # **硬性规则**，不是"再加一路投票"。扁平投票表达不了"持续 45 秒"，
    # 也表达不了"血氧一掉就升级"——血氧的价值恰恰在它延迟但准确，
    # 同权并列会把这个性质摊薄掉（第七棒就是这么做错的）。
    #
    # 规则只能**加严或升级**，不能把风险分已经判出的 CRITICAL 降下来：
    # 这一层是安全网，不是过滤器。
    clinical_enabled: bool = True
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    # ---- 事件级分级证据（第九棒新增）----------------------------------
    # 把 LOO 验证过的逐秒证据（体动持续 + EEG 佐证/救援）接进来，见
    # engine/evidence.py。嵌套 LOO：22/22 零丢弃，CRITICAL 0.019/h。
    #
    # **默认关闭**。两个原因：
    #   1. 验证侧的候选来自 stepwise.py + personal_novelty，生产侧候选
    #      来自本文件的 fuse_signals —— 是两套不同的候选生成器，
    #      0.019/h 不会自动继承。要开之前先跑 validation/run_gate_arch.py
    #      拿到生产路径自己的数。
    #   2. 开着会改变 demo/ 和现有测试的行为。默认关 = 行为不变。
    #
    # 证据层只能**下调到 WARNING**（静默留痕 + 走 warning_escalate_windows
    # 升级），不能把事件丢掉；临床规则判定的 CRITICAL 它也不许碰
    # （见 _apply_evidence 里的钳制）。
    evidence_enabled: bool = False
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)

    # ---- 体位门（第十棒接线）------------------------------------------
    # 竞品 NightWatch 降 5 倍误报靠的就是这一条，而我们 `engine/posture.py`
    # 早就写好了却**从未接到生产路径**（gate.py/signals.py 里一个 tilt
    # 字段都没有，它只活在 validation/run_*_arch.py 离线脚本里）。
    #
    # 依据（Lazeron 2022, Epilepsia Open 7:406-413，NightWatch 儿童版）：
    #   初版 FAR 1.27/h，**>90% 的误报来自心率分支**，根因是
    #   "孩子下床、在床上玩" —— 清醒活动，不是发作。
    #   他们没调心率阈值，只加了"仅水平体位才允许报警"：
    #     497 夜前瞻验证 FAR 1.27 → 0.24/h（降 5 倍），
    #     PPV 中位 11.1% → 58.1%，灵敏度 75.4% → **93.2%（反而升）**
    #   灵敏度不降反升，因为砍掉的全是清醒活动噪声。
    #
    # 安全边界（和证据层同一套）：
    #   · 只把 CRITICAL **降到静默 WARNING**，不丢事件；仍走
    #     warning_escalate_windows（持续不消退会重新升级响铃）
    #   · 临床规则判定的 CRITICAL 不碰
    #   · 算不出倾角（缺三轴/重力估不准）→ **放行**，看不见不等于没事
    #   · 半躺(≤45°)算躺：靠枕头坐起来一点仍可能发作，不能漏
    # 默认关 = 行为不变；要开先跑 validation 拿生产路径自己的数。
    posture_enabled: bool = False
    posture: PostureConfig = field(default_factory=PostureConfig)
    # ---- 仅供消融实验：还原两个已修的闩锁 -------------------------------
    # 生产代码永远用默认值。放在这里而不是让消融脚本 monkeypatch，
    # 是因为 monkeypatch 复现不了真正的环路 —— 第一版消融脚本用
    # "上一窗的级别" 近似 "本窗的级别"，环就断了（基线池 3072 vs 3228，
    # 无一冻结），于是测出来的差异与闩锁无关，结论完全是噪声。
    # 要测一个反馈环的代价，开关必须接在环**本身**上。
    ablate_baseline_latch: bool = False    # True = 恢复"只在 LOG 时学基线"
    ablate_anchor_guard: bool = False      # True = 恢复"每次 CRITICAL 都刷新锚点"

    # ---- 响铃节流（第九棒新增）------------------------------------
    # 实测 418 次误报里 62.7% 彼此间隔 <5 分钟（成簇），而真发作是孤立事件。
    # 成簇的根因是生理上一次事件（觉醒、翻身序列）跨多个窗反复越线，
    # 60s 的合并窗切不断它。节流窗设 120s：同一次事件的余波不再单独响铃。
    #
    # **只压制"响铃"，不压制判定、不压制日志。**被节流的窗口仍然：
    #   · 走完整的分级与临床规则（级别照常算出）
    #   · 写进 journal（家长第二天看得到）
    #   · 更新 last_critical_t 与基线学习门控
    # 它改变的只有 notify 这一个字段 —— 即"要不要在此刻吵醒家长"。
    #
    # 实测收益（137.2h / 17 次发作 / 0.40@45）：
    #   无节流   100% 灵敏度  3.05/h
    #   120s     100% 灵敏度  1.90/h   ← 灵敏度零代价
    # 再往上就要付灵敏度：300s → 88%（15/17）。所以停在 120s。
    # 置 0 关闭。
    alarm_throttle_s: float = 120.0
    # 不应期的**击穿**余量：当前风险比"上次响铃时的风险"高出这么多，
    # 就无视不应期强行响铃。
    #
    # 为什么必须有这条：发作常紧跟在体动之后（翻身→发作，或发作前驱动作）。
    # 没有击穿时，前面的误报占掉响铃资格，真发作落进不应期被压掉
    # ——实测灵敏度从 100% 掉到 71%（17/17 → 12/17）。
    # 不应期只许压"同等或更弱"的证据，绝不许压更强的。
    alarm_breakthrough_margin: float = 0.12
    # 临床规则判定成立（met）时击穿：那是医生判据，不是风险分的猜测。
    alarm_breakthrough_on_clinical: bool = True
    # 但**依赖"刚才报过警"的规则不算独立证据**，必须从击穿路径排除。
    #
    # 坑（实测 diag_bt.py）：post_ictal_window 在 1160 个 CRITICAL 窗里
    # met 了 1142 次 —— 它成立的前提就是 last_critical_t 不为空，
    # 即"刚才已经报过警"。拿它当击穿依据 = 报警给自己续期，
    # 节流完全失效（响铃窗 228.9/h，与不节流的 229.2 无差别）。
    # 这正是本仓库早有记录的自锁环，从新路径又回来了一次。
    alarm_breakthrough_rule_denylist: tuple = ("post_ictal_window",)
    # 响铃不应期**下限**：无论击穿与否，两次响铃至少隔这么久。
    #
    # 为什么单独有这条、而不是靠 alarm_throttle_s：throttle 允许被击穿，
    # 击穿是灵敏度的保命通道（上面那条注释：没它 17/17→12/17），不能堵。
    # 但击穿路径在真实信号上会被阈值附近的抖动反复触发：生产路径复测时
    # sub-024 一夜响铃 2846 次（233/h），而按事件合并只显示 9 个误报事件
    # —— 事件级指标完全看不见这个问题。
    #
    # 下限只**推迟**响铃、不取消（撞上下限的响铃会挂起，到期补响），
    # 所以它压的是"同一段证据被重复播报"，不是"漏掉新事件"。
    # 30s：取值依据是家长被连续叫醒的体感阈，且远小于发作典型时长
    # （>45s 的 sustain 门），不会把一次发作整个吞掉。
    alarm_min_gap_s: float = 30.0

    # ---- 报警发作期（第十一棒新增）------------------------------------
    # 为什么还要这一层：前面所有节流都是**两次响铃之间**的约束
    # （间隔 ≥throttle、≥floor、要击穿），没有任何一层约束
    # **一次事件总共响几次**。实测后果（θ=0.40 下限30s，165.1h）：
    #   误响铃 1198 次，按 300s 合并只有 **12 个事件**
    #   → 平均每个事件响 100 次，相邻响铃间隔中位 60s
    #   → 家长端体感 = 一次误报吵 50 分钟
    # 归因（diag_ring_attribution.py）：45% 的误响铃是"挂起补响"、
    # 29% 是"风险击穿"—— 都是**同一段证据被反复播报**。
    #
    # 事件级指标（0.073/h）把这个问题完全藏住了，响铃级（7.26/h）才看得见。
    # 这也是 gate.py:1099 早就写下的教训："事件级指标会把这种问题
    # 完全藏住，这也是为什么必须同时报响铃次数。"
    #
    # 语义：一次"报警发作期"从首次响铃开始，到**连续安静
    # episode_clear_s** 为止；期内最多响 episode_max_rings 次。
    # 和 alarm_min_gap_s 的区别：下限管间隔（推迟），这里管总数（封顶）。
    #
    # 安全性：封顶只减少**同一次事件内的重复播报**，不影响
    #   · 新事件（安静够久 → 新发作期 → 计数清零，一定能响）
    #   · 已经响过的那次（第一声永远放行，所以灵敏度不受影响）
    # 换句话说它压的是"吵多久"，不是"报不报"。
    # 这是本层能安全压 100 倍的原因：被压掉的全是第 2~100 声。
    #
    # episode_max_rings=2：第一声叫醒，第二声是"还在持续"的确认。
    # 再多对家长无信息增量 —— 人已经醒了，正在走向孩子。
    # episode_clear_s=600：取 clinical.post_ictal_s 同量级。发作后
    # 高危期本就 600s，期内的躁动属于同一次事件，不该重新计数。
    # 默认 False：这个封顶会**静默真发作**。
    # 实测（validation，20 夜 / 22 次发作）：开启时 6 次真发作期间的响铃
    # 被整段压掉（那 6 夜覆盖 1/7，压掉 2154 次响铃），全集召回假性显示
    # 为 16/22；关闭后是真实的 22/22。
    # 失败机制就是下面 step() 里写过的坑"发作前的误报吃掉额度"：
    # 下面声称 escalating 豁免已经修好，但豁免要求 risk 突破
    # **发作期峰值 + margin**，而先到的误报已经把 episode_peak_risk 抬高，
    # 真发作（risk 峰值 0.61-0.74，阈值 0.40）反而突破不了 ——
    # **豁免条件本身被误报污染了**。
    # "别把同一次发作重复播报"的需求改用 ring_collapse + 每夜预算实现，
    # 那两个压的是"一个事件响几声"，不会压掉"哪些事件该响"。
    alarm_episode_enabled: bool = False
    episode_max_rings: int = 2
    episode_clear_s: float = 600.0
    # ---- 响铃塌缩（ring collapse）------------------------------------
    # 一个**事件**只出一声：首声之后，同一事件内的其余响铃一律静默
    # （仍写日志、仍更新基线与证据，随时可被 escalating 击穿）。
    #
    # 为什么这是对的、而 episode 封顶是错的：
    #   episode 封顶按**时间段**计数，跨事件累计 → 误报吃掉真发作的额度。
    #   ring_collapse 按**事件**计数，事件边界由 left_critical 判定，
    #   新事件的计数天然从 0 开始 → 压不到下一个事件。
    # 所以它只改"一个事件响几声"，不改"哪些事件会响"，灵敏度无损。
    #
    # 实测收益（validation/run_ring_budget_final.py）：
    # 用户验收标准是**响铃次数 ≤2 次/10 夜**，事件级误报率不约束响铃数
    # （已知最坏案例：9 个误报事件 = 2846 次响铃）。
    # 既有最好架构事件级 22/22 @ 0.049/h，响铃级却是 63 次/10 夜；
    # 塌缩后 4.0 次/10 夜，灵敏度仍 22/22。
    ring_collapse: bool = True
    # 事件边界：连续安静满 collapse_clear_s 才算"上一个事件结束"。
    # 0 = 用窗级边界（跌出 CRITICAL 即结束）。
    #
    # 为什么不能用窗级：实测生产路径信号在阈值附近反复抖动
    # （sub-024 一夜跌出 CRITICAL 589 次 = 590 个"窗级事件"），
    # 每次抖动都重获一次响铃权 → 塌缩只降 1.1×（986.7→934.8 次/10夜）。
    # 取 300s 与验证口径的事件合并窗（MERGE=300）一致。
    #
    # ⚠⚠ **实测结论：这个参数不能开，保持 0。** ⚠⚠
    # 流式实测（validation/run_streaming_rings.py，20 夜 / 22 次发作）：
    #   窗级边界(0)   22/22   934.8 次误响铃/10夜
    #   300s 边界     16/22   131.0   ← 掉 6 次发作
    #   600s 边界     16/22   113.3   ← 掉 6 次发作
    # 漏的是 sub-024/025/027/069/103/115 —— 和 alarm_episode_enabled
    # 翻车时**一模一样的 6 次、一模一样的机制**："误报先把事件计数占满，
    # 紧跟的真发作被当成同一事件的延续而静默"。
    # 响铃次数确实降了 7 倍，但那 7 倍是用 6 次发作换的。
    #
    # 教训：凡是让"事件"跨越安静期的设计，都会把误报的额度记到
    # 真发作头上。压响铃必须在**不延长事件边界**的前提下做
    # （即上层每夜预算：按患者-夜分配，而不是按时间段合并）。
    collapse_clear_s: float = 0.0
    # 在线持续性代理的参数（见 _online_sustain_run_s）。
    # seg_pct 跟验证侧 SustainConfig.seg_pct 对齐（0.90）；
    # min_bg_windows 要求基线攒够样本才敢用这个门 —— 样本不足就返回
    # None，证据层按 degraded 维持原判，宁可不降级也不误压真发作。
    sustain_online_seg_pct: float = 0.90
    sustain_min_bg_windows: int = 60

    # 哪些字段算"看得见"（任一有值即算）。必须把所有模态都列上：
    # 可穿戴四路 + EEG 四路。否则跑 EEG 配置时会被误判为"信号全失联"。
    # 换信号源（雷达/床垫）时在这里加字段，不要改看门狗逻辑。
    vital_keys: tuple = ("hr", "motion", "breathing", "spo2",
                         "amp", "power", "rhythm", "spread")

    # 每个信号单独能贡献的最大怀疑度（对应 gate.js influence）。
    # 与 gate.js 同理由 noisy-OR 合并：任一强信号可独自拉高总风险，
    # 多个弱信号可以叠加——这正是"一个强信号不被七个弱信号稀释"。
    influence: dict = field(default_factory=lambda: {
        "motion_burst":        0.40,  # 体动爆发（← onsetDelay）
        "motion_rhythm":       0.45,  # 节律性运动 1.5-5Hz 强直-阵挛特征（← fillerLoad）
        "breathing_irreg":     0.30,  # 呼吸紊乱/暂停（← preAnswerPause）
        "hr_surge":            0.35,  # 心率骤升相对个人基线（← arousal）
        "spo2_desat":          0.38,  # 血氧下降（← wordConfidence 掉崖）
        "cross_modal_sync":    0.30,  # 多模态同步激活——发作独有（← revision）
        "post_ictal_shift":    0.25,  # 发作后抑制：体动骤停+心率仍高（← brevity）
        "quiet_violation":     0.22,  # 安静睡眠期违和（正常体动多在微觉醒）（← hedging）
        "autonomic_dissociation": 0.45,  # 【NCSE 专用】自主神经分离：低体动 + 多自主神经异常
        # ---- EEG 检验配置（CHB-MIT）----
        # influence 按实测"判别力"（发作窗触发率 − 正常窗触发率）赋权：
        #   power_shift  70.4%（5.7% → 76.1%）
        #   amp_surge    43.6%（8.3% → 51.9%）
        #   onset_abrupt 23.3%（5.5% → 28.8%）
        #   eeg_rhythm   14.0%（0.6% → 14.6%，触发少但精度高）
        # spatial_shift（跨导联线长变异系数）实测判别力仅 5.6%（7.9% → 13.6%），
        # 在正常窗上触发得和发作窗一样频繁，是纯误报源 → 已从信号集移除。
        "amp_surge":           0.38,
        "power_shift":         0.45,
        "eeg_rhythm":          0.20,
        "onset_abrupt":        0.22,
    })


# ------------------------------------------------------------------ #
# 个人基线（对应 gate.js PatientBaseline）
# ------------------------------------------------------------------ #

class PatientBaseline:
    """跨夜累积"这个孩子"的正常生理范围，之后的窗口都和ta自己的常态比。

    性能注记：中位数/ MAD 需要对整条历史排序，若每个窗口都算一次，
    整夜回放会退化成 O(n²logn)（实测扫一遍 3600 窗要几十秒）。
    因此每 STAT_EVERY 个观测才刷新一次统计量并缓存——与真实设备
    "每半分钟更新一次个人基线"的做法一致。
    """

    KEYS = ("motion", "breathing", "hr", "spo2", "amp", "power", "spread", "rhythm")
    STAT_EVERY = 8

    # 进基线前的生理合理性窗口。超出窗口的值是传感器噪声/算法误检，
    # 不是这个孩子的"正常态"，绝不能学进基线。
    #
    # 为什么必须有这道门：sub-046 的基线池里 14.8% 的样本 > 150 bpm
    # （睡眠中的患儿不可能有 15% 的夜间时间心率超 150），
    # 是 ECG 噪声段上 Pan-Tompkins 误检 R 峰算出的假心率。
    # 它们把 center 从真实的 ~70 抬到 124、MAD 撑到 20.4，于是真实发作
    # 心率 175（+105bpm）算出的 z_doubt 只有 0.002 —— 大发作被判为 LOG。
    # 而且是**自我强化**的：噪声窗判级为 LOG → 被当成正常学进基线 →
    # 基线越高越不报警 → 更多噪声进来。
    # 见 文档/第七棒-根因两个单位错配.md。
    PLAUSIBLE = {
        "hr": (35.0, 180.0),        # 儿童睡眠心率；超出按误检丢弃
        "spo2": (70.0, 100.0),
        "breathing": (6.0, 60.0),
    }

    # 基线取低分位而非中位：对残留污染更稳健。
    # 污染的方向性是单边的（噪声只会算出偏高的假心率，不会算出偏低的），
    # 所以 p25 比中位数更接近真实的安静态水平。
    QUANTILE = {"hr": 0.25}

    def __init__(self, min_windows: int = 20):
        self.values = {k: [] for k in self.KEYS}
        self.windows = 0
        self.min_windows = min_windows
        self.rejected = {k: 0 for k in self.KEYS}   # 质量门控丢弃计数（可观测性）
        self._cache = {}
        self._cache_at = -1

    def observe(self, features: dict):
        """喂一个已确认为正常的窗口（对应 gate.js 只用非关键回合建基线）。"""
        self.windows += 1
        hr_q = features.get("hr_q")
        for k in self.KEYS:
            v = features.get(k)
            if not (isinstance(v, (int, float)) and v == v):  # NaN guard
                continue
            # 第九棒：低置信心率不进基线。质量门控从"检测通道"下移到这里之后，
            # 检测照常拿到心率值，只有"学不学"受约束 —— 防污染的效果不变，
            # 但发作期不再因为丢拍而失去心率这唯一可用通道。
            if k == "hr" and hr_q is not None and hr_q < 0.5:
                self.rejected[k] += 1
                continue
            lo, hi = self.PLAUSIBLE.get(k, (float("-inf"), float("inf")))
            if not (lo <= v <= hi):
                self.rejected[k] += 1      # 生理不可能 → 丢弃，不进基线
                continue
            self.values[k].append(float(v))

    @property
    def ready(self) -> bool:
        return self.windows >= self.min_windows

    def _refresh(self, force=False):
        """缓存新鲜就直接用；否则每 STAT_EVERY 个观测才重算一次。"""
        if not force and self._cache_at == self.windows:
            return
        if not force and self._cache and self.windows % self.STAT_EVERY != 0:
            return
        for k in self.KEYS:
            v = self.values[k]
            if v:
                q = self.QUANTILE.get(k)
                m = quantile(v, q) if q is not None else median(v)
                # 离散度仍以中心点为参照算 MAD（用上面选出的 m，不是中位数）
                self._cache[k] = (m, median([abs(x - m) for x in v]) or None)
            else:
                self._cache[k] = (None, None)
        self._cache_at = self.windows

    def center(self, key) -> Optional[float]:
        self._refresh()
        return self._cache.get(key, (None, None))[0]

    def spread(self, key) -> Optional[float]:
        self._refresh()
        return self._cache.get(key, (None, None))[1]

    def snapshot(self) -> dict:
        self._refresh(force=True)
        return {k: {"center": self._cache[k][0], "spread": self._cache[k][1]}
                for k in self.KEYS} | {"ready": self.ready, "windows": self.windows}


def relative_doubt(value, base_center, base_spread, floor_scale=0.5):
    """偏离个人基线多少个 MAD → 怀疑度。无基线时回退绝对默认值（baselined=False）。"""
    if base_center is None:
        return None
    spread = base_spread or base_center * 0.25 or 1e-9
    z = abs(value - base_center) / max(spread, 1e-9)
    # 3 个 MAD 起算，10 个 MAD 封顶——与 gate.js onsetDelay 的 excess/3 同构
    return clamp01((z - 3.0) / 7.0 * floor_scale * 2)


# ------------------------------------------------------------------ #
# noisy-OR 融合（gate.js:329-331 的逐字对应）
# ------------------------------------------------------------------ #

def noisy_or(components):
    """任一信号失败的概率至少有一个发生——P(至少一信号报警)。

    components: [{key, doubt, influence, detail, note}, ...]
    """
    clean = 1.0
    for c in components:
        clean *= 1.0 - clamp01(c["doubt"]) * c["influence"]
    # 风险 = 1 - P(所有信号都干净) = P(至少一个信号报警)
    return clamp01(1.0 - clean)


def fuse_signals(signals, config: GateConfig):
    """signals: [{key, doubt(0-1), detail, note}, ...] → 风险评分 + 证据链。"""
    comp = []
    evidence = []
    for s in signals:
        if s is None or s.get("doubt") is None:
            continue
        d = clamp01(float(s["doubt"]))
        if d <= 0:
            continue
        influence = config.influence.get(s["key"], 0.2)
        comp.append({
            "key": s["key"],
            "doubt": r3(d),
            "influence": influence,
            "contribution": r3(d * influence),
            "detail": s.get("detail"),
        })
        if d > 0.35 and s.get("note"):
            evidence.append(s["note"])
    comp.sort(key=lambda c: -c["contribution"])
    risk = r3(noisy_or(comp))
    return {"risk": risk, "components": comp, "evidence": evidence}


# ------------------------------------------------------------------ #
# 门控决策（对应 gate.js gateDecision + PROBE_BY_SIGNAL）
# ------------------------------------------------------------------ #

# 二次确认动作：按"哪个信号触发"选择不同的确认策略——不是泛泛的"再看看"，
# 而是"去看它的配套信号是否同步"（对应 Tell 按触发信号选不同的追问话术）。
PROBE_BY_SIGNAL = {
    "motion_burst":     "体动偏高但无节律：进入观察期，等待心率/血氧佐证，不立即报警。",
    "motion_rhythm":    "检出节律性运动：连续两个窗口复核节律持续性，同时查心率是否同步上升。",
    "breathing_irreg":  "呼吸紊乱：检查是否伴随体动或血氧下降；孤立呼吸事件多为正常睡眠呼吸变异。",
    "hr_surge":         "心率升高：检查此前 2 分钟是否有体动（运动后心率属正常）；无体动的心率骤升才升级。",
    "spo2_desat":       "血氧下降：检查传感器接触与体动伪差；持续两个窗口下降才升级。",
    "cross_modal_sync": "多模态同步激活：这是发作的强特征，缩短观察期、提前升级。",
    "post_ictal_shift": "检出发作后抑制模式：回溯补记事件（post-ictal 补录）。",
    "quiet_violation":  "安静期出现大幅偏离：提高该窗口权重，等待下一窗口确认。",
    "amp_surge":        "EEG 幅度骤升：复核功率与空间分布是否同步异常。",
    "power_shift":      "全脑功率偏移：复核是否源于参考电极伪差。",
    "spatial_shift":    "导联间幅度异常不均：提示局灶性放电，缩短观察期。",
    "eeg_rhythm":       "EEG 节律性放电：检查幅度与空间分布是否同步扩大。",
    "onset_abrupt":     "功率骤然起跳：发作起始特征，提高该窗口权重。",
}


# ------------------------------------------------------------------ #
# 二次确认的**执行**（第七棒新增）
# ------------------------------------------------------------------ #
#
# 上面 PROBE_BY_SIGNAL 里写的是"该去查什么"，但在第七棒之前它**只是一段
# 显示给家长看的文字**——没有任何代码真的去查。这等于把架构里
# 「不确定时自行发掘新信息」那一层退化成了文案。
#
# 下面把其中可执行的几条真正执行：从 history 里回溯**已经采到、但当前窗
# 没有参与评分**的信息，用它来否决或确认当前这一窗。
# 这不需要新硬件、不增加延迟（数据已在手里），只是把该用的证据用上。
#
# 返回 (verdict, reason)：
#   "veto"    → 找到了能解释该异常的良性原因，降级（WARNING→LOG）
#   "confirm" → 反而找到了加重证据，允许缩短观察期
#   None      → 查不出结论，按原级别走（不确定时不降级，守住漏报底线）

def _hist_window(history, t, lo, hi, key):
    """取 history 中时间落在 [t-hi, t-lo] 内、key 非空的值。"""
    out = []
    for h in history:
        ht = h.get("t")
        if ht is None or h.get(key) is None:
            continue
        age = t - float(ht)
        if lo <= age <= hi:
            out.append(float(h[key]))
    return out


def probe_hr_surge(features, history, baseline):
    """心率骤升的二次确认：此前 2 分钟有没有体动？

    临床依据（医生访谈）：运动后心率升高属正常生理反应。孩子翻身、
    踢被子、起夜后心率升到 100+ 完全正常——这是家用场景误报的主要来源之一。
    真正危险的是**安静躺着心率却骤升**（自主神经发作表现）。

    注意方向：有体动 → 降级（能解释）；无体动 → 反而更可疑（confirm）。

    「运动后心率升高」要成立，必须是**先动、现在已经停下来**。
    如果体动此刻还在、还在往上走，那不是"运动后"，而是体动与心率同步
    抬升——恰恰是发作的强特征。

    这个方向性约束是用真实漏报换来的：sub-059 一次 59 秒的全面性发作，
    初版规则把发作**自己的**抽动当成"此前的运动"，连续否决 6 个窗口
    （体动比 3.2→12.6× 一路爬升），直接把这次发作漏掉了。
    详见 validation/diag_probe_miss.py。
    """
    t = features.get("t")
    if not isinstance(t, (int, float)):
        return None, None
    t = float(t)
    base_m = baseline.center("motion")
    if base_m is None or base_m <= 0:
        return None, None

    # 近窗（0-20s）：现在还在动吗？  远窗（20-120s）：之前动过吗？
    near = _hist_window(history, t, 0.0, 20.0, "motion")
    far = _hist_window(history, t, 20.0, 120.0, "motion")
    now_m = features.get("motion")
    if now_m is not None:
        near = near + [float(now_m)]
    if len(far) < 5 or not near:
        return None, None

    far_peak = max(far) / base_m
    near_peak = max(near) / base_m

    # 否决的前提有三条，缺一条都不否决：
    #   ① 之前确实动得不少（有可解释的运动史）
    #   ② 现在已经基本安静（"运动后"而不是"运动中"）
    #   ③ 体动没有在上升趋势（排除抽动逐渐加剧）
    if far_peak >= 3.0 and near_peak < 2.0 and near_peak < far_peak * 0.6:
        return "veto", (f"此前 2 分钟有体动（峰值 {far_peak:.1f}× 基线）"
                        f"且现已安静（{near_peak:.1f}× 基线）"
                        f"——运动后心率回落期，属正常生理反应，暂不升级")

    # 体动与心率同步抬升 → 加重，不是否决
    if near_peak >= 3.0 and near_peak >= far_peak:
        return "confirm", (f"心率升高且体动同步持续（现 {near_peak:.1f}× 基线，"
                           f"未回落）——多模态同步激活，发作强特征")

    if far_peak < 1.5 and near_peak < 1.5:
        return "confirm", (f"此前 2 分钟几乎无体动（峰值仅 {far_peak:.1f}× 基线）"
                           f"——安静状态下心率骤升，自主神经发作的典型表现")
    return None, None


def probe_spo2_desat(features, history, baseline):
    """血氧下降的二次确认：是不是体动伪差？

    PPG/血氧探头最常见的假下降来自运动伪差（手腕甩动、探头移位）。
    判据：当前窗体动很大 + 血氧只是单窗掉 → 优先按伪差处理。
    """
    t = features.get("t")
    m = features.get("motion")
    base_m = baseline.center("motion")
    if not isinstance(t, (int, float)) or m is None or not base_m:
        return None, None
    if m / base_m >= 6.0:
        return "veto", (f"血氧下降与大幅体动同窗（{m / base_m:.1f}× 基线）"
                        f"——优先按探头移位伪差处理，等待安静窗复核")
    return None, None


# 单通道极端免疫门限（第九棒）。都是"相对本人基线"的倍数/增量，
# 不是跨患者学出来的参数 —— 这一点是刻意的，见下面 probe_motion_burst 的说明。
IMMUNE_MOTION_RATIO = 8.0     # 体动达本人基线 8 倍 → 免疫否决
IMMUNE_HR_DELTA     = 20.0    # 心率超本人基线 +20bpm → 免疫否决
# 0.35 不是猜的：SeizeIT2 17 次继发全面强直-阵挛的节律性实测
# 落在 0.381~0.772（最小 0.381、p10 0.428、中位 0.560），
# 取 0.35 让全部 17 次都拿得到节律免疫且留有余量。
# 我第一版写 0.45，恰好卡掉最低的两次（0.381/0.390）—— 免疫门限
# 宁可偏松：它只决定"不许否决"，放松只会少否决误报，不会制造漏报。
IMMUNE_RHYTHM       = 0.35    # 节律性（抽动的周期性）达此值 → 免疫否决


def probe_motion_burst(features, history, baseline):
    """体动爆发的二次确认：**所有**可用通道都平淡才否决。

    第九棒改为合取判定。原版只看心率一路：
        if hr - base_hr <= 3.0: return "veto"
    在 SeizeIT2 上实测出这是一条结构性漏报通道 ——
    sub-083 是 15 人里唯一发作时心率**不升**的孩子
    （背景 131.3 → 发作 129.6 bpm，Δ-1.6），他那次真发作的心率升幅
    比**误报**的中位数（+43.8）还低。任何只看心率的否决都必然杀掉他。
    而他那次的体动是本人基线的 48.6 倍（误报中位仅 12.8 倍）——
    体动救得了他，前提是别让心率一路单独做决定。

    （我在 validation/实验/ 下先用人群 logistic 学过这个否决器：
     第一版杀 sub-083，修了匹配 bug 的第二版改杀 sub-015。
     **换了受害者说明病灶在结构不在个体** —— 人群里学出来的心率否决器
     必然切进某个孩子的尾部。所以这里不用任何跨患者学习的参数，
     只用"相对本人基线"的比值。）

    判定顺序（先免疫、后否决，不确定时不降级）：
      1. 任一通道相对本人基线极端 → confirm 或 None，绝不否决
      2. 所有可用通道都平淡，且至少有两路能看 → veto
      3. 只有一路能看 / 都看不了 → None（不降级，偏向报警）
    """
    hr        = features.get("hr")
    motion    = features.get("motion")
    rhythm    = features.get("rhythm")
    base_hr   = baseline.center("hr")
    base_mo   = baseline.center("motion")

    # ---- 第一步：单通道极端 → 免疫 -----------------------------------
    hr_hot = (hr is not None and base_hr is not None
              and hr - base_hr >= IMMUNE_HR_DELTA)
    mo_hot = (motion is not None and base_mo not in (None, 0)
              and motion / base_mo >= IMMUNE_MOTION_RATIO)
    rh_hot = (rhythm is not None and rhythm >= IMMUNE_RHYTHM)
    if hr_hot and mo_hot:
        return "confirm", (f"体动与心率同步抬升（+{hr - base_hr:.0f} bpm，"
                           f"体动 {motion / base_mo:.0f}×基线）——多模态同步是发作强特征")
    if rh_hot and mo_hot:
        return "confirm", (f"节律性体动（节律 {rhythm:.2f}，{motion / base_mo:.0f}×基线）"
                           f"——阵挛期的周期性抽动")
    if hr_hot or mo_hot or rh_hot:
        # 有一路极端但不成对：不敢加重，更不敢否决
        return None, None

    # ---- 第二步：合取否决，且要求至少两路可看 -------------------------
    calm, seen = [], 0
    if hr is not None and base_hr is not None:
        seen += 1
        if hr - base_hr > 3.0:
            return None, None            # 心率已有抬升 → 不否决
        calm.append(f"心率未跟随（{hr:.0f} vs 基线 {base_hr:.0f}）")
    if motion is not None and base_mo not in (None, 0):
        seen += 1
        if motion / base_mo > 3.0:
            return None, None            # 体动不算平淡 → 不否决
        calm.append(f"体动 {motion / base_mo:.1f}×基线")
    if rhythm is not None:
        seen += 1
        if rhythm > 0.25:
            return None, None            # 有节律性 → 不否决
        calm.append(f"无节律性（{rhythm:.2f}）")

    if seen < 2:
        return None, None                # 只有一路能看 → 证据不足，不降级
    return "veto", ("、".join(calm) + "——多通道一致平淡，正常睡眠翻身的典型表现，记录不报警")


# 只登记"能真正执行"的 probe。写在 PROBE_BY_SIGNAL 里但这里没有的，
# 仍然只是给家长看的说明文字——不要让两者看起来一样可信。
PROBE_EXEC = {
    "hr_surge":     probe_hr_surge,
    "spo2_desat":   probe_spo2_desat,
    "motion_burst": probe_motion_burst,
}


@dataclass
class GateResult:
    # "LOG" | "WARNING" | "CRITICAL" | "UNAVAILABLE"
    #   UNAVAILABLE = 看护失效（全部生命信号失联）。它**不是**发作报警，
    #   而是"现在看不见孩子"的设备告警 —— 见 GateConfig.watchdog_seconds。
    #   消费端要分开处理：UNAVAILABLE 不该播发作铃声，但必须让家长知道。
    level: str
    risk: float
    confirmed: bool       # 是否通过二次确认（决定要不要真的吵醒家长）
    driver: Optional[str] # 触发的首要信号
    probe: Optional[str]  # 二次确认策略
    evidence: list
    components: list
    note: Optional[str] = None
    # 二次确认的**执行结果**（None 表示没查或查不出结论）。
    # probe 字段是"该查什么"的说明文字，这两个才是"查了，结论是什么"。
    probe_verdict: Optional[str] = None     # "veto" | "confirm"
    probe_reason: Optional[str] = None
    # 是否真的在此刻响铃叫醒家长。level=="CRITICAL" 但 notify==False
    # 表示"判定成立，但处于上一次报警的节流窗内，不重复吵醒"。
    # 消费端（App/手表）只应对 notify==True 播铃声；UI 仍可显示级别。
    notify: bool = True
    throttled_for_s: Optional[float] = None   # 被节流时距上次响铃多久
    # ---- 临床规则层结果（第八棒）----
    # rules：每条医生判据的 met/partial/unmet + 人话说明 + 出处
    # clinical_note：哪条规则把级别抬上去的（给家长看的第一句话）
    # coverage：哪些判据因缺通道判不了——必须进报告，这是"我看不全"的声明
    rules: list = field(default_factory=list)
    clinical_note: Optional[str] = None
    coverage: Optional[dict] = None


_LEVEL_RANK = {"LOG": 0, "WARNING": 1, "CRITICAL": 2}


def apply_clinical(result: GateResult, outcomes, config: GateConfig) -> GateResult:
    """把临床规则的结论叠加到风险分判出的级别上。

    **只升不降。** 这一层是安全网：风险分漏掉的（心动过缓、呼吸暂停时长、
    持续状态）由规则补上；风险分已经判出的 CRITICAL 规则无权否决。
    想降级只有二次确认（probe）那一条路，且它也只能降 WARNING。

    两种诉求：
      demand="CRITICAL"  → 直接唤醒（血氧确认缺氧、发作>2分钟、长时呼吸暂停）
      demand="ESCALATE"  → 至少 WARNING；若已 WARNING 则升 CRITICAL
                           （大发作主判据成立、发作后高危期、心动过缓）

    之所以 ESCALATE 在已 WARNING 时直接升级：这些判据本身就是医生给的
    报警条件，它与风险分同时成立意味着两套独立逻辑都指向发作。
    """
    if not outcomes:
        return result
    cov = coverage_report(outcomes)
    met = [o for o in outcomes if o.status == MET]
    level = result.level
    notes = []
    demand_crit = [o for o in met if o.demand == "CRITICAL"]
    demand_esc = [o for o in met if o.demand == "ESCALATE"]

    if demand_crit:
        level = "CRITICAL"
        notes = [o.note for o in demand_crit if o.note]
    elif demand_esc:
        if _LEVEL_RANK.get(level, 0) >= 1:
            level = "CRITICAL"
        else:
            level = "WARNING"
        notes = [o.note for o in demand_esc if o.note]

    # 绝不降级
    if _LEVEL_RANK.get(level, 0) < _LEVEL_RANK.get(result.level, 0):
        level = result.level

    rules = [{"rule": o.key, "status": o.status, "note": o.note,
              "detail": o.detail, "source": o.source,
              "demand": o.demand, "missing": list(o.missing)}
             for o in outcomes]
    evidence = list(result.evidence)
    for n in notes:
        if n and n not in evidence:
            evidence.append(n)

    changed = level != result.level
    return replace(
        result,
        level=level,
        confirmed=result.confirmed or (changed and level == "CRITICAL"),
        evidence=evidence,
        rules=rules,
        coverage=cov,
        clinical_note=notes[0] if notes else None,
        note=(notes[0] if changed and notes else result.note),
    )


def gate_decision(scored, config: GateConfig, streak: int, escalating: bool,
                  sustained: Optional[float] = None) -> GateResult:
    """分级门控。

    - risk < warning → LOG：仅记录进事件日志（不打扰家长）
    - warning ≤ risk < critical → WARNING：推送通知 + 按信号选择二次确认策略
    - risk ≥ critical 且连续 confirm_windows 个窗口维持 → CRITICAL：唤醒
    - 连续 escalate_windows 个窗口强证据（escalating=True）→ 直接 CRITICAL
      （对应 gate.js 的 attempt/exhausted：追问一次仍不确定就按低置信记录——
        这里反过来：连续确认两次就必须按高风险处理，宁可误叫不可漏叫）

    sustained：最近一段时间（sustain_seconds）或最近 sustain_windows 窗内
    风险的中位数。判定级别时瞬时值与"持续值"必须**同时**越过阈值——
    一个孤立的尖峰不构成报警，这对"区分正常睡眠行为与异常发作"是决定性的。
    """
    risk = scored["risk"]
    sus = risk if sustained is None else sustained
    components = scored["components"]
    driver = components[0]["key"] if components else None
    evidence = scored["evidence"]

    if risk >= config.critical_threshold and sus >= config.critical_threshold:
        if streak + 1 >= config.confirm_windows or escalating:
            return GateResult("CRITICAL", risk, True, driver,
                              PROBE_BY_SIGNAL.get(driver), evidence, components,
                              note="多窗口确认，立即唤醒")
        return GateResult("WARNING", risk, False, driver,
                          PROBE_BY_SIGNAL.get(driver), evidence, components,
                          note=f"高风险第 {streak + 1} 窗，等待下一窗口确认")

    if risk >= config.warning_threshold and sus >= config.warning_threshold:
        if streak + 1 >= config.warning_escalate_windows:
            return GateResult("CRITICAL", risk, True, driver,
                              PROBE_BY_SIGNAL.get(driver), evidence, components,
                              note=f"WARNING 持续 {streak + 1} 窗未消退，升级唤醒")
        return GateResult("WARNING", risk, False, driver,
                          PROBE_BY_SIGNAL.get(driver), evidence, components)

    return GateResult("LOG", risk, False, driver, None, evidence, components)


# ------------------------------------------------------------------ #
# 事件记录（赛题明确要求"事件记录"）
# ------------------------------------------------------------------ #

class EventJournal:
    """夜间事件日志：报警必须可追溯——时间、级别、风险分、证据链、驱动信号。"""

    def __init__(self):
        self.events = []

    def record(self, t_start, t_end, result: GateResult):
        self.events.append({
            "start": t_start, "end": t_end, "level": result.level,
            "risk": result.risk, "driver": result.driver,
            "evidence": result.evidence,
            "components": result.components,
            "confirmed": result.confirmed,
            "probe_verdict": result.probe_verdict,
            "probe_reason": result.probe_reason,
        })

    def summary(self):
        return {
            "total": len(self.events),
            "critical": sum(1 for e in self.events if e["level"] == "CRITICAL"),
            "warning": sum(1 for e in self.events if e["level"] == "WARNING"),
            "log": sum(1 for e in self.events if e["level"] == "LOG"),
        }


# ------------------------------------------------------------------ #
# 流式评分器：把上面所有部件串成一条"夜"
# ------------------------------------------------------------------ #

class NightGate:
    """逐窗口流式评分。用法：

        ng = NightGate()
        for win in windows:            # win: {t, features..., signals 由 profile 生成}
            out = ng.step(win)         # → GateResult
    """

    def __init__(self, config: GateConfig = None, signal_fn=None):
        self.config = config or GateConfig()
        self.baseline = PatientBaseline(min_windows=self.config.min_baseline_windows)
        self.signal_fn = signal_fn          # (features, baseline, history) -> [signals]
        self.history = []                   # 最近窗口的简况（供 post_ictal / quiet 判断）
        self.risk_hist = []                 # 近期 (t, risk)：持续性判据（时间基或窗口基）
        self.streak = 0                     # 连续高风险窗口数
        self.journal = EventJournal()
        self.blind_since = None             # 看门狗：连续"看不见"的起始时间
        self.blind_windows = 0              # 无时间戳时按窗口计
        self.probe_vetoes = 0               # 二次确认否决掉的 WARNING 数（可观测）
        self.probe_confirms = 0             # 二次确认加重的次数
        self.evidence_demotions = 0         # 证据层把 CRITICAL 降成 WARNING 的次数
        self.posture_demotions = 0          # 体位门把 CRITICAL 降成 WARNING 的次数
        # 最近若干窗的倾角（度），体位门用。长度取 15 窗 ≈ 30s：
        # 够覆盖一次事件的主体，又不至于把半分钟前的体位算进来。
        self.tilt_hist = []
        # 上一窗已成立的临床规则（给击穿做边沿判定，见节流段注释）
        self.prev_breakthrough_rules = set()
        self.pending_ring = False           # 被响铃下限挡住、等补响的响铃
        self.ring_floor_blocks = 0          # 被下限挡住的次数
        self.pending_rings_fired = 0        # 下限到期后补响的次数
        self.last_critical_t = None         # 上次 CRITICAL 的时刻（发作后高危期用）
        self.last_notify_t = None           # 上次**真的响铃**的时刻（响铃节流用，与上者分开）
        self.throttled = 0                  # 被节流的窗口数（可观测性）
        self.left_critical = True           # 自上次响铃后是否降下过 CRITICAL
        self.quiet_since = None             # 上次事件**结束**的时刻（冷却基准）
        self.last_notify_risk = 0.0         # 上次响铃时的风险分（击穿判据用）
        self.broke_through = 0              # 击穿次数（可观测性）
        self.event_peak_risk = 0.0          # 本次事件迄今最高风险（击穿基准）
        self.event_rings = 0                # 本次事件已响次数（响铃塌缩用）
        self.collapse_quiet_t = None        # 塌缩用：连续安静的起点
        self.collapsed = 0                  # 被响铃塌缩压掉的响铃数（可观测）
        self.clinical_hits = {}             # 各条临床规则命中次数（可观测）
        # ---- 报警发作期（alarm episode）----------------------------------
        # 一次"报警发作期"= 从首次响铃开始，到连续安静 episode_clear_s 为止。
        # 期内最多响 episode_max_rings 次。详见 step() 节流段的说明。
        self.episode_t0 = None              # 本次发作期的首次响铃时刻
        self.episode_rings = 0              # 本次发作期已响次数
        self.episode_quiet_t = None         # 本次发作期内连续安静的起点
        self.episode_capped = 0             # 因上限被压掉的响铃次数（可观测）
        self.episodes = 0                   # 发作期总数（= 家长被叫醒的次数）
        # 本发作期迄今最高风险。与 event_peak_risk 的区别很关键：
        # 后者在每次跌出 CRITICAL 时清零（gate.py 的"离开 CRITICAL = 事件
        # 结束"），所以抖动信号每次重回 CRITICAL 都能让 stronger 成立；
        # 本字段**只在发作期过期时清零**，跨抖动保留，才能表达
        # "相对这一整段躁动，事态是否真的在升级"。
        self.episode_peak_risk = 0.0

    def _episode_expired(self, t: float) -> bool:
        """本次报警发作期是否已经结束（连续安静满 episode_clear_s）。

        用 episode_quiet_t（连续安静的**起点**）而不是"距上次响铃"：
        后者会在一次长事件里随时间自然到期，等于发作期给自己续期 ——
        和 post_ictal_window 自锁（gate.py:1020）是同一个坑，
        本仓库已经踩过两次，不再踩第三次。
        """
        if self.episode_t0 is None:
            return True
        q = self.episode_quiet_t
        if q is None:
            return False                     # 还没开始安静 → 仍在发作期内
        return (t - q) >= self.config.episode_clear_s

    # 进基线的"平静"判据。只读原始信号，**不读级别、不读风险分** ——
    # 这是它能打破正反馈环的前提（见 step() 内 baseline.observe 处的说明）。
    # 判据刻意全部**无量纲**（比例），不含任何绝对门限。
    # 体动的单位随 profile 变（SeizeIT2 加速度计约 0.001-0.05，
    # 模拟器约 0.5-4.0），任何写死的绝对值必然在另一个 profile 上失效 ——
    # 这和第八棒心动过缓踩的坑是同一个：先量分布再定值，不要凭感觉写门限。
    QUIET_MOTION_X = 4.0       # 体动超过自身基线的几倍 → 不学
    QUIET_HR_JUMP = 0.25       # 相邻窗心率跳变比例上限 → 急变不学

    def _is_quiet_for_baseline(self, features: dict) -> bool:
        """本窗是否平静到可以当作"这个孩子的常态"。

        只读原始信号与基线自身，**不读报警级别、不读风险分**，
        因此不会形成"级别↔基线"的正反馈环（那正是第三个闩锁的成因）。

        两条判据：
        · 体动高于自身基线 QUIET_MOTION_X 倍 → 翻身或发作，不学
        · 心率相对上一有效窗跳变超过 QUIET_HR_JUMP → 过渡态或噪声，不学
        基线尚未建立时一律接受（bootstrap），否则冷启动时什么都学不到。

        发作期这两条几乎必然触发，所以发作不会被学成常态；
        而稳定的正常睡眠（哪怕心率 120）会被正常学入 —— 这是修掉闩锁的关键：
        基线要学的是"这个孩子的心率水平"，不是"报警系统认可的水平"。
        """
        mo = features.get("motion")
        if isinstance(mo, (int, float)) and mo == mo:
            c_mo = self.baseline.center("motion")
            if c_mo and c_mo > 0 and mo > c_mo * self.QUIET_MOTION_X:
                return False
        hr = features.get("hr")
        if isinstance(hr, (int, float)) and hr == hr:
            prev = next((h["hr"] for h in reversed(self.history)
                         if isinstance(h.get("hr"), (int, float))
                         and h["hr"] == h["hr"]), None)
            if prev and prev > 0 and abs(hr - prev) / prev > self.QUIET_HR_JUMP:
                return False
        return True

    def _watchdog(self, features: dict):
        """看护失效检测 → 返回 UNAVAILABLE 的 GateResult，或 None 表示看得见。

        判定"看得见"的标准很宽：vital_keys 里**任一**信号有值即算。
        所以触发 UNAVAILABLE 意味着全部生命信号同时失联——
        设备脱落、电极松动、整段噪声。这种状态持续下去必须让家长知道。
        """
        cfg = self.config
        if not cfg.watchdog_seconds:
            return None
        alive = any(isinstance(features.get(k), (int, float))
                    and features.get(k) == features.get(k)
                    for k in cfg.vital_keys)
        t = features.get("t")
        if alive:
            self.blind_since = None
            self.blind_windows = 0
            return None

        # 看不见：累计时长
        if isinstance(t, (int, float)):
            if self.blind_since is None:
                self.blind_since = float(t)
            blind_for = float(t) - self.blind_since
        else:                                # 无时间戳 → 退回窗口计数
            self.blind_windows += 1
            blind_for = self.blind_windows * 2.0      # 保守按 2s/窗估

        if blind_for >= cfg.watchdog_seconds:
            return GateResult(
                "UNAVAILABLE", 0.0, False, "signal_lost",
                "检查设备是否脱落/松动；这不是发作报警，是监护已中断",
                [f"全部生命信号已失联 {blind_for:.0f} 秒——现在看不见孩子"],
                [], note="看护失效")
        return None

    def step(self, features: dict):
        blind = self._watchdog(features)
        if blind is not None:
            self.streak = 0
            self.history.append({"t": features.get("t"), "motion": None,
                                 "hr": None, "risk": 0.0})
            if len(self.history) > 120:
                self.history.pop(0)
            self.journal.record(features.get("t"), features.get("t"), blind)
            return blind

        # 倾角滚动缓冲（体位门用）。放在 watchdog 之后、判级之前：
        # 盲区窗不该污染体位历史，但正常窗无论判成什么级别都要记。
        self.tilt_hist.append(features.get("tilt"))
        if len(self.tilt_hist) > 15:
            self.tilt_hist.pop(0)

        signals = self.signal_fn(features, self.baseline, self.history)
        scored = fuse_signals(signals, self.config)

        # 持续性判据：优先按"最近 sustain_seconds 秒"取中位数（时间基），
        # 无 t 或未配置时退回"最近 sustain_windows 窗"（窗口基，保持兼容）。
        t = features.get("t")
        ss = self.config.sustain_seconds
        if ss is not None and isinstance(t, (int, float)):
            self.risk_hist.append((float(t), scored["risk"]))
            cutoff = float(t) - float(ss)
            while self.risk_hist and self.risk_hist[0][0] < cutoff:
                self.risk_hist.pop(0)
            sustained = median([r for _, r in self.risk_hist])
        else:
            sw = max(1, int(self.config.sustain_windows))
            self.risk_hist.append((None, scored["risk"]))
            if len(self.risk_hist) > sw:
                self.risk_hist.pop(0)
            sustained = median([r for _, r in self.risk_hist]) if sw > 1 else None

        escalating = self._is_escalating()

        # 发作后高危期：确认过一次 CRITICAL 后的 5-10 分钟内下调门限。
        # 医生明言这段是呼吸心跳骤停的窗口，不能用平时的门限看。
        cfg = self.config
        if (cfg.clinical_enabled and self.last_critical_t is not None
                and isinstance(t, (int, float))
                and 0 <= t - self.last_critical_t <= cfg.clinical.post_ictal_s):
            r = cfg.clinical.post_ictal_relief
            cfg = replace(cfg,
                          critical_threshold=max(0.05, cfg.critical_threshold - r),
                          warning_threshold=max(0.05, cfg.warning_threshold - r))

        result = gate_decision(scored, cfg, self.streak, escalating, sustained)
        result = self._run_probe(result, features)

        # 记下**风险分层**的判断，供基线学习使用。
        # 临床层随后可能把级别抬上去，但那不该改变"这一窗生理上是否正常"
        # 这个事实判断 —— 见下方 baseline.observe 处的说明。
        risk_level = result.level

        # 临床规则层：在风险分判出的级别之上叠加医生判据（只升不降）。
        # 放在 probe 之后：probe 可以降 WARNING，但规则补的漏报不该被它否决。
        clinical_demanded = False
        if cfg.clinical_enabled:
            feat_for_rules = dict(features)
            feat_for_rules["_last_critical_t"] = self.last_critical_t
            outcomes = evaluate_rules(feat_for_rules, self.baseline,
                                      self.history, cfg.clinical)
            before = result.level
            result = apply_clinical(result, outcomes, cfg)
            # 临床层把级别抬上去过 → 这是规则要求的 CRITICAL，证据层不许降。
            clinical_demanded = (result.level == "CRITICAL"
                                 and before != "CRITICAL")

        # 事件级分级证据：只降噪，不丢事件（详见 _apply_evidence 的边界说明）。
        result = self._apply_evidence(result, features, clinical_demanded)
        # 体位门：清醒活动（下床、床上玩）不该吵醒家长。
        # 放在证据层之后：两层都只能降到静默 WARNING，顺序不影响安全性，
        # 但体位是最便宜的判据，放最后能让 note 里留下更具体的原因。
        result = self._apply_posture(result, features, clinical_demanded)

        # streak 同理用 risk_level：它是风险分层自己的状态机，
        # 临床层的升级不该灌进去（否则发作后高危期会把 streak 顶满，
        # 下一次 escalate 判断失真）。
        if risk_level in ("WARNING", "CRITICAL"):
            self.streak += 1
        else:
            self.streak = 0

        # 发作后高危期的锚点：只认**新发作**，不认高危期内的后续 CRITICAL。
        #
        # 坑（实测 diag_fa_autopsy.py）：原来每次 CRITICAL 都刷新锚点，
        # 而高危期本身在下调门限、更容易判出 CRITICAL —— 于是
        # 高危期给自己续期，一次真发作后能自锁 18 分钟（实测 1152 秒）。
        # 家长端看到的是"一次报警响了十八分钟"，这正是误报率的主要来源。
        #
        # 正确语义：高危期是"距上一次**独立发作**多久"。要重新计时，
        # 必须先离开当前高危期（即经历一段平静），否则锚点不动。
        if result.level == "CRITICAL" and isinstance(t, (int, float)):
            in_window = (self.last_critical_t is not None
                         and cfg.clinical_enabled
                         and 0 <= t - self.last_critical_t
                         <= cfg.clinical.post_ictal_s)
            if cfg.ablate_anchor_guard or not in_window:
                self.last_critical_t = float(t)
        # 响铃节流：判定已定稿，这里只决定"此刻是否吵醒家长"。
        # 放在 last_critical_t 更新**之后**，才能用"上次响铃"而非"上次判定"
        # 做参照；且必须用独立的 last_notify_t —— 若复用 last_critical_t，
        # 被节流的窗口会不断把参照点往后推，成簇反而永远静默（自锁）。
        # 条件里带 ring_collapse：塌缩必须在 alarm_throttle_s=0 时也生效。
        # 它是独立机制（管"一个事件响几声"），不该被节流窗的开关连坐。
        if ((cfg.alarm_throttle_s > 0 or cfg.ring_collapse)
                and isinstance(t, (int, float))):
            if result.level != "CRITICAL":
                # 离开 CRITICAL = 事件结束，峰值清零。
                #
                # 位置很关键：必须在这里（安静窗）清零，才能在**下一个**
                # CRITICAL 窗计算 stronger 之前生效 —— 于是降级后重新成立的
                # 报警一定击穿、一定响铃。
                # 我把它挪到 stronger 之后试过，等价于"峰值跨窗保留"：
                #   清零（现在）  17/17 100%，响铃窗 87.9/h
                #   保留          15/17  88%，响铃窗 71.6/h
                # 保留版多压一点成簇余波，却把"紧跟体动簇之后的真发作"压掉了。
                # 红线是灵敏度，不拿它换安静。
                #
                # 本轮又验证了一次：我试过"安静够久才清零"（抖动期间保留
                # 峰值），想压住 sub-024 的响铃风暴。响铃确实从 233/h 掉到
                # 9.5/h，但生产路径灵敏度从 22/22 掉到 21/22、
                # CRITICAL 档 21→18。确认这条路不能走，已回退。
                # 风暴要用"响铃不应期下限"解决（见下面 floor），那只推迟
                # 响铃、不取消响铃，代价是延迟而不是漏报。
                self.event_peak_risk = 0.0
                # 事件结束 → 响铃计数清零，下一个事件重新有权响第一声。
                # 这一行是 ring_collapse 不会压掉真发作的根本原因。
                # collapse_clear_s>0 时要求"连续安静够久"才算结束，
                # 否则抖动信号每次跌出 CRITICAL 都重获响铃权（实测 589 次）。
                if cfg.collapse_clear_s <= 0:
                    self.event_rings = 0
                else:
                    q = self.collapse_quiet_t
                    if q is None:
                        self.collapse_quiet_t = float(t)
                    elif float(t) - q >= cfg.collapse_clear_s:
                        self.event_rings = 0
                # 记下**事件结束**的时刻。
                # 冷却要从"上次事件结束"算，不是从"上次开始响铃"算 ——
                # 否则一次长事件响 10 分钟后只需再安静 1 窗就能重新响铃，
                # 因为距"开始响铃"早已超过冷却窗。
                if not self.left_critical:
                    self.quiet_since = float(t)
                self.left_critical = True
                # 发作期的"连续安静"起点：只在**刚开始**安静时打点，
                # 后续安静窗不刷新 —— 否则永远在"刚安静"状态，发作期
                # 永不过期。回到 CRITICAL 会清掉它（见下面 else 分支）。
                if self.episode_quiet_t is None:
                    self.episode_quiet_t = float(t)
            else:
                # 不应期语义："一次事件只响一次"。要重新响铃必须同时满足
                #   (a) 中间确实降下过 CRITICAL（是**新**事件，不是同一次的延续）
                #   (b) 距上次响铃 >= alarm_throttle_s（成簇的余波不单独响）
                #
                # 只用 (b) 的纯时间版我先写错过一次：长事件每 120s 放行一次，
                # 评估口径按 60s 合并反而把一次事件记成 5~6 次，
                # 0.30@0 的误报从 10.60 涨到 16.90/h。少吵醒不等于少记账。
                # 判定只在"刚从安静恢复成 CRITICAL"这一刻做，且看的是
                # **安静间隔本身**有多长。
                #
                # 坑（实测）：原先写成"left_critical 挂住 + t-quiet_since>=窗"，
                # 于是一次连续事件里被压制后，随着 t 增长迟早满足条件，
                # 会在同一次事件中途重新响铃 —— 响铃次数从 3.05 涨到 5.20/h。
                # left_critical 必须在恢复的那一刻**被消耗掉**，无论响不响。
                # 击穿：对标**本次事件已出现过的峰值**，不是上次响铃那一刻。
                #
                # 坑（实测）：用 last_notify_risk 做基准时，一次事件内风险
                # 本来就上下波动 >0.12，于是每窗都"击穿"，节流完全失效
                # （响铃窗 228.7/h，和不节流的 229.2 没差别）。
                # 对标峰值才是"证据真的在升级"的意思：只有突破本次事件
                # 迄今最高风险 + 余量，才算新信息，才值得再吵一次。
                # 回到 CRITICAL：安静被打断，发作期的安静计时作废。
                # 但若安静已满 episode_clear_s，发作期**已经结束**，
                # 这里不能清 —— 清了就等于发作期永不过期（自锁）。
                if (self.episode_quiet_t is not None
                        and not self._episode_expired(float(t))):
                    self.episode_quiet_t = None
                stronger = (result.risk - self.event_peak_risk
                            >= cfg.alarm_breakthrough_margin)
                # 临床规则击穿必须是**边沿触发**：只在规则"刚刚成立"的
                # 那一窗放行，不是它"持续成立"的每一窗都放行。
                #
                # 坑（实测，生产路径复测时才暴露）：原先写成电平触发
                # （只要本窗 status==met 就击穿）。而 hr_surge_sustained /
                # apnea 这类判据按设计会连续成立好几分钟，于是每 2 秒
                # 击穿一次，120s 节流被完全绕过：
                #   sub-024 一夜响铃 2846 次，其中 2821 次是"临床击穿"。
                # 按事件合并看只有 9 个误报事件，很好看；但家长实际被
                # 吵醒 2846 次。**事件级指标会把这种问题完全藏住**，
                # 这也是为什么必须同时报响铃次数。
                # 边沿触发后：同一条规则持续成立只在起始响一次。
                now_met = {r.get("rule") for r in (result.rules or [])
                           if r.get("status") == "met"
                           and r.get("rule")
                           not in cfg.alarm_breakthrough_rule_denylist}
                newly_met = now_met - self.prev_breakthrough_rules
                # 仅边沿触发还不够：实测 hr_surge_sustained 会在同一段
                # 心率抬升里 met→unmet→met 来回抖（sub-024 一夜重新成立
                # 171 次，最短间隔 4s），每次抖动都算一个新边沿。
                # 所以临床击穿**自己也要受冷却约束**：距上次响铃不足
                # alarm_throttle_s 就不放行。
                #
                # 风险击穿（stronger）不需要这层：它要求突破"本次事件
                # 迄今峰值 + 0.12"，峰值单调上升，天然自限（一次事件最多
                # 击穿 (1-阈值)/0.12 次）。临床击穿是布尔量，没有这个性质。
                since_ring = (float(t) - self.last_notify_t
                              if self.last_notify_t is not None else None)
                clinical_ok = (since_ring is None
                               or since_ring >= cfg.alarm_throttle_s)
                clinical_met = bool(cfg.alarm_breakthrough_on_clinical
                                    and newly_met and clinical_ok)
                breakthrough = stronger or clinical_met

                # 回到 CRITICAL → 连续安静中断。必须在这里清，
                # 否则"安静够久"会在一次长事件中途自然满足（自锁坑，
                # 本仓库在 post_ictal_window 和 episode 上各踩过一次）。
                self.collapse_quiet_t = None

                def _ring():
                    self.last_notify_t = float(t)
                    self.last_notify_risk = float(result.risk)
                    self.left_critical = False
                    # 发作期记账：安静够久就是新发作期，计数清零。
                    if (self.episode_t0 is None
                            or self._episode_expired(float(t))):
                        self.episode_t0 = float(t)
                        self.episode_rings = 0
                        self.episode_peak_risk = 0.0   # 新发作期，升级基准归零
                        self.episodes += 1
                    self.episode_rings += 1
                    self.event_rings += 1
                    self.episode_quiet_t = None
                    self.episode_peak_risk = max(self.episode_peak_risk,
                                                 float(result.risk))

                first = self.last_notify_t is None
                # 刚从安静恢复：先看安静间隔够不够长，决定这是"新事件"
                # 还是"同一串成簇的余波"。
                fresh = False
                gap = None
                if self.left_critical and not first:
                    gap = (t - self.quiet_since) if self.quiet_since is not None else 0.0
                    self.left_critical = False      # 消耗掉，不留给后续窗
                    if gap >= cfg.alarm_throttle_s:
                        fresh = True
                # 峰值在判定**之后**更新，否则当前窗把自己的风险算进峰值，
                # stronger 永远为假。
                self.event_peak_risk = max(self.event_peak_risk, float(result.risk))

                # 响铃不应期下限：两次响铃之间至少隔 alarm_min_gap_s。
                #
                # 这是对"响铃风暴"的正解。实测生产路径上信号会在阈值附近
                # 反复抖动（sub-024 一夜跌出 CRITICAL 589 次），每次抖动
                # 都让 fresh/击穿 重新成立，于是响铃 2846 次 ≈ 233/h
                # —— 按事件合并看只有 9 个误报事件，**事件级指标把它完全
                # 藏住了**，但家长是真的被吵 2846 次。
                #
                # 和"保留峰值"的区别（后者已实测否决）：
                #   保留峰值 = 该响的不响 → 灵敏度 22/22 掉到 21/22
                #   下限     = 该响的**推迟到下限到期再响** → 不丢事件
                # 代价从"漏报"换成"最多延迟 alarm_min_gap_s"。对夜间看护
                # 来说，30s 内的延迟可接受，漏一次发作不可接受。
                #
                # pending_ring 保证"推迟"不退化成"取消"：被下限挡住的响铃
                # 挂起，等下限一到立刻补响（见下面 due 分支）。
                floor = cfg.alarm_min_gap_s
                blocked = (floor > 0 and since_ring is not None
                           and since_ring < floor)
                want = first or fresh or breakthrough

                # 发作期封顶：同一次发作期内响够 episode_max_rings 次后，
                # 其余一律静默（仍写日志、仍更新基线门控）。
                # 第一声永远不受此限 —— 新发作期的 episode_rings 为 0。
                # **封顶绝不压"比迄今更强的证据"。**
                #
                # 坑（实测 run_production_target.py，第一版封顶）：原先封顶
                # 对所有响铃一视同仁，结果灵敏度 22/22 → 15/22。机制是
                # "发作前的误报吃掉额度"：患者躁动 → 误报响满 N 声 → 真发作
                # 紧接着到来 → 被封顶静默；而患者一直不安静，发作期永不过期。
                # 这和 alarm_min_gap_s=300 把真发作推迟出命中窗是同一类错误
                # ——都是"压住了该响的那一声"。
                #
                # stronger（风险突破本次事件峰值 + margin）正是"事态在升级"
                # 的信号，也是发作紧跟误报时唯一能区分两者的东西。
                # 它天然自限：峰值单调上升，一次事件最多击穿
                # (1-阈值)/margin 次（gate.py:1178 已有论证），
                # 所以豁免它不会把封顶架空。
                # 临床击穿（clinical_met）不豁免：它是布尔量、没有自限性质，
                # 正是当年 sub-024 响 2846 次的原因。
                # 升级判据用**发作期**峰值，不用事件峰值：event_peak_risk
                # 每次跌出 CRITICAL 就清零，抖动信号能让 stronger 反复成立，
                # 封顶被架空（实测纯封顶仍响 75 次）。
                escalating = (float(result.risk) - self.episode_peak_risk
                              >= cfg.alarm_breakthrough_margin)
                capped = False

                # 响铃塌缩：本**事件**已经响过 → 其余响铃静默。
                # 用 event_rings（跌出 CRITICAL 即清零，见上面清零段）而不是
                # episode_rings（跨事件累计），所以压不到下一个事件 ——
                # 这正是它和 alarm_episode_enabled 的本质区别。
                # escalating 仍可击穿：事态升级必须出声。
                if (cfg.ring_collapse and want and not escalating
                        and self.event_rings >= 1):
                    capped = True
                    self.collapsed += 1

                if cfg.alarm_episode_enabled and want and not escalating:
                    if (self.episode_t0 is not None
                            and not self._episode_expired(float(t))
                            and self.episode_rings >= cfg.episode_max_rings):
                        capped = True

                if capped:
                    # 封顶不挂起：挂起会在发作期结束后补响一次陈旧证据，
                    # 那正是"挂起补响占 45%"的来源。此处直接静默丢弃
                    # 这**一声**（不是丢事件：事件已经响过前两声了）。
                    self.pending_ring = False
                    self.episode_capped += 1
                    result = replace(result, notify=False,
                                     throttled_for_s=None,
                                     note=(result.note or "")
                                     + (f"｜本事件已响 {self.event_rings} 次，静默"
                                        if cfg.ring_collapse
                                        and self.event_rings >= 1
                                        else f"｜发作期已响 "
                                             f"{self.episode_rings} 次，静默"))
                    self.throttled += 1
                elif want and blocked:
                    # 该响但撞上下限：挂起，等下限到期补响。
                    self.pending_ring = True
                    self.ring_floor_blocks += 1
                    result = replace(result, notify=False,
                                     throttled_for_s=round(floor - since_ring, 1))
                    self.throttled += 1
                elif want or (self.pending_ring and not blocked):
                    # 正常响铃，或下限到期后补响挂起的那一次。
                    if not want:
                        self.pending_rings_fired += 1
                    if breakthrough and not (first or fresh):
                        self.broke_through += 1
                    self.pending_ring = False
                    _ring()
                elif gap is not None:
                    # 从安静恢复但间隔不够长：成簇余波，不单独响。
                    result = replace(result, notify=False,
                                     throttled_for_s=round(float(gap), 1))
                    self.throttled += 1
                else:
                    # 同一次事件的延续：一次事件只响一次。
                    # 这条分支本来就实现了"一事件一声"（节流开启时）。
                    # ring_collapse 的增量有两处：
                    #   · alarm_throttle_s=0 时也生效（见上面的条件）
                    #   · 临床击穿（布尔量、无自限）不再豁免 ——
                    #     那正是 sub-024 响 2846 次的来源；
                    #     只有 escalating（风险突破峰值+margin，天然自限）
                    #     还能击穿。
                    if cfg.ring_collapse and self.event_rings >= 1:
                        self.collapsed += 1
                        note = ((result.note or "")
                                + f"｜本事件已响 {self.event_rings} 次，静默")
                    else:
                        note = result.note
                    result = replace(
                        result, notify=False, note=note,
                        throttled_for_s=round(float(t) - self.last_notify_t, 1))
                    self.throttled += 1

        # 记下本窗成立的规则集，供下一窗做边沿判定。
        # 无条件更新（不只在 CRITICAL 窗）：规则在 WARNING 期间持续成立时
        # 也要算"已经成立过"，否则回到 CRITICAL 会被当成新成立而误击穿。
        self.prev_breakthrough_rules = {
            r.get("rule") for r in (result.rules or [])
            if r.get("status") == "met" and r.get("rule")
            not in cfg.alarm_breakthrough_rule_denylist}

        for r in (result.rules or []):
            if r.get("status") == "met":
                self.clinical_hits[r["rule"]] = self.clinical_hits.get(r["rule"], 0) + 1

        # 基线只吃"确认正常"的窗口（对应 gate.js 只用闲聊回合建语音基线），
        # 避免把发作本身学成常态。
        # 注意：被二次确认否决的窗口**不能**进基线。
        # veto 的含义是"这次异常找到了良性解释"，不是"这个数值是常态"。
        # 若让它进基线，一串被否决的心率骤升会把基线教高，
        # 下一次真发作时 doubt 反而变小——等于自己把灵敏度喂掉。
        #
        # 用 risk_level（临床层叠加**之前**的级别）而不是 result.level：
        # 实测教训（validation/diag_sens_drop.py）。临床层抬高级别的理由
        # 常常与"本窗生理是否正常"无关 —— 发作后高危期就是纯时间判据，
        # 它一抬 300 窗，基线在这十分钟里完全停学。sub-110 的基线因此冻结，
        # 50 秒后的真发作风险分从 0.610 掉到 0.176，灵敏度 88%→82%。
        # 分层的"只升不降"必须包括：不得改变下层的学习输入。
        # ───────────────────────────────────────────────────────────────
        # 第三个闩锁（实测 diag_baseline_latch.py）：级别门控会把基线锁死。
        #
        # "只在 LOG 时学"构成一个正反馈环：
        #     基线偏低 → 正常心率看起来像骤升 → 判 CRITICAL
        #            → 不是 LOG，拿不到新样本 → 基线永远偏低 → 回到第一步
        # 入夜头几分钟正好是入睡过渡期（心率全夜最低），基线被钉在那儿，
        # 整夜再也回不来。实测 sub-073 基线池在第 157 个样本冻结，
        # 全夜 HR 中位 104 而基线 58.2，873 窗里 707 窗（81%）判 CRITICAL；
        # sub-070 池停在 134，745/885 窗（84%）CRITICAL。
        # 风险分恒为 0.350 的"报警"其实是饱和值，不含任何判断。
        #
        # 修法：基线学习与报警级别**解耦**。是否进基线只看两件与报警无关的事：
        #   1. 生理合理性（PatientBaseline.PLAUSIBLE 已在 observe 内把门）
        #   2. 本窗是否"平静"——体动低且心率没有急变
        # 平静判据只用当窗和近窗的原始信号，不读级别、不读风险分，
        # 因此不可能形成环路。发作本身体动剧烈/心率急变，自然被挡在外面。
        #
        # veto 窗仍然排除：veto 的含义是"这次异常找到了良性解释"，
        # 不是"这个数值是常态"。让它进基线会把基线教高，
        # 下一次真发作的 doubt 反而变小 —— 自己把灵敏度喂掉。
        if cfg.ablate_baseline_latch:
            # 消融专用：还原闩锁二的**原始**形态 —— 门控用本窗 risk_level。
            # 必须是本窗（不是上一窗），否则"基线低→判CRITICAL→学不到"
            # 这个环不成立，消融就测不到任何东西。
            learn_ok = (risk_level == "LOG")
        else:
            learn_ok = self._is_quiet_for_baseline(features)
        if result.probe_verdict != "veto" and learn_ok:
            self.baseline.observe(features)

        self.history.append({
            "t": features.get("t"),
            "motion": features.get("motion"),
            "hr": features.get("hr"),
            "risk": scored["risk"],
        })
        if len(self.history) > 120:
            self.history.pop(0)

        # 被二次确认否决的窗口也要入日志：家长第二天该能看到
        # "昨夜 2:10 有一次体动，系统判断为翻身"。降级 ≠ 丢弃。
        if result.level != "LOG" or result.probe_verdict == "veto":
            self.journal.record(features.get("t"), features.get("t"), result)
        return result

    def _online_sustain_run_s(self, features: dict) -> Optional[float]:
        """在线算"体动连续超背景分位的秒数"，供证据层用。

        为什么需要这个函数
        ------------------
        证据层 `_apply_evidence` 读的是 `features["sustain_run_s"]`，
        但**流式路径里没有任何人写这个字段** —— 只有测试在手工塞。
        于是线上永远是 None → classify 判 degraded → 一律维持原判，
        证据层接了等于没接。生产路径复测时实测降噪 0%，根因就在这里。

        验证侧（run_gate_arch.py）能算是因为它回头读原始加速度、
        切 1 秒段、拿整段录制的背景池定分位。流式只有每窗一个
        `motion` 标量，拿不到逐秒粒度，所以这里用**窗级代理**：
          · 背景门 = 安静期 motion 的 seg_pct 分位（基线已经在收集了）
          · run_s  = 最近连续超过该门的窗数 × 窗时长
        粒度粗（2s 一档）但方向一致：发作是连续数十秒的高体动，
        翻身碰撞连不成片。

        和验证侧的差异要记在案上：这是代理，不是同一口径。
        所以 gate 自己算出来的 run_s 只用于"够不够强"的分档，
        不拿它去复现验证侧的绝对数字。
        """
        cfg = self.config
        if not isinstance(features.get("motion"), (int, float)):
            return None
        pool = [v for v in self.baseline.values.get("motion", [])
                if isinstance(v, (int, float))]
        # 背景样本太少时不猜：返回 None → 证据层按 degraded 维持原判。
        # 宁可不降级，也不要用一个还没学好的门去压真发作。
        if len(pool) < cfg.sustain_min_bg_windows:
            return None
        thr = quantile(pool, cfg.sustain_online_seg_pct)
        if thr is None:
            return None
        # 往前数连续超门的窗，**容忍单窗掉落**。
        #
        # 严格连续在真实信号上太脆：sub-015 的发作里 t=52496 有一窗
        # motion 掉到门下（0.0025），严格计数把 run_s 从 24s 砍回 0，
        # 发作后半段直接被证据层判成"证据不足"降级 —— 丢了一次发作。
        # sustain.py 的 best_vote_frac 早就记了这件事（SeizeIT2 的 8/10
        # 规则："允许发作中途有个别段被伪迹打断"），这里沿用同一思路：
        # 连续两窗都在门下才算真的断了。
        if float(features["motion"]) <= thr:
            return 0.0
        run_ts = [features.get("t")]
        miss = 0
        for h in reversed(self.history):
            m = h.get("motion")
            if isinstance(m, (int, float)) and float(m) > thr:
                run_ts.append(h.get("t"))
                miss = 0
            else:
                miss += 1
                if miss >= 2:
                    break
        # 时长按**真实时间跨度**算，不按"窗数 × 假定窗长"——
        # 配置里没有窗长字段，写死 2.0 会在别的采样率下悄悄算错。
        ts = [x for x in run_ts if isinstance(x, (int, float))]
        if len(ts) >= 2:
            return float(max(ts) - min(ts))
        # 只有一窗超门：跨度为 0，但确实超了。返回 0 会被当成"完全没证据"，
        # 这里据 history 的典型窗间隔折算一窗的时长。
        gaps = [abs(b["t"] - a["t"]) for a, b in zip(self.history, self.history[1:])
                if isinstance(a.get("t"), (int, float))
                and isinstance(b.get("t"), (int, float))]
        gaps = [g for g in gaps if g > 0]
        return float(median(gaps)) if gaps else 0.0

    def _apply_posture(self, result: GateResult, features: dict,
                       clinical_demanded: bool) -> GateResult:
        """体位门：只在"躺着"时才允许吵醒家长。

        为什么是这一层而不是一个信号：体位不是"异常证据"，是**状态**。
        把它当信号喂进 fuse_signals 会被其他信号的权重稀释；
        竞品的做法也是门控（水平体位 **且** 心率/运动判据），不是加权。

        数据来源：features["tilt"]（逐窗倾角，度；0=躺平 90=立起）。
        由 wearable_features.tilt_windows 产出，需要先按记录标定身体长轴。
        **缺这个字段 → 放行**，不开新的漏报通道。

        判据用一段窗口的"躺着占比"而不是单窗：强直-阵挛本身会让躯干
        大幅位移，个别窗的重力估计会被加速度污染（这也是 posture.py 里
        horiz_frac_min 默认 0.5 的原因）。
        """
        if not self.config.posture_enabled:
            return result
        if result.level != "CRITICAL":
            return result
        if clinical_demanded:
            return result                    # 临床规则要求的，不动
        tilts = [t for t in self.tilt_hist if t is not None]
        if not tilts:
            return result                    # 算不出体位 → 放行
        keep, frac = posture_gate(self.tilt_hist, self.config.posture)
        if keep:
            return result
        self.posture_demotions += 1
        pct = f"{frac:.0%}" if frac is not None else "未知"
        return replace(result, level="WARNING", confirmed=False, notify=False,
                       note=f"体位门降级：窗内躺着占比仅 {pct}，"
                            f"判为清醒活动（非卧床）")

    def _apply_evidence(self, result: GateResult, features: dict,
                        clinical_demanded: bool) -> GateResult:
        """事件级分级证据：按逐秒证据把"够不够硬到叫醒家长"再判一次。

        判据与阈值见 engine/evidence.py（嵌套 LOO 22/22 零丢弃，
        CRITICAL 0.019/h，18/18 折一致）。这里只负责接线和守边界。

        安全边界（和 _run_probe 同一套道理，但这层更危险，写细一点）：
          · **只能把 CRITICAL 下调到 WARNING，绝不丢事件。** evidence 判
            NONE 时保持原级别不动 —— 本模块的 NONE 是"候选证据不足"，
            用在验证侧是"不进报警"，但在生产流式路径上，候选已经由
            fuse_signals 判成 WARNING/CRITICAL 了，这里再抹掉就是开一条
            漏报通道。证据层的作用是**降噪不是把门关上**。
          · **临床规则判定的 CRITICAL 不许碰。** clinical 层是安全网
            （gate.py:654-656 有单调向上钳制），证据层不能绕过它。
          · 降为 WARNING 时 notify=False（静默进日志），随后仍走已有的
            warning_escalate_windows：持续 N 窗不消退会重新升级响铃。
            所以"降级"不等于"放过"，只是不在第一秒吵醒家长。
          · 算不出证据（run_s / vote 都是 None）→ evidence 返回 WARNING
            且 degraded=True，这里按"保留原级别"处理：
            看不见不等于没事，不能因为缺数据就降级。
        """
        if not self.config.evidence_enabled:
            return result
        if result.level != "CRITICAL":
            return result
        if clinical_demanded:
            return result                    # 临床规则要求的 CRITICAL，不动
        # 外部给了就用外部的（离线评测/带 EEG 的设备可以直接喂逐秒结果），
        # 否则用窗级代理在线算。不写死成只信外部 —— 那就是线上永远 None。
        run_s = features.get("sustain_run_s")
        if run_s is None:
            run_s = self._online_sustain_run_s(features)
        v = classify(run_s, features.get("eeg_vote"), self.config.evidence)
        if v.degraded:
            return result                    # 数据不足 → 维持原判
        if v.level == EV_CRITICAL:
            return replace(result, note=f"证据层确认：{v.reason}")
        # WARNING 和 NONE 都降为静默 WARNING。
        #
        # 为什么 NONE 也只降到 WARNING 而不是丢掉：验证侧的 NONE 语义是
        # "不进报警"（那边在筛候选），但在生产流式路径上候选已经由
        # fuse_signals 判成 CRITICAL 了，直接抹掉就是开一条漏报通道。
        # 降为 WARNING = 不吵醒家长、仍写日志、仍受
        # warning_escalate_windows 保护（持续不消退会重新升级响铃）。
        #
        # 第一版这里让 NONE 保持原级别，结果证据层一个误报都降不下来
        # （实测生产路径降噪 0%）—— 过度保守到自我抵消。实测 NONE 这一
        # 档里 180/189 是误报、0 次发作（发作的 run_s 都 ≥47s），
        # 降级是安全的。
        self.evidence_demotions += 1
        why = v.reason if v.level == EV_WARNING else f"证据不足（{v.reason}）"
        return replace(result, level="WARNING", confirmed=False,
                       notify=False, note=f"证据层降级：{why}")

    def _run_probe(self, result: GateResult, features: dict) -> GateResult:
        """执行二次确认，按结论调整级别。

        安全边界（这条比压误报重要，先写下来免得以后被改坏）：
          · **veto 只能降 WARNING，绝不降 CRITICAL。** CRITICAL 已经过
            多窗口持续性确认，再让一条回溯证据否决它就是新开一条漏报通道。
          · veto 后记为 LOG 但**仍写入事件日志**（带 probe_reason），
            家长第二天能看到"昨夜 2:10 有一次体动，判断为翻身"。
            降级不等于丢弃——可追溯是赛题明确要求。
          · 查不出结论（None）时保持原级别，不降级。不确定时偏向报警。
        """
        if not self.config.probe_enabled:
            return result
        if result.level not in ("WARNING",):
            return result
        fn = PROBE_EXEC.get(result.driver)
        if fn is None:
            return result
        try:
            verdict, reason = fn(features, self.history, self.baseline)
        except Exception:
            return result                    # 二次确认失败绝不能拖垮主链路
        if verdict is None:
            return result
        if verdict == "veto":
            self.probe_vetoes += 1
            return replace(result, level="LOG", confirmed=False,
                           probe_verdict="veto", probe_reason=reason,
                           note=f"二次确认否决：{reason}")
        self.probe_confirms += 1
        return replace(result, probe_verdict="confirm", probe_reason=reason,
                       note=f"二次确认加重：{reason}")

    def _is_escalating(self) -> bool:
        """连续 escalate_windows 个窗口风险抬升 → 允许直接升级（不误伤快发作）。"""
        if len(self.history) < self.config.escalate_windows:
            return False
        risks = [h["risk"] for h in self.history[-self.config.escalate_windows:]]
        return all(risks[i] < risks[i + 1] for i in range(len(risks) - 1)) and risks[-1] > 0.5
