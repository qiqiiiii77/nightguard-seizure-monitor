"""
signals.py — 8 个生理信号检测器（可穿戴版）+ EEG 检验版

可穿戴 8 信号与 Tell 语音 8 信号的映射（见 docs/ARCHITECTURE.md）：
  Tell            →  NightGuard
  onsetDelay      →  motion_burst      体动爆发偏离个人基线
  fillerLoad      →  motion_rhythm     节律性运动 1.5–5Hz（强直-阵挛）
  preAnswerPause  →  breathing_irreg   呼吸紊乱/疑似暂停
  arousal         →  hr_surge          心率偏离个人基线（自主神经激活）
  wordConfidence  →  spo2_desat        血氧下降
  revision        →  cross_modal_sync  多模态同步激活（发作独有：运动+自主神经同时变）
  brevity         →  post_ictal_shift  发作后抑制（爆发后体动骤停而心率仍高）
  hedging         →  quiet_violation   安静睡眠期的违和偏离

每个检测器输出 {key, doubt(0-1), detail, note}；无个人基线时回退绝对阈值
（对应 gate.js baselined=false 的行为）。
"""

from __future__ import annotations
from .gate import clamp01, relative_doubt


# ------------------------------------------------------------------ #
# 可穿戴版（体动 / 呼吸 / 心率 / 血氧）
# ------------------------------------------------------------------ #

def motion_amp_gate(features, baseline, abs_floor=2.5):
    """体动幅度门限（0-1）——**必须无量纲**。

    为什么不能用绝对值：体动强度的单位取决于信号源。
      · 模拟器：任意量纲 0-10
      · 真实加速度计 RMS：g 单位，安静 0.0007、发作峰值 0.04
      · 毫米波雷达：又是另一个量纲
    早期版本写死 `clamp01(motion / 2.5)`，在真实 ACC 上乘数最大只有 0.02，
    把 motion_burst / motion_rhythm 两路信号**整体归零**——A 组 7 次
    ΔHR +50~105bpm、体动 36-114 倍基线的大发作因此 risk 只有 0.005~0.008。
    见 文档/第七棒-根因两个单位错配.md。

    所以改为**相对本人安静基线的倍数**，这个比值与设备、单位、佩戴位置都无关，
    换雷达也不用改。

    门限 3 倍起算 / 10 倍满格**由真实数据选出**，不是猜的
    （`validation/run_motion_ratio_dist.py`，SeizeIT2 A 组 8606 个背景窗
    vs 931 个发作窗，比值分布）：

        分位      背景(安静)    发作期
        p50          1.3        16.2
        p90          5.6        50.4
        p99         24.3        89.3

      起算/满格    背景触发   发作触发   区分度
        1.5/4.5     20.2%     75.0%     54.7%
        3/10         8.3%     66.7%   **58.4%**   ← 选它
        5/20         4.0%     55.9%     51.9%
        10/40        0.9%     35.1%     34.2%

    注：这个 58% 的区分度**远高于**第六棒测到的"体动判据判别力只有 10%"——
    因为那个实验用的是绝对阈值判据。同一个信号，改成相对个人基线的比值
    就从"几乎分不开"变成主力判据之一。

    基线未就绪时才回退绝对阈值（abs_floor 由调用方按信号源给，不写死）。
    """
    m = features.get("motion")
    if m is None:
        return 0.0
    c = baseline.center("motion") if baseline.ready else None
    if c is not None and c > 0:
        ratio = m / c
        return clamp01((ratio - 3.0) / 7.0)
    return clamp01(m / abs_floor)


def wearable_signals(features, baseline, history):
    """features 窗口字段约定：
    t, motion(体动强度RMS), motion_psd(1.5-5Hz节律功率占比 0-1),
    breathing(呼吸率/分), breath_reg(呼吸规整度0-1，越低越乱),
    hr(心率), spo2(血氧%)
    """
    out = []

    # 1. 体动爆发（带绝对下限：临床上运动性发作的体动量级远大于肌肉抽动）
    if features.get("motion") is not None:
        if baseline.ready and baseline.center("motion") is not None:
            doubt = relative_doubt(features["motion"],
                                   baseline.center("motion"), baseline.spread("motion"))
            ref = round(baseline.center("motion"), 2)
        else:
            doubt = clamp01((features["motion"] - 3.0) / 6.0)  # 绝对回退
            ref = 3.0
        # 软下限：幅度不足本人基线 1.5 倍的"偏离"大概率是浅睡抽动，按比例压低
        if doubt:
            doubt = doubt * motion_amp_gate(features, baseline)
        out.append({"key": "motion_burst", "doubt": doubt,
                    "detail": {"motion": round(features["motion"], 2), "ref": ref},
                    "note": f"体动达 {features['motion']:.1f}（本人常态 {ref}）" if doubt and doubt > 0.35 else None})

    # 2. 节律性运动（强直-阵挛的 1.5–5Hz 律动）。
    #    幅度门限：入睡抽动/REM 肌抽的节律性无临床意义，体动量级不足时压低。
    if features.get("motion_psd") is not None:
        amp_gate = motion_amp_gate(features, baseline)
        d = clamp01((features["motion_psd"] - 0.45) / 0.4) * amp_gate
        out.append({"key": "motion_rhythm", "doubt": d,
                    "detail": {"rhythm_ratio": round(features["motion_psd"], 3)},
                    "note": f"运动呈节律性（1.5-5Hz 占比 {features['motion_psd']:.0%}）" if d > 0.35 else None})

    # 3. 呼吸紊乱 / 疑似暂停（绝对偏离 <5 次/分在临床上无意义——
    #    浅睡/REM 本来就会让呼吸合法波动 ±3-5）
    if features.get("breathing") is not None:
        # dev 只在"有呼吸基线"时才有定义。baseline.ready 是**全局**窗口计数，
        # 它为真不代表 breathing 这一路有样本（设备可能整夜没有呼吸通道，
        # 或该路全被质量门控丢弃）。下面的 abs_gate 必须用同一个条件，
        # 否则 ready=True 而 center=None 时 dev 未绑定 → UnboundLocalError。
        dev = None
        if baseline.ready and baseline.center("breathing") is not None:
            dev = abs(features["breathing"] - baseline.center("breathing"))
            spread = max(baseline.spread("breathing") or 0.8, 0.8)  # 离散度下限防 z 爆炸
            d_br = clamp01((dev / spread - 2.0) / 6.0)
        else:
            d_br = clamp01((abs(features["breathing"] - 16) - 8) / 10)
        d_reg = clamp01((0.45 - (features.get("breath_reg") or 1.0)) / 0.4)  # 规整度塌了
        if dev is not None:
            abs_gate = clamp01(dev / 5.0)          # 偏离不足 5 次/分 → 不计
            doubt = max(d_br, d_reg) * max(abs_gate, 0.3 if d_reg > 0.5 else 0.0)
        else:
            doubt = max(d_br, d_reg)
        out.append({"key": "breathing_irreg", "doubt": doubt,
                    "detail": {"rate": features.get("breathing"), "regularity": features.get("breath_reg")},
                    "note": "呼吸节律紊乱" if doubt > 0.35 else None})

    # 4. 心率骤升（相对本人基线——对应 Tell 的 arousal）。
    #    双门限：MAD-z + 绝对升幅。深睡基线极稳（MAD~1.5），没有绝对门限时
    #    翻身带来的 +10bpm 也会算出 z=7——但临床上 +10bpm 什么都不说明。
    if features.get("hr") is not None:
        if baseline.ready and baseline.center("hr") is not None:
            spread = max(baseline.spread("hr") or 2.0, 2.0)  # 离散度下限
            up = (features["hr"] - baseline.center("hr")) / spread
            z_doubt = clamp01((up - 2.5) / 5.0)
            rise = features["hr"] - baseline.center("hr")
            ref = round(baseline.center("hr"))
        else:
            z_doubt = clamp01((features["hr"] - 110) / 40)
            rise = features["hr"] - 100
            ref = 110
        # 绝对升幅门限：+12bpm 起算，+30bpm 满格（发作的自主神经反应是 +30-50）
        abs_gate = clamp01((rise - 12) / 18.0)
        doubt = z_doubt * abs_gate
        out.append({"key": "hr_surge", "doubt": doubt,
                    "detail": {"hr": features.get("hr"), "ref": ref, "rise": round(rise, 1)},
                    "note": f"心率 {features['hr']:.0f}（本人常态 {ref}，升 {rise:+.0f}）" if doubt and doubt > 0.35 else None})

    # 5. 血氧下降（对应 wordConfidence 掉崖——单独就足以报警）
    if features.get("spo2") is not None:
        if baseline.ready and baseline.center("spo2") is not None:
            drop = baseline.center("spo2") - features["spo2"]
        else:
            drop = 96 - features["spo2"]
        doubt = clamp01((drop - 3.0) / 6.0)
        out.append({"key": "spo2_desat", "doubt": doubt,
                    "detail": {"spo2": features.get("spo2"), "drop": round(drop, 1)},
                    "note": f"血氧下降 {drop:.1f} 个点" if doubt and doubt > 0.35 else None})

    # 6. 多模态同步激活——创新点：发作 = 多系统"同时"大幅"病理性方向"偏离。
    #    方向性判据：心率只算骤升、血氧只算下降（REM 心率升+血氧微降同时
    #    出现会被误判，但 REM 的血氧降幅小——用 0.8 的离散度下限滤掉）。
    #    呼吸不参与同步计分：浅睡/REM 本来就会合法改变呼吸。
    soft = []
    for key, val, get, direction, floor in (
        ("motion", features.get("motion"), lambda: baseline.center("motion"), "any", 0.15),
        ("hr", features.get("hr"), lambda: baseline.center("hr"), "up", 2.0),
        ("spo2", features.get("spo2"), lambda: baseline.center("spo2"), "down", 0.8),
    ):
        if val is None:
            continue
        c = get()
        if c is None:
            continue
        s = max(baseline.spread(key) or floor, floor)
        dev = val - c
        if direction == "up" and dev > 3.5 * s:
            soft.append(key)
        elif direction == "down" and dev < -3.5 * s:
            soft.append(key)
        elif direction == "any" and abs(dev) > 3.5 * s:
            soft.append(key)
    n = len(soft)
    doubt = clamp01((n - 1.6) / 1.4)  # 2 个模态 → 轻度；3+ 个 → 满格
    out.append({"key": "cross_modal_sync", "doubt": doubt,
                "detail": {"synced": soft},
                "note": f"多模态同步偏离：{'+'.join(soft)}" if doubt > 0.3 else None})

    # 7. 发作后抑制（创新点：爆发后体动骤停、心率仍高——typical post-ictal）
    if len(history) >= 6 and features.get("motion") is not None:
        past = history[-6:]
        was_burst = any((h.get("motion") or 0) > 3.5 for h in past[:-2])
        recent = past[-2:]
        now_calm = all((h.get("motion") or 99) < 1.0 for h in recent) and features["motion"] < 1.0
        hrs = [h.get("hr") for h in recent] + [features.get("hr")]
        hr_still_high = any(h and h > 95 for h in hrs if h)
        doubt = clamp01(0.9 if (was_burst and now_calm and hr_still_high) else 0.0)
        out.append({"key": "post_ictal_shift", "doubt": doubt,
                    "detail": {"was_burst": was_burst, "now_calm": now_calm, "hr_high": hr_still_high},
                    "note": "发作后抑制模式：体动骤停而心率仍高" if doubt > 0.3 else None})

    # 8. 安静期违和（正常体动多发生在微觉醒；深睡眠安静期的大偏离更可疑）
    #    只有伴随自主神经变化（心率/血氧同步异常）才给高怀疑度——
    #    纯粹的安静期翻身仍然常见，单独不构成警报。
    if len(history) >= 10 and features.get("motion") is not None:
        prev_motion = [h.get("motion") or 0 for h in history[-10:-2]]
        quiet = all(m < 1.2 for m in prev_motion)
        big = features["motion"] > 2.5
        hr_elevated = features.get("hr") is not None and baseline.ready and \
            baseline.center("hr") is not None and \
            features["hr"] > baseline.center("hr") + 15
        spo2_dropping = features.get("spo2") is not None and baseline.ready and \
            baseline.center("spo2") is not None and \
            features["spo2"] < baseline.center("spo2") - 3
        doubt = clamp01((0.7 if (hr_elevated or spo2_dropping) else 0.3)
                        if (quiet and big) else 0.0)
        out.append({"key": "quiet_violation", "doubt": doubt,
                    "detail": {"quiet_before": quiet, "motion_now": round(features["motion"], 2)},
                    "note": "深睡安静期突现大幅偏离" if doubt > 0.3 else None})

    # 9. 【NCSE 专用】自主神经分离信号——非惊厥性癫痫持续状态的核心特征
    #    NCSE 患者外观安静（体动 <1.5），但自主神经系统同时出现：
    #    · 心率显著偏移（±20bpm+）
    #    · 血氧持续下降（≥4 点）
    #    · 呼吸节律紊乱（regularity <0.5）
    #    这是市面所有竞品都无法检测的信号模式——他们全部依赖运动信号。
    #    我们的引擎多模态融合天然适合捕捉这种"无声的危机"。
    # 同一类坑（见上方 breathing 处）：baseline.ready 是全局窗口计数，
    # 不保证 hr / spo2 这两路各自有样本。这里必须显式确认两个 center 都在，
    # 否则 ready=True 而 center=None 时会拿 None 做减法。
    if (features.get("motion") is not None and features.get("hr") is not None and
            features.get("spo2") is not None and baseline.ready
            and baseline.center("hr") is not None
            and baseline.center("spo2") is not None):
        c_hr = baseline.center("hr")
        c_sp = baseline.center("spo2")
        s_hr = max(baseline.spread("hr") or 2.0, 2.0)
        s_sp = max(baseline.spread("spo2") or 0.5, 0.5)
        # 心率显著偏移（绝对 +20bpm 或相对基线 4 MAD）
        hr_dev = abs(features["hr"] - c_hr)
        hr_flag = (hr_dev > 20) or ((hr_dev / s_hr) > 4.0)
        # 血氧持续下降（绝对 ≥4 点）
        sp_drop = c_sp - features["spo2"]
        sp_flag = sp_drop >= 4.0
        # 体动保持低水平（NCSE 的关键标志）
        low_motion = features["motion"] < 1.5
        # 呼吸紊乱（如果有）
        br_irreg = (features.get("breath_reg") or 1.0) < 0.5
        # 综合判断：低体动 + 至少两项自主神经异常
        n_flags = sum([hr_flag, sp_flag, br_irreg])
        if low_motion and n_flags >= 2:
            doubt = clamp01((n_flags - 1.5) / 1.0) * clamp01(hr_dev / 30.0)
            out.append({"key": "autonomic_dissociation", "doubt": doubt,
                        "detail": {"hr_dev": round(hr_dev, 1), "sp_drop": round(sp_drop, 1),
                                   "motion": round(features["motion"], 2), "flags": n_flags},
                        "note": f"NCSE 自主神经分离：体动低({features['motion']:.1f}) + HR偏离{hr_dev:.0f} + SpO2降{sp_drop:.0f}"
                                if doubt > 0.3 else None})

    return out


# ------------------------------------------------------------------ #
# EEG 检验版（CHB-MIT 真实数据用）——同一套融合/基线/门控，换一组信号
#
# 这一组信号是**实测筛出来的**，不是想出来的。初版用了"2-5Hz 谱占比 > 0.5"
# 这种在合成数据上调的绝对阈值，结果在真实 CHB-MIT 上一次都没触发
# （真实发作期该占比只有 0.33-0.47）。三轮单变量 AUC 筛选后改为：
#   amp_surge    ← 线长 75 分位        AUC .89（对局灶发作不稀释）
#   power_shift  ← 总功率 75 分位      AUC .93（最强单特征）
#   spatial_shift← 跨导联线长变异系数   AUC .72，与幅度相关仅 -0.25（正交轴）
#   onset_abrupt ← log 功率 30s 增量   AUC .61，与幅度相关 +0.21（正交轴）
# 弃用：2-5Hz 占比(.61，chb05 反向)、谱熵(.44，方向翻转)、频谱质心/过零率
# （chb01 上 .06-.09、chb03 上 .85-.87，跨患者方向翻转，不可作通用特征）。
# ------------------------------------------------------------------ #

def eeg_signals(features, baseline, history):
    """features: t, amp(线长75分位), power(总功率75分位),
    spread(跨导联线长变异系数), rhythm(2-5Hz自相关峰值), slope(log功率30s增量)"""
    out = []

    def rel(key, feat_val):
        if baseline.ready and baseline.center(key) is not None:
            return relative_doubt(feat_val, baseline.center(key), baseline.spread(key))
        return None

    amp = features.get("amp")
    if amp is not None:
        d = rel("amp", amp)
        if d is None:   # 绝对回退：真实 EEG 常态线长约 5-7，15 以上已属明显异常
            d = clamp01((amp / 8.0 - 1.0) / 1.8)
        out.append({"key": "amp_surge", "doubt": d, "detail": {"amp": round(amp, 1)},
                    "note": f"EEG 幅度骤升（线长 {amp:.0f}）" if d and d > 0.35 else None})

    power = features.get("power")
    if power is not None:
        d = rel("power", power)
        if d is None:   # 绝对回退：常态总功率约 1e9 量级
            d = clamp01((power / 2.0e9 - 1.0) / 4.0)
        out.append({"key": "power_shift", "doubt": d, "detail": {"power": power},
                    "note": "全脑功率偏移" if d and d > 0.35 else None})

    spread = features.get("spread")
    # 说明：早期版本这里有一条 spatial_shift 信号（跨导联线长变异系数）。
    # 实测判别力只有 5.6%（正常窗触发 7.9%、发作窗 13.6%）——在正常睡眠上
    # 触发得和发作一样频繁，纯误报源，已移除。保留读取以便日志/诊断使用。
    _ = spread

    slope = features.get("slope")
    if slope is not None:
        d = clamp01((slope - 0.15) / 0.35)   # 30s 内功率涨 1.5 倍(0.15 dex)起算
        out.append({"key": "onset_abrupt", "doubt": d, "detail": {"slope": round(slope, 3)},
                    "note": "功率骤然起跳" if d > 0.35 else None})

    rhythm = features.get("rhythm")
    if rhythm is not None:
        d = rel("rhythm", rhythm)
        if d is None:
            d = clamp01((rhythm - 0.30) / 0.20)
        out.append({"key": "eeg_rhythm", "doubt": d, "detail": {"acf": round(rhythm, 3)},
                    "note": f"节律性放电（2-5Hz 自相关 {rhythm:.2f}）" if d and d > 0.35 else None})

    return out

