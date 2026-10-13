"""
simulator.py — 四模态夜间数据模拟器

生成一整晚的体动/呼吸/心率/血氧序列，可注入：
  - 正常睡眠结构（深睡-浅睡-REM 循环、体动微觉醒、翻身）
  - 强直-阵挛发作（tonic→clonic→post-ictal 三相 + SpO2 延迟下降）
  - 陷门场景（专测误报）：正常翻身 / REM 噩梦 / 孤立呼吸暂停

体动有两路（对齐真实硬件）：
  - motion  1Hz 粗粒度强度（供基线/融合）
  - accel   8Hz 细粒度波形（供 1.5-5Hz 节律检测——阵挛抽动的关键特征）

用途：引擎单元测试 + Demo 仪表盘数据源。种子固定 → 现场演示可复现。
"""

from __future__ import annotations
import math
import random

FS = 8  # accel 采样率 Hz


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_night(seed=42, minutes=480, seizures=None, traps=True):
    """seizures: [{'onset_min': 120, 'dur_s': 75, 'intensity': 1.0,
                  'type': 'GTC'|'NCSE'}]  默认 GTC（强直-阵挛）
    NCSE 型：无大幅体动，但呼吸紊乱 + 心率异常 + 血氧下降——
    这是市面所有竞品都做不到的检测目标。"""
    rng = random.Random(seed)
    n = minutes * 60
    motion = [0.0] * n
    breathing = [0.0] * n
    breath_reg = [1.0] * n
    hr = [0.0] * n
    spo2 = [0.0] * n
    labels = [0] * n

    # ---------- 正常睡眠结构 ----------
    base_hr, base_br = 72.0, 16.0
    t = 0
    while t < n:
        cycle_pos = (t % (90 * 60)) / (90 * 60)
        depth = math.sin(cycle_pos * math.pi)
        seg_end = min(t + 30, n)
        for i in range(t, seg_end):
            hr[i] = base_hr - 6 * depth + rng.gauss(0, 1.2)
            breathing[i] = base_br - 2 * depth + rng.gauss(0, 0.6)
            breath_reg[i] = _clamp(0.85 + 0.1 * depth + rng.gauss(0, 0.05), 0, 1)
            spo2[i] = 97.5 + rng.gauss(0, 0.5)
            motion[i] = max(0, rng.gauss(0.15 + 0.25 * (1 - depth), 0.12))
        t = seg_end

    # ---------- accel 8Hz（正常底噪 + 事件叠加）----------
    accel = [max(0.0, rng.gauss(0.15, 0.1)) for _ in range(n * FS)]

    def accel_write(sec, dur, fn):
        for j in range(int(dur * FS)):
            idx = sec * FS + j
            if 0 <= idx < len(accel):
                accel[idx] = fn(j / FS)

    # 翻身/微觉醒：大幅非节律动作（每 ~20min，15-40s）
    if traps:
        t = 15 * 60
        while t < n - 60:
            dur = rng.randint(15, 40)
            for i in range(t, min(t + dur, n)):
                motion[i] = max(motion[i], rng.uniform(2.0, 4.5))
                hr[i] += rng.uniform(8, 15)
                spo2[i] += rng.gauss(0, 0.2)
            accel_write(t, dur, lambda s: max(0.0, rng.gauss(3.2, 1.2)))  # 乱而有力
            t += rng.randint(15, 25) * 60

        # REM 噩梦：心率快、呼吸快、体动轻微无节律
        rem_t = int(n * 0.55)
        for i in range(rem_t, min(rem_t + 120, n)):
            hr[i] += rng.uniform(18, 28)
            breathing[i] += rng.uniform(3, 5)
            motion[i] = max(motion[i], rng.uniform(0.6, 1.5))
            breath_reg[i] = _clamp(breath_reg[i] - 0.3, 0, 1)

        # 孤立中枢性呼吸暂停：呼吸乱，其他一切正常
        ap_t = int(n * 0.3)
        for i in range(ap_t, min(ap_t + 25, n)):
            breathing[i] = max(4, breathing[i] - 10)
            breath_reg[i] = 0.2

    # ---------- 发作注入（GTC / NCSE 两种类型）----------
    for sz in (seizures or []):
        on = int(sz["onset_min"] * 60)
        dur = int(sz.get("dur_s", 70))
        k = sz.get("intensity", 1.0)
        sz_type = sz.get("type", "GTC")
        pre_hr = hr[on - 1] if on > 0 else base_hr

        if sz_type == "NCSE":
            # ── 非惊厥性持续状态（NCSE）──
            # 核心特征：体动维持低水平（<1.2），但自主神经信号必须足够强：
            #   · 心率偏移 +35~50 bpm（确保 hr_surge 跨越门限）
            #   · 血氧下降 5~8 点（确保 spo2_desat 跨越门限）
            #   · 呼吸紊乱频率高（breathing_irreg 持续触发）
            ncse_hr_shift = rng.choice([-30, +40, +45])
            spo2_baseline = 97.0
            for j in range(dur):
                i = min(on + j, n - 1)
                labels[i] = 1
                # 体动：始终 <1.2（这是 NCSE 的关键标志——无大幅运动）
                motion[i] = max(motion[i], rng.uniform(0.15, 1.0))
                # 呼吸：高度紊乱 + 频繁短暂暂停（35% 概率低谷）
                base_br_at_onset = base_br - 2 * math.sin(((on + j * FS) % (90*60)) / (90*60) * math.pi)
                if rng.random() < 0.35:
                    breathing[i] = max(4, base_br_at_onset - 10 + rng.gauss(0, 1.5))
                    breath_reg[i] = 0.08
                else:
                    breathing[i] = base_br_at_onset + rng.gauss(0, 4.0)
                    breath_reg[i] = _clamp(0.35 + rng.gauss(0, 0.10), 0, 1)
                # 心率：显著偏移（+40~50 bpm，跨越绝对门限 +12bpm）
                hr[i] = pre_hr + ncse_hr_shift * k + rng.gauss(0, 2)
                # 血氧：持续下降 6~8 点（跨越 3 点阈值）
                desat = 7.0 * k * (1 - 0.3 * math.exp(-j / 18))
                spo2[i] = _clamp(spo2_baseline - desat + rng.gauss(0, 0.3), 83, 100)
            # accel 极低
            accel_write(on, dur, lambda s: max(0.0, rng.gauss(0.08, 0.05)))
            # 发作后恢复
            for j in range(60):
                i = min(on + dur + j, n - 1)
                if labels[i] == 0:
                    hr[i] += ncse_hr_shift * k * 0.12 * math.exp(-j / 20)
        else:
            # ── GTC 经典三相（tonic → clonic → post-ictal）──
            tonic, clonic = int(dur * 0.3), int(dur * 0.5)
            for j in range(dur):
                i = min(on + j, n - 1)
                labels[i] = 1
                if j < tonic:      # 强直相：持续肌张力
                    motion[i] = max(motion[i], rng.uniform(2.5, 4.0) * k)
                    hr[i] = pre_hr + (j / max(tonic, 1)) * 45 * k + rng.gauss(0, 2)
                    breathing[i] += 8
                    breath_reg[i] = 0.35
                elif j < tonic + clonic:  # 阵挛相：3Hz 节律抽动
                    motion[i] = max(motion[i], 4.5 * k + rng.gauss(0, 0.5))
                    hr[i] = pre_hr + 50 * k + rng.gauss(0, 3)
                    breathing[i] = 26 + rng.gauss(0, 3)
                    breath_reg[i] = 0.15
                else:              # 发作后抑制
                    motion[i] = min(motion[i], 0.3)
                    hr[i] = pre_hr + 30 * k * (1 - (j - tonic - clonic) / max(dur - tonic - clonic, 1)) + rng.gauss(0, 2)
                    breathing[i] = 10 + rng.gauss(0, 2)
                    breath_reg[i] = 0.5
            # accel：强直=中幅持续抖动；阵挛=3Hz 正弦（节律检测的目标信号）
            accel_write(on, tonic, lambda s: max(0.0, 3.0 * k + rng.gauss(0, 0.8)))
            accel_write(on + tonic, clonic,
                        lambda s: max(0.0, 4.5 * k * (0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * s))
                                      + rng.gauss(0, 0.5)))
            # SpO2 延迟下降（循环代偿滞后，持续 60s）
            d0 = on + tonic + int(clonic * 0.4)
            for j in range(60):
                i = min(d0 + j, n - 1)
                spo2[i] = _clamp(spo2[i] - (5.5 if j < 35 else 3.0) * k + rng.gauss(0, 0.4), 80, 100)
            # 发作后 10 分钟 HR 缓慢回落
            for j in range(600):
                i = min(on + dur + j, n - 1)
                if labels[i] == 0:
                    hr[i] += 12 * math.exp(-j / 180)

    return {
        "t": list(range(n)),
        "motion": motion, "accel": accel, "breathing": breathing,
        "breath_reg": breath_reg, "hr": hr, "spo2": spo2, "labels": labels,
    }


def _rhythm_score(seg, win_s, fs=FS):
    """1.5–5Hz 节律占比：一阶差分域的归一化自相关峰值。
    差分可以消除窗口内电平跳变（如翻身起止的台阶）造成的伪自相关；
    白噪声差分自相关≈0，正弦差分仍是正弦（保真）。"""
    d = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    mean = sum(d) / len(d)
    d = [x - mean for x in d]
    var = sum(x * x for x in d)
    if var <= 1e-9 or len(d) < 12:
        return 0.0, 0.0
    best_c, best_lag = 0.0, 0
    for L in range(2, 6):  # 对应 0.25–1s 周期
        a, b = d[:-L], d[L:]
        m = min(len(a), len(b))
        cov = sum(a[i] * b[i] for i in range(m))
        c = abs(cov / var)
        if c > best_c:
            best_c, best_lag = c, L
    freq = fs / best_lag
    in_band = 1.5 <= freq <= 5.0
    amp = max(seg) - min(seg)
    amp_scale = _clamp(amp / 3.0, 0, 1.0)
    score = _clamp((best_c - 0.45) / 0.45, 0, 1.0) * amp_scale * (1.0 if in_band else 0.3)
    return score, freq


def windowize(night, win_s=8, hop_s=4):
    """秒级序列 → 引擎窗口特征。motion_psd 由 8Hz accel 自相关节律分求得。"""
    n = len(night["motion"])
    accel = night["accel"]
    wins = []
    for start in range(0, n - win_s, hop_s):
        sl = slice(start, start + win_s)
        m = night["motion"][sl]
        seg = accel[start * FS:(start + win_s) * FS]
        rhythm_score, _freq = _rhythm_score(seg, win_s)
        wins.append({
            "t": start,
            "motion": sum(m) / len(m),
            "motion_psd": rhythm_score,
            "breathing": sum(night["breathing"][sl]) / win_s,
            "breath_reg": sum(night["breath_reg"][sl]) / win_s,
            "hr": sum(night["hr"][sl]) / win_s,
            "spo2": min(night["spo2"][sl]),
            "label": 1 if any(night["labels"][sl]) else 0,
        })
    return wins
