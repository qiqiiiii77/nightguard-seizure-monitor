"""LLM 二级裁决：给引擎报出的候选报警做特异性复核。

为什么是这个架构（不是别的）
----------------------------
引擎侧的路已经走尽了，两条都有实测证据：
  · 全局单阈值：整条曲线没有一点同时满足高灵敏 + <=0.05/h
      θ=0.40 → 17/17 100% 但 1.24/h；θ=0.80 → 0.09/h 但只剩 53%
  · per-patient 预算标定（只看背景、不碰发作标签，可部署）：
      预算 0.05/h 时灵敏度只剩 8/17 = 47%
结论：**误报与发作在当前特征空间里不可分**，再调参数是徒劳。
所以引入一个能看懂"形态"而不只是看阈值的裁决者。

两个硬约束决定了形态：

1. **只审候选，不扫全程。** 137h 有 68 万个 2s 窗，逐窗调用不可能
   （延迟、成本、可靠性全不成立）。引擎 θ=0.40 时一夜约 187 个候选
   事件，这个量级可行。

2. **只允许否决，不允许新增。** 引擎保持 θ=0.40 的 100% 召回当初筛，
   LLM 只做减法。这样灵敏度的上界由引擎锁死，LLM 失灵（超时、网络断、
   返回垃圾）时按"保留报警"降级，**绝不会因为 LLM 不可用而漏报**。

目标算术：要从 1.24/h 降到 0.05/h，需否掉 96% 的误报，
且 17 次真发作一个都不能否。这是很高的门槛，先验证再谈集成。

安全兜底（fail-open）
-------------------
任何异常 —— 超时、HTTP 错、JSON 解析失败、字段缺失、置信度不足 ——
统一返回 keep=True（保留报警）。宁可多响一次，不可漏一次。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_ENDPOINT = "https://api.deepseek.com/anthropic/v1/messages"
DEFAULT_MODEL = "deepseek-chat"
KEY_FILE = os.path.expanduser("~/.config/nightguard/deepseek.key")


def load_api_key() -> Optional[str]:
    """环境变量优先，其次本地 600 权限文件。拿不到返回 None（调用方 fail-open）。"""
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


@dataclass
class AdjudicatorConfig:
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout_s: float = 25.0
    # 只有 LLM 明确判"非发作"且置信度 >= 这个值才否决。
    # 调高 = 更保守 = 更接近纯引擎行为。
    veto_min_confidence: float = 0.80
    # 真发作的典型时长下限：引擎的 sustain 已是 45s，这里只作为提示给模型。
    max_retries: int = 1


@dataclass
class Verdict:
    keep: bool                      # True = 保留报警
    reason: str = ""
    confidence: float = 0.0
    label: str = ""                 # seizure / artifact / unsure
    fail_open: bool = False         # True = 因异常兜底，不是模型的判断
    raw: str = ""


SYSTEM_PROMPT = """你是夜间癫痫监测系统的二级复核员。初筛引擎已经报警，你判断这是真的运动性发作，还是伪影/正常生理波动。

判据（来自临床文献与本系统的实测）：
真发作（强直-阵挛/局灶运动性）的典型形态
- 体动突然起始，不是缓慢爬升
- 1.5-5Hz 的节律性成分占比高（阵挛期），且能持续数十秒
- 心率在发作起始后快速上升并维持，不是单峰尖刺
- 持续 20 秒以上，常见 30-120 秒
- 结束后常有一段心率仍高、体动已停的"发作后"状态

伪影/非发作的典型形态
- 翻身/调整姿势：体动幅度可以很大但**无节律性**，持续 2-10 秒就结束
- 单纯心率波动（做梦/呼吸事件/体位性）：心率动了但体动几乎没有
- 节律性但幅度很小：可能是搔抓、腿部周期性运动（PLMS），幅度远低于发作
- 传感器接触不良：数值跳变突兀、前后不连续、hr_q 质量标记差
- 缓慢起始缓慢结束的低幅波动

只输出一个 JSON 对象，不要任何其他文字：
{"label":"seizure"|"artifact"|"unsure","confidence":0.0-1.0,"reason":"一句话中文依据"}

判断规则
- 像发作、或证据不足、或形态罕见无法归类 → label 用 "seizure" 或 "unsure"，报警会保留
- 只有当你能明确指出是哪一类伪影时才用 "artifact"，并给出 >=0.8 的 confidence

**低体动发作必须保留（这条优先于上面所有伪影判据）**
有一类真发作体动极小 —— 强直期、仅自主神经症状的发作，主要表现是
心率骤升/骤降而几乎不动。这类发作在夜间最危险，而且**正是靠心率通道
抓到的**。所以：
- 当体动很小但心率相对基线有明确、持续的异常时，**不许**判 artifact，
  即使体动"无节律""幅度不足"也不许 —— 体动看不见不等于没发作。
- "体动幅度极小 + 无节律" 这组理由只能用于**心率也正常**的事件。
- 心率通道有缺失窗或低置信窗时，不要据此判"传感器接触不良"进而否决：
  缺数据是"看不见"，不是"没事"。此时应判 unsure 保留。
实测教训：曾有一次真发作因"体动 RMS 中位 0.003、无节律爆发、
心率跳变疑似接触不良"被否决，而那恰恰是靠心率抓到的低体动发作。

**漏掉一次真发作的代价远大于多响一次铃。** 拿不准就不要否决。"""


def _fmt_series(vals: list, nd: int = 2, every: int = 1) -> str:
    out = []
    for i, v in enumerate(vals):
        if i % every:
            continue
        out.append("—" if v is None else f"{float(v):.{nd}f}")
    return " ".join(out)


def build_event_brief(event: dict) -> str:
    """把一个候选事件压成紧凑文本。

    event 需要:
      t0            报警时刻（秒）
      windows       事件窗口列表，每项含 t/motion/motion_psd/hr/hr_q
      risk_peak     引擎峰值风险
      rules_met     成立的临床规则名列表
      baseline      {'motion':..,'hr':..} 该患者的夜间基线
    """
    wins = event.get("windows") or []
    base = event.get("baseline") or {}
    dur = (wins[-1]["t"] - wins[0]["t"]) if len(wins) > 1 else 0.0

    # 2s 窗太密，降采样到约 30 个点，够看形态又不浪费 token
    every = max(1, len(wins) // 30)

    mot = [w.get("motion") for w in wins]
    psd = [w.get("motion_psd") for w in wins]
    hr = [w.get("hr") for w in wins]
    hrq = [w.get("hr_q") for w in wins]

    def _stat(xs):
        v = [float(x) for x in xs if x is not None]
        if not v:
            return "无数据"
        return f"最小 {min(v):.3f} / 中位 {sorted(v)[len(v)//2]:.3f} / 最大 {max(v):.3f}"

    bm = base.get("motion")
    bh = base.get("hr")
    lines = [
        f"事件时长：{dur:.0f} 秒（{len(wins)} 个 2 秒窗）",
        f"引擎峰值风险：{event.get('risk_peak', 0):.3f}",
        f"成立的临床规则：{', '.join(event.get('rules_met') or []) or '无'}",
        "",
        f"该患者夜间基线：体动 {bm:.4f}" if bm is not None else "该患者夜间基线：体动 未知",
        f"                心率 {bh:.1f} bpm" if bh is not None else "                心率 未知",
        "",
        f"体动 RMS（{_stat(mot)}）",
        f"  {_fmt_series(mot, 3, every)}",
        "",
        f"1.5-5Hz 节律功率占比 0-1（{_stat(psd)}）",
        f"  {_fmt_series(psd, 2, every)}",
        "",
        f"心率 bpm（{_stat(hr)}）",
        f"  {_fmt_series(hr, 0, every)}",
    ]
    # hr_q 语义（见 wearable_features.py:138）：1.0 = 高置信，0.0 = 低置信。
    # 不要搞反 —— 搞反会让提示词告诉模型"忽略心率"，而那恰是最可用的通道。
    bad = sum(1 for q in hrq if q is not None and float(q) < 0.5)
    miss = sum(1 for x in hr if x is None)
    if bad or miss:
        lines.append("")
        frac = bad / max(1, len(hrq))
        lines.append(
            f"心率通道质量：{len(hrq)-bad-miss}/{len(hrq)} 窗高置信，"
            f"{bad} 窗低置信，{miss} 窗无心率。")
        if frac > 0.5:
            lines.append(
                "  ⚠ 该事件过半窗口心率低置信，心率证据请降权，"
                "优先依据体动幅度与节律性形态判断。")
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse(text: str) -> Optional[dict]:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def adjudicate(event: dict, cfg: AdjudicatorConfig = None,
               api_key: str = None) -> Verdict:
    """复核一个候选报警。任何异常都 keep=True。"""
    cfg = cfg or AdjudicatorConfig()
    key = api_key or load_api_key()
    if not key:
        return Verdict(True, "无 API key，保留报警", fail_open=True)

    brief = build_event_brief(event)
    body = json.dumps({
        "model": cfg.model,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": brief}],
    }).encode()

    last_err = ""
    for _ in range(max(1, cfg.max_retries + 1)):
        try:
            req = urllib.request.Request(
                cfg.endpoint, data=body, method="POST",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                payload = json.loads(resp.read().decode())
            blocks = payload.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            d = _parse(text)
            if not d:
                last_err = "返回无法解析为 JSON"
                continue
            label = str(d.get("label", "")).lower().strip()
            try:
                conf = float(d.get("confidence", 0))
            except Exception:
                conf = 0.0
            reason = str(d.get("reason", ""))[:200]
            # 只有"明确是伪影且足够自信"才否决
            veto = (label == "artifact" and conf >= cfg.veto_min_confidence)
            return Verdict(keep=not veto, reason=reason, confidence=conf,
                           label=label, raw=text[:400])
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
        except Exception as e:                      # 超时/网络/解析
            last_err = f"{type(e).__name__}"
    return Verdict(True, f"复核不可用（{last_err}），保留报警", fail_open=True)
