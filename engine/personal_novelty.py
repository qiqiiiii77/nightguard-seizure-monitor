"""个体化 novelty detection：学"这个孩子的正常"，把发作当离群点。

与之前两次失败的个性化有什么不同
--------------------------------
`文档/思路与踩坑.md:162` 记录了两次失败，结论是"自适应放基线层，
阈值保持全局，这个方向别再试"。那两次调的都是**一个标量阈值**：
拿首夜无发作数据把风险分最高的 0.5% 定为个人报警线。
全量 23 人复核：全局 49%/1.45次h vs 个人 50%/1.14次h，只换 1 个点。
失败原因文档写得很清楚 —— 风险分已经是"跟自己比"的归一化分数，
再按人调阈值是把同一件事做两遍。

本模块做的是不同的事：**在多维特征空间里学该患者正常状态的分布形状**。
依据 De Cooman 2018（Seizure 59:48-53）：患者无关算法起步，
短初始化期后用 low-complexity novelty detection 适应患者，
**不需要发作标签**，同灵敏度下误报比患者无关算法少 57%。

为什么这次有机会成立
  · 之前失败的是"阈值"这一个自由度；这里学的是协方差结构 ——
    "心率升但体动没动"对安静型孩子是异常，对好动型孩子是常态。
    这种**特征间关系**是标量阈值表达不了的。
  · 训练只需无发作背景（我们有 290 个 run 可取），不需要发作标签，
    所以绕开了"13/15 患者只有 1 次发作"这个致命约束。

架构位置：引擎之后，只做减法
---------------------------
和 LLM 裁决一样挂在引擎候选事件之后，只允许否决。引擎 θ=0.40 的
100% 召回是上界，个体化失灵时按"保留报警"降级。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 事件级特征。刻意不含绝对体动/心率值 —— 那些个体差异太大，
# 要的是"相对这个孩子平时"的偏离形态。
# 全部特征必须**与事件时长无关**。
# 为什么（这是第一版的方法论 bug）：训练侧是固定 60 秒的滑窗伪事件，
# 测试侧是引擎候选事件，长度 20-900 秒不等。只要特征含绝对秒数
# （原来的 psd_sustain / hr_sustain / dur），训练和测试就不在同一尺度上，
# 测试分数必然系统性偏高 —— 实测 124/180 个事件分位饱和在 1.0，
# 门槛因此几乎失效。改成**占比**后两侧可比。
FEATURES = [
    "mot_peak_ratio",      # 体动峰值 / 个人体动基线
    "mot_rise",            # 体动上升陡度（突然起始 vs 缓慢爬升）
    "psd_peak",            # 1.5-5Hz 节律占比峰值
    "psd_frac",            # 节律占比 >0.3 的窗数占比（时长无关）
    "hr_delta",            # 心率峰值 - 个人心率基线
    "hr_frac",             # 心率高于基线+20 的窗数占比（时长无关）
    "cross_sync",          # 体动与心率同步上升的程度
]


def _safe(x, d=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def event_features(windows: list, baseline: dict) -> np.ndarray:
    """把一个事件（窗口序列）压成固定维特征向量。

    baseline 用**该患者自己**的中位数，所以所有比值都是"跟自己比"。
    """
    t = np.array([_safe(w.get("t")) for w in windows])
    mot = np.array([_safe(w.get("motion")) for w in windows])
    psd = np.array([_safe(w.get("motion_psd")) for w in windows])
    hr_raw = [w.get("hr") for w in windows]
    hrq = [w.get("hr_q") for w in windows]
    # hr_q: 1.0=高置信, 0.0=低置信（见 wearable_features.py:138）
    hr = np.array([_safe(h) for h, q in zip(hr_raw, hrq)
                   if h is not None and (q is None or _safe(q, 1.0) >= 0.5)])

    bm = max(_safe(baseline.get("motion"), 1e-4), 1e-6)
    bh = _safe(baseline.get("hr"), 70.0)

    mot_peak_ratio = float(mot.max() / bm) if mot.size else 0.0

    # 上升陡度：峰值前最大的单窗增量，按个人基线归一
    if mot.size > 2:
        pk = int(mot.argmax())
        seg = mot[: pk + 1]
        mot_rise = float(np.diff(seg).max() / bm) if seg.size > 1 else 0.0
    else:
        mot_rise = 0.0

    psd_peak = float(psd.max()) if psd.size else 0.0
    # 占比而非秒数（0.3 来自引擎 clinical_config 的量级）
    psd_frac = float((psd > 0.3).mean()) if psd.size else 0.0

    hr_delta = float(hr.max() - bh) if hr.size else 0.0
    hr_frac = float((hr > bh + 20).mean()) if hr.size else 0.0

    # 跨模态同步：体动与心率在时间上一起升的相关性。
    # 这是 INNOVATION A2 (cross_modal_sync) 的事件级版本。
    #
    # 原来写 mot[:n] vs hr[:n] 是错的 —— hr 已被质量门过滤过，
    # 两个数组的第 k 个元素不是同一时刻，相关性算的是错位序列。
    # 这里改为只在**两者都有效的同一窗**上配对。
    cross = 0.0
    pair_m, pair_h = [], []
    for w in windows:
        h, q = w.get("hr"), w.get("hr_q")
        m = w.get("motion")
        if h is None or m is None:
            continue
        if q is not None and _safe(q, 1.0) < 0.5:
            continue
        pair_m.append(_safe(m))
        pair_h.append(_safe(h))
    if len(pair_m) >= 4:
        a = np.array(pair_m)
        b = np.array(pair_h)
        if a.std() > 1e-9 and b.std() > 1e-9:
            c = float(np.corrcoef(a, b)[0, 1])
            cross = c if math.isfinite(c) else 0.0

    return np.array([mot_peak_ratio, mot_rise, psd_peak, psd_frac,
                     hr_delta, hr_frac, cross], dtype=float)


@dataclass
class PersonalModel:
    """单患者的正常分布模型。

    用 Mahalanobis 距离做 novelty score ——「low-complexity」是刻意的：
    样本少（每人几十个背景事件），复杂模型必然过拟合。
    De Cooman 那篇用的也是低复杂度分类器。
    """
    mu: Optional[np.ndarray] = None
    prec: Optional[np.ndarray] = None       # 协方差的伪逆
    n_train: int = 0
    train_scores: list = field(default_factory=list)
    feature_names: list = field(default_factory=lambda: list(FEATURES))

    def fit(self, X: np.ndarray) -> "PersonalModel":
        X = np.asarray(X, dtype=float)
        X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] < 5:
            return self                      # 样本太少，保持未拟合（score 恒 0）
        # log1p 压量级（体动比值能到几百，时长到上千）
        Z = np.log1p(np.clip(X, 0, None))
        self.mu = Z.mean(axis=0)
        C = np.cov(Z, rowvar=False)
        # 收缩正则：样本少时协方差不稳，往对角线收
        d = C.shape[0]
        lam = 0.3
        C = (1 - lam) * C + lam * np.eye(d) * np.trace(C) / max(d, 1)
        self.prec = np.linalg.pinv(C)
        self.n_train = int(X.shape[0])
        self.train_scores = sorted(float(self._maha(z)) for z in Z)
        return self

    def _maha(self, z: np.ndarray) -> float:
        dv = z - self.mu
        return float(math.sqrt(max(0.0, dv @ self.prec @ dv)))

    def score(self, x: np.ndarray) -> float:
        """novelty 分数：越大越不像这个孩子的常态。未拟合时返回 0。"""
        if self.mu is None or self.prec is None:
            return 0.0
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            return 0.0
        return self._maha(np.log1p(np.clip(x, 0, None)))

    def percentile_of(self, s: float) -> float:
        """该分数在训练分布里的分位（0-1）。用于设"比平时罕见多少"的门槛。"""
        if not self.train_scores:
            return 0.0
        import bisect
        return bisect.bisect_left(self.train_scores, s) / len(self.train_scores)

    def to_dict(self) -> dict:
        return {"mu": None if self.mu is None else self.mu.tolist(),
                "prec": None if self.prec is None else self.prec.tolist(),
                "n_train": self.n_train,
                "train_scores": self.train_scores,
                "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalModel":
        m = cls(n_train=int(d.get("n_train", 0)),
                train_scores=list(d.get("train_scores") or []),
                feature_names=list(d.get("feature_names") or FEATURES))
        if d.get("mu") is not None:
            m.mu = np.array(d["mu"], dtype=float)
        if d.get("prec") is not None:
            m.prec = np.array(d["prec"], dtype=float)
        return m


def save_models(models: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in models.items()}, f)


def load_models(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {k: PersonalModel.from_dict(v) for k, v in json.load(f).items()}
