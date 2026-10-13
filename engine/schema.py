"""
schema.py — 统一回放格式与信号可用性声明（第八棒,框架层）

为什么需要这一层
================
第七棒之前的致命结构问题:**"这路信号不存在"和"这路信号正常"无法区分**。

SeizeIT2(A 组主力数据集,94% 那个成绩)只有 ACC + ECG 两路,
没有血氧、没有呼吸。但引擎里 `features.get("spo2")` 返回 None 时,
该信号直接跳过融合 —— 于是风险自然偏低,判级偏轻,
而报告里写的是"多模态融合"。等于拿两路信号的成绩,
宣称四路信号的架构。

更糟的是产品层面:家长看到"一切正常",无法知道
究竟是"血氧正常"还是"这台设备根本没有血氧探头"。

所以把通道状态做成**三态**,而不是"有值/无值":

    PRESENT    通道存在且本窗取到可用值
    DROPOUT    通道存在,但本窗质量不过关(探头脱落、运动伪差、噪声段)
               → 这是看门狗要管的"看不见",必须让家长知道
    ABSENT     设备根本没有这路传感器
               → 不是异常,但必须在报告里声明"本次结论未包含血氧"

判据覆盖也靠它:医生给的大发作判据要求心率+呼吸+血氧三路,
如果血氧 ABSENT,那条判据的结论就只能标为"部分满足",
不能假装满足。宁可说"我看不全",不能假装看全了。

用法
====
    from engine.schema import Window, Channels, ABSENT

    # 数据加载器声明设备能力(一次)
    caps = Channels(hr=True, motion=True, spo2=False, breathing=False, temp=False)

    # 逐窗构造
    w = Window.build(t=1234.0, caps=caps, hr=78.0, motion=0.4)
    w.status("spo2")       # → "absent"
    w.status("hr")         # → "present"
    w.declared_missing()   # → ("spo2", "breathing", "temp")  用于报告声明
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
import math

PRESENT = "present"
DROPOUT = "dropout"
ABSENT = "absent"

# 回放格式的标准字段(开源资产调研里推荐的统一格式,照它来)
# timestamp, patient_id, event_id, acc/gyro, heart_rate, spo2,
# respiration, temperature, signal_quality, device_worn, label
VITAL_FIELDS = ("hr", "motion", "gyro", "breathing", "spo2", "temp")
# EEG 侧字段(研究支线用,家庭产品不含)
EEG_FIELDS = ("amp", "power", "rhythm", "spread")
ALL_FIELDS = VITAL_FIELDS + EEG_FIELDS


def _ok(v: Any) -> bool:
    """有值且不是 NaN。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


@dataclass(frozen=True)
class Channels:
    """设备能力声明:这台设备**物理上**有哪些传感器。

    与"本窗有没有取到值"是两件事 —— 后者看 Window.status()。
    默认全 False,必须由数据加载器显式声明,避免"忘了声明=当作有"。
    """
    hr: bool = False
    motion: bool = False
    # 角速度（陀螺仪）。和 motion（加速度）是**两种物理量**，不可互替:
    # 强直期是等长收缩 —— 肌肉绷紧、身体不位移，加速度计几乎看不见，
    # 但伴随姿态扭转，角速度看得见。实测（validation/run_r12_hw_config.py）
    # 仅 ACC+心率 致命级 21/23，加陀螺仪 → 23/23。
    # 位置无关:耳后与胸部陀螺仪实测等效（run_r15_chest_gyr.py，均 2 次 / 6 桶）。
    gyro: bool = False
    breathing: bool = False
    spo2: bool = False
    temp: bool = False
    amp: bool = False
    power: bool = False
    rhythm: bool = False
    spread: bool = False

    def has(self, key: str) -> bool:
        return bool(getattr(self, key, False))

    def present_fields(self) -> Tuple[str, ...]:
        return tuple(k for k in ALL_FIELDS if self.has(k))

    def absent_fields(self) -> Tuple[str, ...]:
        return tuple(k for k in ALL_FIELDS if not self.has(k))

    @classmethod
    def wearable_full(cls) -> "Channels":
        """PRD 设想的完整腕表:心率+体动+呼吸+血氧+皮温。"""
        return cls(hr=True, motion=True, breathing=True, spo2=True, temp=True)

    @classmethod
    def seizeit2(cls) -> "Channels":
        """SeizeIT2:胸部 ACC + ECG。**没有**血氧和呼吸 —— 这是 A 组成绩的真实边界。

        注:mov.edf 里**另有**耳后/胸部各三轴陀螺仪（见 gyro 字段注释），
        前十四轮一直没用。要带陀螺仪请用 `seizeit2_gyro()`。
        """
        return cls(hr=True, motion=True)

    @classmethod
    def seizeit2_gyro(cls) -> "Channels":
        """SeizeIT2 + 陀螺仪 —— 致命级 23/23 所需的最小通道集。"""
        return cls(hr=True, motion=True, gyro=True)

    @classmethod
    def home_device(cls) -> "Channels":
        """拟采购的居家设备:胸贴（ACC + GYR + 单导 ECG）。

        这是 r12/r15 实测支撑的硬件下限 —— 不需要耳后电极、不需要 EEG/EMG。
        但注意 EEG/EMG 是**压误报**的那一路（42.0 → 6.7 次/10夜），
        仅凭本配置误报会高，须配合 LLM 裁决与分级提示。
        """
        return cls(hr=True, motion=True, gyro=True)

    @classmethod
    def ucd(cls) -> "Channels":
        """UCD 睡眠呼吸库:ECG + 血氧 + 气流/呼吸努力。没有体动。"""
        return cls(hr=True, breathing=True, spo2=True)

    @classmethod
    def chbmit(cls) -> "Channels":
        """CHB-MIT:单导联 EEG 派生的四个特征。无可穿戴生命体征。"""
        return cls(amp=True, power=True, rhythm=True, spread=True)


@dataclass
class Window:
    """一个分析窗口。字段与回放格式一一对应。

    `values` 只放**本窗真正可用**的值;通道存在但质量不过关的进 `dropouts`。
    引擎读值统一走 `get()`,所以 DROPOUT 和 ABSENT 都拿不到值 ——
    但 `status()` 能区分二者,这正是看门狗和报告声明需要的。
    """
    t: Optional[float]
    caps: Channels
    values: Dict[str, float] = field(default_factory=dict)
    dropouts: Tuple[str, ...] = ()
    patient_id: Optional[str] = None
    event_id: Optional[str] = None
    label: Optional[str] = None          # 真值标注(验证用,引擎不读)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, t, caps: Channels, *, patient_id=None, event_id=None,
              label=None, meta=None, **kv) -> "Window":
        """从散字段构造。通道存在但传 None/NaN → 记为 DROPOUT;
        通道不存在 → 忽略该字段(即便传了值也不采信,避免悄悄多出一路信号)。"""
        vals, drops = {}, []
        for k in ALL_FIELDS:
            if not caps.has(k):
                continue
            v = kv.get(k)
            if _ok(v):
                vals[k] = float(v)
            else:
                drops.append(k)
        return cls(t=t, caps=caps, values=vals, dropouts=tuple(drops),
                   patient_id=patient_id, event_id=event_id, label=label,
                   meta=dict(meta or {}))

    # ---- 读取 ----
    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def status(self, key: str) -> str:
        if not self.caps.has(key):
            return ABSENT
        return PRESENT if key in self.values else DROPOUT

    def declared_missing(self) -> Tuple[str, ...]:
        """设备根本没有的通道 —— 报告里必须声明"本结论未包含这些"。"""
        return self.caps.absent_fields()

    def live_fields(self) -> Tuple[str, ...]:
        return tuple(k for k in ALL_FIELDS if k in self.values)

    def is_blind(self) -> bool:
        """所有**本该有**的通道都没取到值 → 真正的"看不见"。

        关键:ABSENT 的通道不参与判断。只有一路血氧的设备,
        血氧掉了就是全盲;而没有血氧探头的设备不该因此报看护失效。
        """
        expected = self.caps.present_fields()
        if not expected:
            return False
        return len(self.values) == 0

    # ---- 兼容层:现有 signals.py / gate.py 吃 dict ----
    def as_features(self) -> dict:
        """转成旧引擎吃的 dict。额外塞入 `_window` 反向引用,
        让需要通道状态的新代码(临床规则层)能拿到三态信息。"""
        d = {"t": self.t}
        d.update(self.values)
        d["_window"] = self
        return d


def features_window(features: dict) -> Optional[Window]:
    """从旧式 features dict 里取回 Window(没有则 None)。

    这让临床规则层可以渐进接入:有 Window 就用三态,
    没有就退化成"有值/无值"的旧行为,不破坏现存调用。
    """
    w = features.get("_window")
    return w if isinstance(w, Window) else None


def caps_of(features: dict) -> Channels:
    """拿不到 Window 时,按"有值即存在"反推一个能力声明(保守兼容)。"""
    w = features_window(features)
    if w is not None:
        return w.caps
    return Channels(**{k: _ok(features.get(k)) for k in ALL_FIELDS})
