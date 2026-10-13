"""体位（躺 / 非躺）—— NightWatch 降 5 倍误报的那一条。

为什么必须有这路
----------------
Lazeron 2022（Epilepsia Open 7:406-413，NightWatch 儿童版）发现：
  · 儿童版初版误报 1.27/h（均值），**>90% 的误报来自心率分支**
  · 查下来大多是"孩子下床、在床上玩"——清醒活动，不是发作
  · 他们没有去调心率阈值，而是加了一个**体位门**：
    只有佩戴者处于**水平体位**时才允许报警
修完的效果（同一批孩子，前瞻验证 497 夜）：
  误报 1.27/h → 0.24/h（降 5 倍）
  PPV 中位 11.1% → 58.1%
  每患者中位灵敏度 75.4% → **93.2%**（反而升了）
灵敏度不降反升，因为砍掉的全是清醒活动期的噪声。

原文的最终判据是：
  水平体位 **且**（心率超斜率/绝对阈 **或** 运动值连续超阈 ≥7 秒）

怎么算体位（这里是本模块的难点）
--------------------------------
体动特征把重力减掉了（窗内减中位数），而**体位信息正在重力分量里**：
加速度计静止时测到的就是重力方向，也就是设备相对地面的朝向。

但我们不知道这批数据里设备是怎么贴的（哪个轴朝头顶），所以不能
写死"Z 轴朝上"。做法是**按记录自标定**（与项目里其他通道一致，
见 [[nightguard-absolute-thresholds-fail]] 的教训）：

  人躺着翻身时，重力矢量在**垂直于身体长轴的平面**内扫来扫去
  （仰卧、侧卧差 90°，但都是躺）；而站/坐时重力**沿着长轴**。
  所以整夜重力矢量**方差最小的方向 ≈ 身体长轴**。
  求出长轴后：
    倾角 = arcsin(|重力·长轴| / |重力|)
    躺着 → 倾角 ≈ 0°；站/坐 → 倾角 ≈ 90°

这样无论仰卧还是侧卧都算"躺"，只有真正立起来才出门限 —— 这是
单用"偏离某个固定朝向多少度"做不到的（那样侧卧会被误判成起床）。

长轴标定在 validation 侧用协方差最小特征向量算（那边有 numpy），
本模块只做运行时的廉价计算：给定长轴，算倾角、判是否水平。
这也更像真实产品：一次性标定 + 逐秒廉价判断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PostureConfig:
    # 倾角小于这个角度算"躺"。45° 是躺与立的几何中点，
    # 留出足够余量容纳半躺（靠枕头坐起来一点仍算躺，不漏报）。
    horiz_max_deg: float = 45.0
    # 事件窗内"躺"的窗数占比至少这么多才算这次事件发生在床上。
    # 不要求 100%：强直-阵挛本身会让躯干大幅位移，
    # 个别窗的重力估计会被加速度污染。
    horiz_frac_min: float = 0.5


def gravity_of_window(ax, ay, az):
    """一个窗的重力矢量估计 = 三轴各自的均值。

    均值把发作性抖动（零均值的高频成分）平掉，剩下的是重力。
    """
    n = len(ax)
    if n == 0 or len(ay) != n or len(az) != n:
        return None
    return (sum(ax) / n, sum(ay) / n, sum(az) / n)


def tilt_deg(g, long_axis):
    """重力矢量与身体长轴的夹角偏离水平多少度。

    返回 0 = 完全躺平，90 = 完全立起。long_axis 需为单位矢量。
    """
    if g is None or long_axis is None:
        return None
    gm = math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    if gm < 1e-9:
        return None
    dot = abs(g[0] * long_axis[0] + g[1] * long_axis[1] + g[2] * long_axis[2])
    # 数值安全：dot/gm 可能因浮点误差略超 1
    r = min(1.0, dot / gm)
    return math.degrees(math.asin(r))


def is_horizontal(tilt, cfg: PostureConfig = None):
    """单窗是否算躺。tilt=None（算不出）时返回 None，**不要当成 False**。

    踩坑提醒：把"测不到"当成"没躺"会直接漏报。缺数据一律降级为放行，
    由上层决定怎么处理（见 engine/gate.py 的同类处理）。
    """
    cfg = cfg or PostureConfig()
    if tilt is None:
        return None
    return tilt <= cfg.horiz_max_deg


def horizontal_fraction(tilts, cfg: PostureConfig = None):
    """一串窗里"躺"的占比；全部算不出时返回 None。"""
    cfg = cfg or PostureConfig()
    # tilts 本身可能是 None（上游连倾角序列都没算出来）。
    # 这是安全门：任何"算不出"都必须降级放行，不能抛异常 ——
    # 抛出去会让整条报警链路挂掉，等于静默漏报。
    vals = [is_horizontal(t, cfg) for t in (tilts or [])]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def posture_gate(tilts, cfg: PostureConfig = None):
    """体位门：这次事件是否发生在"躺着"的状态下。

    返回 (keep, frac)。算不出体位时 keep=True（降级放行，不漏报）。
    """
    cfg = cfg or PostureConfig()
    frac = horizontal_fraction(tilts, cfg)
    if frac is None:
        return True, None
    return frac >= cfg.horiz_frac_min, frac
