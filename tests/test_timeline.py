#!/usr/bin/env python3
"""第十八棒：夜间事件时间轴（赛题功能需求 3）的语义约束。

锁住的是**赛题原文里的硬性数字和规则**，不是实现细节：
· 持续 >2 分钟必须被标记（赛题需求 2 的硬性规则）
· 发作后高风险期 5-10 分钟（取上界 600s，与 clinical_config 一致）
· 非惊厥发作（LOG 级）必须留档但不报警（赛题技术可行性一节）
· 归并窗复用 events.DEFAULT_MERGE_S，不另造一个
· 落盘字段里没有姓名位（赛题 solutions/README.md 的隐私红线）
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.events import DEFAULT_MERGE_S                          # noqa: E402
from engine.timeline import (LONG_SEIZURE_S, POST_ICTAL_S,         # noqa: E402
                             NightTimeline, TimelineEntry,
                             VitalSignature, weekly_trend)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def win(t, level="CRITICAL", risk=0.9):
    return {"start": t, "end": t + 2.0, "level": level, "risk": risk}


print("== 赛题硬性数字 ==")
check("持续状态门槛 = 2 分钟（赛题需求 2）", LONG_SEIZURE_S == 120.0)
check("发作后高风险期 = 10 分钟上界（赛题 5-10 分钟）",
      POST_ICTAL_S == 600.0)
check("归并窗复用 events 的 300 秒，不另造",
      NightTimeline("x").merge_s == DEFAULT_MERGE_S == 300.0)

print()
print("== 时长标记 ==")
_short = TimelineEntry(t0=0, t1=60, level="CRITICAL")
_long = TimelineEntry(t0=0, t1=130, level="CRITICAL")
check("60 秒不标持续状态", not _short.is_long)
check("130 秒标持续状态（≥2 分钟）", _long.is_long)
check("恰好 120 秒也算（用 >=，赛题说「超过 2 分钟」取闭区间更安全）",
      TimelineEntry(t0=0, t1=120, level="CRITICAL").is_long)
check("发作后高风险期截止 = 结束 + 600s",
      _long.post_ictal_until == 130 + 600.0)

print()
print("== 归并：逐窗 → 事件 ==")
tl = NightTimeline("child-01", night=3)
# 一次 2 分钟发作（逐窗 2s，共 60 窗）+ 隔很久的另一次短事件
burst = [win(t) for t in range(0, 130, 2)]
later = [win(t) for t in range(1000, 1010, 2)]
tl.build_from_windows(burst + later)
check("60 个连续窗归并成 1 次事件", len(tl.entries) == 2,
      f"→ {len(tl.entries)} 次")
check("第一次事件时长 ≈130 秒且被标为持续状态",
      tl.entries[0].duration_s >= 128 and tl.entries[0].is_long,
      f"→ {tl.entries[0].duration_s}")
check("间隔 >300 秒判为独立事件",
      tl.entries[1].t0 >= 1000)
check("持续状态事件自动写备注",
      "2 分钟" in tl.entries[0].note)

print()
print("== 非惊厥发作：留档但不报警（赛题技术可行性一节）==")
tl2 = NightTimeline("child-01")
tl2.build_from_windows([win(10, "LOG", 0.2), win(2000, "CRITICAL", 0.9)])
check("LOG 级事件进时间轴（不能丢档）", len(tl2.entries) == 2,
      f"→ {len(tl2.entries)}")
s2 = tl2.summary()
check("汇总能分出「仅记录」与「响铃」", s2["n_log_only"] == 1,
      f"→ {s2}")
check("未接 tier_fn 时默认未响铃（不许默认响）",
      all(not e.rang for e in tl2.entries))

print()
print("== 接 tiers.py ==")
from engine.tiers import decide                                    # noqa: E402

tl3 = NightTimeline("child-01")
tl3.build_from_windows(
    [win(t) for t in range(0, 130, 2)],
    tier_fn=lambda e: (lambda d: (d["tier"], d["rings"]))(
        decide(support=2)))
check("tier 与响铃状态写入条目",
      tl3.entries[0].tier == "T1" and tl3.entries[0].rang is True)
check("汇总按 tier 计数", tl3.summary()["per_tier"] == {"T1": 1},
      f"→ {tl3.summary()['per_tier']}")

print()
print("== 体征特征：相对基线，缺值不编 ==")
v = VitalSignature(hr_rise_bpm=38.0, gyro_fraction=0.73,
                   channels_missing=["血氧"])
txt = v.to_text()
check("输出相对量而非绝对值", "高 38" in txt, f"→ {txt}")
check("缺通道如实写出", "血氧" in txt)
check("全缺时不编数字",
      VitalSignature().to_text() == "无可用体征特征")

print()
print("== 落盘与隐私红线 ==")
doc = tl3.export_json()
check("导出含 summary 与 entries", "summary" in doc and "entries" in doc)
check("条目含时长/持续状态/体征文本",
      {"duration_s", "is_long", "vitals_text"} <= set(doc["entries"][0]))
_fields = set(doc["entries"][0]) | set(doc["summary"])
check("字段里没有姓名/电话/病历位（隐私红线）",
      not any(k in _fields for k in
              ("name", "姓名", "phone", "电话", "record", "病历")),
      f"→ {sorted(_fields)}")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "t.json")
    tl3.export_json(p)
    check("能写文件且是合法 JSON",
          isinstance(json.load(open(p, encoding="utf-8")), dict))

print()
print("== 家长可读文本 ==")
t = tl3.to_text()
check("文本含相对时刻 h:mm:ss", "0:00:00" in t, f"→ {t[:80]}")
check("文本标出持续状态", "持续≥2分钟" in t or "≥2 分钟" in t)
check("空夜不报错且给出明确话术",
      "无可疑事件" in NightTimeline("child-01").to_text())

print()
print("== 需求冲突自检：只报不改判 ==")
# 赛题说非惊厥"仅作记录"，但本项目实测静默档藏住 26/29 次致命级。
# 冲突不自动解决 —— 暴露出来让人裁决。
tl_c = NightTimeline("child-01")
tl_c.build_from_windows([win(t, "LOG", 0.2) for t in range(0, 140, 2)])
a = tl_c.audit_consistency()
check("LOG 级却持续 ≥2 分钟 → 标为需人工复核",
      a["log_but_long"] == 1 and a["consistent"] is False, f"→ {a}")
check("冲突条目给出时刻便于定位", a["suspicious_t0"] == [0])
tl_ok = NightTimeline("child-01")
tl_ok.build_from_windows([win(10, "LOG", 0.2)])
check("短 LOG 事件不报冲突",
      tl_ok.audit_consistency()["consistent"] is True)
check("自检不改判（LOG 仍是 LOG、响铃状态不变）",
      tl_c.entries[0].level == "LOG")

print()
print("== 日/周趋势（赛题需求 3 后半句）==")
n1, n2 = NightTimeline("c"), NightTimeline("c")
n1.build_from_windows([win(t) for t in range(0, 130, 2)])
n2.build_from_windows([win(10), win(2000), win(4000)])
tr = weekly_trend([n1, n2])
check("逐夜事件数序列", tr["events_per_night"] == [1, 3],
      f"→ {tr['events_per_night']}")
check("统计持续状态总次数", tr["total_long_ge_2min"] == 1)
check("指出最差一夜（1-based）", tr["worst_night"] == 2)
check("空输入不崩", weekly_trend([])["n_nights"] == 0)

print()
print("=" * 40)
print(f"通过 {PASS} / 失败 {FAIL}")
sys.exit(1 if FAIL else 0)
