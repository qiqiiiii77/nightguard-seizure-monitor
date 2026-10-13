"""
export_demo_data.py — 生成 Demo 仪表盘用的整夜数据（含引擎逐窗判定）

输出 demo/night_data.js：一条 6 小时模拟夜（2 次发作 + 3 类正常陷门），
含 4 路生理信号、风险分、门控级别、证据链、事件日志。
现场演示可复现（固定种子）。
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.gate import NightGate
from engine.signals import wearable_signals
from sim.simulator import generate_night, windowize

SEED = 2026
MINUTES = 360            # 6 小时（演示用，完整夜任一分钟数）
SEIZURES = [
    {"onset_min": 95, "dur_s": 70, "intensity": 1.0},
    {"onset_min": 208, "dur_s": 55, "intensity": 0.75},
]

night = generate_night(seed=SEED, minutes=MINUTES, seizures=SEIZURES, traps=True)
wins = windowize(night)
ng = NightGate(signal_fn=wearable_signals)

frames = []
for w in wins:
    r = ng.step(w)
    frames.append({
        "t": w["t"],
        "m": round(w["motion"], 2),
        "r": round(w["motion_psd"], 2),
        "br": round(w["breathing"], 1),
        "hr": round(w["hr"], 1),
        "sp": round(w["spo2"], 1),
        "risk": r.risk,
        "level": r.level,
        "driver": r.driver,
        "probe": r.probe,
        "evidence": r.evidence,
        "confirmed": r.confirmed,
        "truth": w["label"],
    })

out = {
    "meta": {
        "seed": SEED, "minutes": MINUTES,
        "seizures": SEIZURES,
        "win_s": 8, "hop_s": 4,
    },
    "frames": frames,
    "journal": ng.journal.events,
    "journal_summary": ng.journal.summary(),
    "baseline": ng.baseline.snapshot(),
}

path = os.path.join(os.path.dirname(__file__), "night_data.js")
with open(path, "w") as f:
    f.write("const NIGHT_DATA = ")
    json.dump(out, f, ensure_ascii=False)
    f.write(";\n")

truth_wins = sum(1 for w in wins if w["label"] == 1)
det = [f for f in frames if f["level"] == "CRITICAL"]
print(f"窗口 {len(frames)}（发作窗 {truth_wins}）")
print(f"事件日志：{ng.journal.summary()}")
print(f"首个 CRITICAL @ {det[0]['t'] if det else None}s")
print(f"→ {path}  ({os.path.getsize(path)/1024:.0f} KB)")
