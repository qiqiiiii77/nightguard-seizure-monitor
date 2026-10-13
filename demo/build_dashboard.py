"""
build_dashboard.py — 生成自包含的演示仪表盘 demo/dashboard.html

把 night_data.js 的数据内联进单文件 HTML（本地双击/预览都能跑，无需服务器）。
夜视暗色主题：4 路生理信号 + 风险轨迹 + 门控级别 + 事件日志 + 回放。
"""

import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
data_js = open(os.path.join(HERE, "night_data.js"), encoding="utf-8").read()
data_json = data_js.split("=", 1)[1].strip().rstrip(";")

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>守夜 NightGuard · 癫痫患儿夜间监测演示</title>
<style>
  :root{
    --bg:#0b1220; --panel:#111c2e; --panel2:#16233a; --line:#22314d;
    --text:#e8eefb; --dim:#8ea2c4; --motion:#60a5fa; --hr:#f87171;
    --spo2:#34d399; --breath:#a78bfa; --risk:#fbbf24;
    --crit:#ef4444; --warn:#f59e0b; --log:#64748b; --truth:#f43f5e;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;font-size:13px}
  header{display:flex;align-items:center;gap:14px;padding:12px 18px;
    border-bottom:1px solid var(--line);background:linear-gradient(180deg,#0e1728,#0b1220)}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.5px}
  header .tag{font-size:11px;color:var(--dim);border:1px solid var(--line);
    padding:2px 8px;border-radius:10px}
  header a.tag{text-decoration:none;color:#7fb0ff;border-color:#2b4a7d}
  header a.tag:hover{background:#16233a;color:#a8c8ff}
  .wrap{display:grid;grid-template-columns:1fr 320px;gap:12px;padding:12px}
  .col{display:flex;flex-direction:column;gap:12px;min-width:0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:10px 12px;min-width:0}
  .card h2{font-size:12px;margin:0 0 8px;color:var(--dim);font-weight:600;
    letter-spacing:.4px;display:flex;justify-content:space-between;align-items:center}
  canvas{width:100%;display:block}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button{background:var(--panel2);color:var(--text);border:1px solid var(--line);
    border-radius:7px;padding:6px 12px;cursor:pointer;font-size:12px;font-family:inherit}
  button:hover{border-color:#3b5680}
  button.primary{background:#1d4ed8;border-color:#1d4ed8}
  .clock{font-variant-numeric:tabular-nums;font-size:20px;font-weight:600;letter-spacing:1px}
  .status{display:flex;align-items:center;gap:10px;padding:10px;border-radius:9px;
    background:var(--panel2);border:1px solid var(--line)}
  .dot{width:11px;height:11px;border-radius:50%;flex:none}
  .status .lab{font-size:16px;font-weight:700}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:12px;
    color:var(--dim);margin-top:8px}
  .kv b{color:var(--text);font-weight:600}
  .riskbar{height:8px;border-radius:5px;background:#1c2b45;overflow:hidden;margin-top:8px}
  .riskbar i{display:block;height:100%;width:0;background:var(--risk);transition:width .12s}
  .ev{font-size:12px;padding:7px 9px;border-radius:8px;border:1px solid var(--line);
    background:var(--panel2);margin-bottom:6px;cursor:pointer}
  .ev:hover{border-color:#3b5680}
  .ev .t{color:var(--dim);font-variant-numeric:tabular-nums}
  .ev .lv{font-weight:700;margin-left:6px}
  .ev .drv{color:var(--dim);margin-top:3px;font-size:11px}
  .ev.CRITICAL{border-left:3px solid var(--crit)}
  .ev.WARNING{border-left:3px solid var(--warn)}
  .log{max-height:290px;overflow:auto}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--dim)}
  .legend i{display:inline-block;width:10px;height:3px;border-radius:2px;
    margin-right:5px;vertical-align:middle}
  .probe{margin-top:8px;padding:8px 10px;border-radius:8px;background:#1b2740;
    border:1px dashed #35507d;color:#c9d8f5;font-size:12px;line-height:1.5}
  .foot{color:var(--dim);font-size:11px;padding:0 6px 14px;line-height:1.7}
  .badge{font-size:10px;padding:1px 6px;border-radius:8px;border:1px solid var(--line);color:var(--dim)}
</style>
</head>
<body>
<header>
  <h1>守夜 NightGuard</h1>
  <span class="tag">癫痫患儿夜间无感监测 · 多模态融合 + 分级门控</span>
  <span class="tag" id="src">模拟夜（种子 2026，6 小时，2 次发作注入）</span>
  <a class="tag link" href="validation.html">真实数据检验报告 →</a>
</header>

<div class="wrap">
  <div class="col">
    <div class="card">
      <h2><span>整夜趋势 · 四路生理信号</span><span class="legend">
        <span><i style="background:var(--motion)"></i>体动</span>
        <span><i style="background:var(--hr)"></i>心率</span>
        <span><i style="background:var(--spo2)"></i>血氧</span>
        <span><i style="background:var(--breath)"></i>呼吸</span>
        <span><i style="background:var(--truth)"></i>真实发作</span>
      </span></h2>
      <canvas id="cSignals" height="200"></canvas>
    </div>

    <div class="card">
      <h2><span>风险评分与门控判定（noisy-OR 融合 8 信号）</span>
        <span class="legend">
          <span><i style="background:var(--log)"></i>记录</span>
          <span><i style="background:var(--warn)"></i>观察通知</span>
          <span><i style="background:var(--crit)"></i>唤醒家长</span>
        </span></h2>
      <canvas id="cRisk" height="150"></canvas>
      <div class="controls" style="margin-top:10px">
        <button id="play" class="primary">暂停</button>
        <button data-speed="10">10×</button>
        <button data-speed="30">30×</button>
        <button data-speed="120">120×</button>
        <button id="prev">上一个事件</button>
        <button id="next">下一个事件</button>
        <button id="reset">回到开头</button>
        <span class="clock" id="clock">00:00:00</span>
      </div>
    </div>

    <div class="foot" id="foot"></div>
  </div>

  <div class="col">
    <div class="card">
      <h2><span>当前时刻</span><span class="badge" id="truthBadge">真相：正常</span></h2>
      <div class="status">
        <span class="dot" id="dot" style="background:var(--log)"></span>
        <span class="lab" id="level">记录中</span>
        <span style="color:var(--dim);font-size:12px" id="lvlNote"></span>
      </div>
      <div class="riskbar"><i id="riskfill"></i></div>
      <div class="kv">
        <span>风险分</span><b id="riskval">0.00</b>
        <span>驱动信号</span><b id="driver">—</b>
        <span>体动 / 心率 / 血氧</span><b id="vitals">—</b>
      </div>
      <div class="probe" id="probe">门控运行中：风险未达阈值时仅写入事件日志，不惊动家长。</div>
    </div>

    <div class="card">
      <h2><span>事件日志</span><span class="badge" id="evCount"></span></h2>
      <div class="log" id="log"></div>
    </div>

    <div class="card">
      <h2>为什么要"门控"</h2>
      <div class="foot" style="padding:0">
        一次性报警的门槛降低了，代价是夜里被误叫醒。守夜把判定拆成两级：
        风险越过观察线先只推送通知并按触发信号做<b>定向复核</b>（如"心率升高但无体动，
        更像翻身后的心率回落"）；只有连续窗口确认或证据持续抬升，才升级为叫醒家长。
        这条思路来自 AssemblyAI 语音黑客松的 Tell 项目——它的语音确定度门控
        （不确定就拒绝写入并追问）在这里换成了生理信号的风险门控。
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const F = DATA.frames, WIN = DATA.meta.win_s, HOP = DATA.meta.hop_s;
const totalMin = DATA.meta.minutes;
const cS = document.getElementById('cSignals'), cR = document.getElementById('cRisk');
const gS = cS.getContext('2d'), gR = cR.getContext('2d');

function fit(c){ const dpr = window.devicePixelRatio||1;
  const w = c.clientWidth, h = parseInt(c.getAttribute('height'));
  c.width = w*dpr; c.height = h*dpr; c.getContext('2d').setTransform(dpr,0,0,dpr,0,0);
  return {w, h}; }

let dimS, dimR;
function layout(){ dimS = fit(cS); dimR = fit(cR); drawStatic(); draw(); }
window.addEventListener('resize', layout);

const xOfRisk = (i) => (i / (F.length-1)) * dimR.w;

function line(g, dim, values, color, lo, hi, padTop, padBot){
  const h = dim.h - padTop - padBot;
  const xs = (i) => (i/(values.length-1))*dim.w;
  const ys = (v) => padTop + h - ((v-lo)/(hi-lo))*h;
  g.beginPath(); g.strokeStyle = color; g.lineWidth = 1.1;
  for (let i=0;i<values.length;i++){ const x=xs(i), y=ys(values[i]);
    i? g.lineTo(x,y) : g.moveTo(x,y); }
  g.stroke();
  return {xs, ys};
}

function drawStatic(){
  gS.clearRect(0,0,dimS.w,dimS.h);
  const strips = [
    {key:'m', color:'#60a5fa', lo:0, hi:8, label:'体动'},
    {key:'hr', color:'#f87171', lo:55, hi:135, label:'心率'},
    {key:'sp', color:'#34d399', lo:88, hi:100, label:'血氧'},
    {key:'br', color:'#a78bfa', lo:4, hi:32, label:'呼吸'},
  ];
  const pad = 6, gap = 6;
  const sh = (dimS.h - pad*2 - gap*3)/4;
  strips.forEach((s, k) => {
    const top = pad + k*(sh+gap);
    gS.save();
    gS.fillStyle = 'rgba(255,255,255,.02)';
    gS.fillRect(0, top, dimS.w, sh);
    // 真实发作底纹
    F.forEach((f,i)=>{ if(f.truth){ gS.fillStyle='rgba(244,63,94,.22)';
      gS.fillRect(xOfRisk(i), top, dimS.w/F.length+0.6, sh); } });
    gS.beginPath(); gS.strokeStyle = s.color; gS.lineWidth = 1.1;
    const vals = F.map(f=>f[s.key]);
    for (let i=0;i<vals.length;i++){
      const x = (i/(vals.length-1))*dimS.w;
      const y = top + sh - ((vals[i]-s.lo)/(s.hi-s.lo))*sh;
      i? gS.lineTo(x, y) : gS.moveTo(x, y);
    }
    gS.stroke();
    gS.fillStyle = s.color; gS.font = '11px sans-serif';
    gS.fillText(s.label, 6, top + 12);
    gS.restore();
  });

  gR.clearRect(0,0,dimR.w,dimR.h);
  const padT = 10, padB = 16, h = dimR.h - padT - padB;
  // 阈值线
  [[0.55,'#f59e0b','观察线 0.55'],[0.80,'#ef4444','唤醒线 0.80']].forEach(([v,c,t])=>{
    const y = padT + h - v*h;
    gR.strokeStyle = c; gR.setLineDash([4,4]); gR.lineWidth = 1;
    gR.beginPath(); gR.moveTo(0,y); gR.lineTo(dimR.w,y); gR.stroke();
    gR.setLineDash([]); gR.fillStyle = c; gR.font = '10px sans-serif';
    gR.fillText(t, dimR.w-72, y-3);
  });
  // 真实发作底纹
  F.forEach((f,i)=>{ if(f.truth){ gR.fillStyle='rgba(244,63,94,.16)';
    gR.fillRect(xOfRisk(i), padT, dimR.w/F.length+0.6, h); } });
  // 风险面积图
  const grad = gR.createLinearGradient(0,padT,0,padT+h);
  grad.addColorStop(0,'rgba(251,191,36,.55)'); grad.addColorStop(1,'rgba(251,191,36,.03)');
  gR.beginPath(); gR.moveTo(0, padT+h);
  F.forEach((f,i)=>{ const x=xOfRisk(i), y=padT+h-f.risk*h; gR.lineTo(x,y); });
  gR.lineTo(dimR.w, padT+h); gR.closePath(); gR.fillStyle = grad; gR.fill();
  gR.beginPath(); gR.strokeStyle='#fbbf24'; gR.lineWidth=1.2;
  F.forEach((f,i)=>{ const x=xOfRisk(i), y=padT+h-f.risk*h; i?gR.lineTo(x,y):gR.moveTo(x,y); });
  gR.stroke();
  // 报警标记
  F.forEach((f,i)=>{
    if (f.level==='CRITICAL'){ gR.fillStyle='#ef4444'; gR.fillRect(xOfRisk(i), padT, 2, 10); }
    else if (f.level==='WARNING'){ gR.fillStyle='#f59e0b'; gR.fillRect(xOfRisk(i), padT, 1.5, 6); }
  });
  // 时间刻度
  gR.fillStyle='#8ea2c4'; gR.font='10px sans-serif';
  for (let m=0;m<=totalMin;m+=60){ const x=(m*60/(totalMin*60))*dimR.w;
    gR.fillText(`${String(Math.floor(m/60)).padStart(2,'0')}:00`, Math.min(x, dimR.w-30), dimR.h-4); }
}

let idx = 0, playing = true, speed = 30, last = 0;
function draw(){
  const x = xOfRisk(idx);
  ['cSignals','cRisk'].forEach(id=>{});
  gS.save(); gS.strokeStyle='#e8eefb'; gS.globalAlpha=.75; gS.lineWidth=1;
  gS.beginPath(); gS.moveTo(x,0); gS.lineTo(x,dimS.h); gS.stroke(); gS.restore();
  gR.save(); gR.globalAlpha=.8; gR.strokeStyle='#e8eefb'; gR.lineWidth=1;
  gR.beginPath(); gR.moveTo(x,0); gR.lineTo(x,dimR.h); gR.stroke(); gR.restore();
}
let raf = null;
function tick(ts){
  if (!playing) return;
  if (!last) last = ts;
  const dt = (ts-last)/1000; last = ts;
  idx += speed*dt/HOP;
  if (idx >= F.length-1){ idx = F.length-1; playing=false; document.getElementById('play').textContent='播放'; }
  render(); raf = requestAnimationFrame(tick);
}
function fmt(t){ const h=Math.floor(t/3600), m=Math.floor(t%3600/60), s=Math.floor(t%60);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }

const LV = {
  LOG:{c:'#64748b', t:'记录中', n:'不打扰'},
  WARNING:{c:'#f59e0b', t:'观察通知', n:'推送 + 二次确认'},
  CRITICAL:{c:'#ef4444', t:'唤醒家长', n:'已连续确认'},
};
function render(redrawStatic=false){
  const f = F[Math.round(idx)];
  document.getElementById('clock').textContent = fmt(f.t);
  const L = LV[f.level];
  document.getElementById('dot').style.background = L.c;
  document.getElementById('level').textContent = L.t;
  document.getElementById('level').style.color = L.c;
  document.getElementById('lvlNote').textContent = f.confirmed ? L.n + ' · 已确认' : L.n;
  document.getElementById('riskval').textContent = f.risk.toFixed(2);
  document.getElementById('driver').textContent = f.driver || '—';
  document.getElementById('vitals').textContent =
    `${f.m.toFixed(1)} / ${f.hr.toFixed(0)} bpm / ${f.sp.toFixed(1)}%`;
  const fill = document.getElementById('riskfill');
  fill.style.width = (f.risk*100).toFixed(1)+'%';
  fill.style.background = f.level==='CRITICAL' ? '#ef4444' : f.level==='WARNING' ? '#f59e0b' : '#64748b';
  document.getElementById('probe').textContent = f.probe ||
    '门控运行中：风险未达阈值时仅写入事件日志，不惊动家长。';
  document.getElementById('truthBadge').textContent = f.truth ? '真相：发作期' : '真相：正常';
  document.getElementById('truthBadge').style.color = f.truth ? '#f43f5e' : '#8ea2c4';
  draw();
  highlightLog(f.t);
}

// ---- 事件日志 ----
const J = DATA.journal;
const logEl = document.getElementById('log');
const evTimes = [];
J.forEach((e, i) => {
  const d = document.createElement('div');
  d.className = 'ev ' + e.level;
  d.dataset.t = e.start;
  d.innerHTML = `<span class="t">${fmt(e.start)}</span>
    <span class="lv" style="color:${LV[e.level].c}">${e.level==='CRITICAL'?'唤醒':'观察'}</span>
    <span class="t">风险 ${e.risk.toFixed(2)}</span>
    <div class="drv">驱动：${e.driver||'—'}${e.evidence && e.evidence.length ? ' · ' + e.evidence.slice(0,2).join('；') : ''}</div>`;
  d.onclick = () => { idx = e.start/HOP; render(); };
  logEl.appendChild(d);
  evTimes.push(e.start);
});
document.getElementById('evCount').textContent =
  `唤醒 ${DATA.journal_summary.critical} · 观察 ${DATA.journal_summary.warning}`;

function highlightLog(t){
  [...logEl.children].forEach(c=>{
    const dt = Math.abs(parseFloat(c.dataset.t) - t);
    c.style.background = dt < WIN ? '#1e3a63' : '#16233a';
  });
}

// ---- 控件 ----
document.getElementById('play').onclick = (e) => {
  playing = !playing; last = 0;
  e.target.textContent = playing ? '暂停' : '播放';
  if (playing) raf = requestAnimationFrame(tick);
};
document.querySelectorAll('button[data-speed]').forEach(b=>{
  b.onclick = () => { speed = parseInt(b.dataset.speed); };
});
document.getElementById('prev').onclick = () => {
  const t = F[Math.round(idx)].t;
  const prev = evTimes.filter(x=>x < t-5).pop();
  if (prev !== undefined){ idx = prev/HOP; render(); }
};
document.getElementById('next').onclick = () => {
  const t = F[Math.round(idx)].t;
  const nx = evTimes.find(x=>x > t+5);
  if (nx !== undefined){ idx = nx/HOP; render(); }
};
document.getElementById('reset').onclick = () => { idx = 0; render(); };

layout();
render();
raf = requestAnimationFrame(tick);
</script>
</body>
</html>
"""

out = HTML.replace("__DATA__", data_json)
p = os.path.join(HERE, "dashboard.html")
with open(p, "w", encoding="utf-8") as f:
    f.write(out)
print(f"→ {p}  ({os.path.getsize(p)/1024:.0f} KB)")
