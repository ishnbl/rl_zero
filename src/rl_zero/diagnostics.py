"""Read local scalar traces without requiring W&B or a running web service."""

import html
import json
from pathlib import Path

from rl_zero.io import read_jsonl


def dashboard(run_dir, output):
    run_dir = Path(run_dir)
    traces = []
    for name in ("metrics.jsonl", "gradients.jsonl"):
        path = run_dir / name
        if path.exists():
            traces.extend(read_jsonl(path))
    if not traces:
        raise ValueError("No metrics yet; the trainer writes metrics.jsonl after its first logged step")
    traces.sort(key=lambda r: r["time"])
    payload = json.dumps(traces).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>RL Zero diagnostics</title>
<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 2rem;color:#172b3b}
select{font:inherit;padding:.5rem;width:100%}svg{width:100%;height:360px;background:#f4f7fa;border-radius:12px}
table{border-collapse:collapse;width:100%}td,th{padding:.5rem;text-align:left;border-bottom:1px solid #dce4ec}
.muted{color:#607080}code{background:#eef3f7;padding:.2rem}</style>
<h1>RUN_NAME</h1><p class="muted">Local training observations · select a metric to inspect its trajectory.</p>
<select id="metric"></select><p id="summary"></p><svg id="plot" viewBox="0 0 1000 360"></svg>
<p class="muted">Horizontal axis: optimizer step. Vertical axis: selected metric. Nonfinite observations are
excluded from the line and counted below. RLVR rewards may be scaled by verification_reward.</p>
<table><thead><tr><th>Metric</th><th>Latest</th><th>Min</th><th>Max</th></tr></thead><tbody id="rows"></tbody></table>
<script>const records=PAYLOAD;const keys=[...new Set(records.flatMap(r=>Object.keys(r.metrics)))].sort();
const select=document.querySelector('#metric');
function values(key){return records.filter(r=>key in r.metrics).map((r,i)=>[r.step??i,r.metrics[key]])}
for(const key of keys){const o=document.createElement('option');o.textContent=key;select.append(o);
 const all=values(key),ys=all.map(v=>v[1]).filter(v=>typeof v==='number'&&Number.isFinite(v));
 const tr=document.createElement('tr');for(const v of [key,all.at(-1)[1],ys.length?Math.min(...ys):'—',
 ys.length?Math.max(...ys):'—']){const td=document.createElement('td');td.textContent=v;tr.append(td)}
 document.querySelector('#rows').append(tr)}
function draw(){const all=values(select.value),v=all.filter(p=>typeof p[1]==='number'&&Number.isFinite(p[1]));
 document.querySelector('#summary').textContent=`${v.length} finite observations; ${all.length-v.length} nonfinite.`;
 const svg=document.querySelector('#plot');svg.replaceChildren();if(!v.length)return;
 const xs=v.map(p=>p[0]),ys=v.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),
 ymin=Math.min(...ys),ymax=Math.max(...ys);
 const coords=v.map(p=>[60+880*(p[0]-xmin)/(xmax-xmin||1),310-260*(p[1]-ymin)/(ymax-ymin||1)]);
 function node(tag,attrs){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);
 for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);svg.append(n);return n}
 node('polyline',{points:coords.map(p=>p.join(',')).join(' '),fill:'none',stroke:'#007b83','stroke-width':3});
 for(const [x,y]of coords)node('circle',{cx:x,cy:y,r:3,fill:'#007b83'});
 for(const [x,y,t]of [[60,340,xmin],[900,340,xmax],[8,50,ymax.toPrecision(4)],[8,310,ymin.toPrecision(4)]])
 node('text',{x,y,'font-size':13}).textContent=t;}
select.value=keys.find(k=>k.includes('grad_norm'))??keys[0];select.onchange=draw;draw();</script>"""
    page = page.replace("RUN_NAME", html.escape(run_dir.name)).replace("PAYLOAD", payload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    return str(output)
