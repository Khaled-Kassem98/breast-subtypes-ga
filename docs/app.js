async function loadJSON(p){ const r = await fetch(p,{cache:"no-store"}); return r.json(); }

const els = {
  model: document.getElementById('model'),
  pop:   document.getElementById('pop'),
  gens:  document.getElementById('gens'),
  cxp:   document.getElementById('cxp'),
  mutp:  document.getElementById('mutp'),
  heat:  document.getElementById('heat'),
  table: document.getElementById('table'),
  res:   document.getElementById('results'),
  mval:  document.getElementById('mval'),
  pval:  document.getElementById('pval'),
  gval:  document.getElementById('gval'),
  cval:  document.getElementById('cval'),
  uval:  document.getElementById('uval'),
};

function syncHints(){
  els.mval.textContent = els.model.value;
  els.pval.textContent = els.pop.value;
  els.gval.textContent = els.gens.value;
  els.cval.textContent = els.cxp.value;
  els.uval.textContent = els.mutp.value;
}
['change','input'].forEach(ev=>{
  [els.model,els.pop,els.gens,els.cxp,els.mutp].forEach(e=>e.addEventListener(ev,syncHints));
});
syncHints();

let M=null, labels=null;

async function init(){
  M = await loadJSON('data/matrix.json');      // {genes, gsms, X}
  labels = await loadJSON('data/labels.json');  // {labels}
  drawHeat();
  await refreshResults();
}
init();

function drawHeat(){
  const ctx = els.heat.getContext('2d');
  const g = M.genes.length, s = M.gsms.length;
  const W = els.heat.width = Math.max(900, s*10);
  const H = els.heat.height = Math.max(260, g*10);
  const cellW = W/s, cellH = H/g;

  const flat = M.X.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const scale = v => (v - lo)/(hi - lo + 1e-9);

  ctx.clearRect(0,0,W,H);
  for(let i=0;i<g;i++){
    for(let j=0;j<s;j++){
      const v = scale(M.X[i][j]);
      // blue-white-red
      const c = Math.round(v*255);
      ctx.fillStyle = `rgb(${c},${Math.round(255*(1-Math.abs(v-0.5)*2))},${255-c})`;
      ctx.fillRect(j*cellW, i*cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function drawTable(){
  const g = M.genes.length, s = M.gsms.length;
  const header = `<tr><th>Gene \\ GSM</th>${M.gsms.map(x=>`<th>${x}</th>`).join('')}</tr>`;
  const rows = [];
  for(let i=0;i<g;i++){
    const cells = M.X[i].map(v=>`<td>${v.toFixed(2)}</td>`).join('');
    rows.push(`<tr><th>${M.genes[i]}</th>${cells}</tr>`);
  }
  els.table.innerHTML = `<div class="scroll"><table>${header}${rows.join('')}</table></div>`;
}

async function refreshResults(){
  try {
    const R = await loadJSON('data/results.json');
    els.res.textContent = JSON.stringify(R, null, 2);
  } catch(e){
    els.res.textContent = '{}';
  }
}

document.getElementById('btn-heat').onclick = ()=>{
  els.heat.classList.remove('hidden'); els.table.classList.add('hidden');
  drawHeat();
};
document.getElementById('btn-table').onclick = ()=>{
  els.table.classList.remove('hidden'); els.heat.classList.add('hidden');
  drawTable();
};
document.getElementById('btn-refresh').onclick = refreshResults;
