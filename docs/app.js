const S = sel => document.querySelector(sel);
let Xrows = [], genes = [], labels = {};

async function loadData(){
  const [xj, gj, lj, mj] = await Promise.all([
    fetch("data/Xv.json").then(r=>r.json()),
    fetch("data/genes.json").then(r=>r.json()),
    fetch("data/labels.json").then(r=>r.json()),
    fetch("data/manifest.json").then(r=>r.json()).catch(()=>({}))
  ]);
  Xrows = xj; genes = gj; labels = lj;
  // populate gene select
  const gsel = S("#genes");
  gsel.innerHTML = "";
  genes.forEach(g=>{
    const o = document.createElement("option");
    o.value=g; o.textContent=g; gsel.appendChild(o);
  });
  S("#results").textContent = JSON.stringify(mj, null, 2);
  S("#status").textContent = `Loaded ${genes.length} genes • ${Xrows.length} samples`;
}

function getSelectedGenes(){
  return Array.from(S("#genes").selectedOptions).map(o=>o.value).slice(0,50);
}

function pivot(selected){
  // Xrows is [{gsm, G1, G2, ...}]
  const gsm = Xrows.map(r=>r.gsm);
  const Z = selected.map(g => Xrows.map(r => Number(r[g])));
  return {gsm, Z};
}

function showHeatmap(){
  const sel = getSelectedGenes();
  if(sel.length===0){ alert("Select 1–50 genes"); return; }
  const {gsm, Z} = pivot(sel);
  const lab = gsm.map(id => labels[id] || "NA");
  const data=[{z:Z, x:gsm, y:sel, type:"heatmap", hoverongaps:false, zmin:-3, zmax:3}];
  const layout={margin:{l:90,r:10,t:10,b:80}, xaxis:{tickangle:45}, yaxis:{automargin:true}};
  Plotly.newPlot('heat', data, layout, {responsive:true});
  S("#table").style.display = "none";
}

let dt=null;
function showTable(){
  const sel = getSelectedGenes();
  const cols = ["gsm", ...sel];
  const data = Xrows.map(r => cols.map(c => r[c]));
  const columns = cols.map(c => ({title:c}));
  S("#table").style.display = "table";
  if(dt){dt.destroy();}
  dt = new DataTable('#table', {data, columns, pageLength:25, scrollX:true});
}

S("#btnHeat").onclick = showHeatmap;
S("#btnTable").onclick = showTable;
loadData();
