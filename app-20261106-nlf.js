(function(){
var data, visibleAssets=60;
var byId=function(id){return document.getElementById(id)};
var esc=function(v){return String(v==null?'':v).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})};
var fileUrl=function(path){return './evidence/'+path.split('/').map(encodeURIComponent).join('/')};
var size=function(n){return n>1048576?(n/1048576).toFixed(1)+' MB':(n/1024).toFixed(0)+' KB'};
function setRun(){
  var name=byId('runFilter').value,run=data.runs.find(function(x){return x.id===name});
  if(!run)return;
  byId('sourceVideo').src=run.source_url||'';
  byId('nativeVideo').src=run.native_url;
  var h='<b>'+esc(run.id)+'</b> · '+esc(run.case)+'<br>原始 Stage3 文件 SHA256: '+esc(run.native_sha256)+'<br>执行: '+esc(run.execution_status)+'；接受状态: '+esc(run.acceptance_status)+'<br>';
  h+='<a href="'+esc(run.complete_url)+'" target="_blank">打开完整成片审查转码 ↗</a> · <a href="'+esc(run.record_url)+'" target="_blank">原生运行记录 ↗</a>';
  if(run.source_review_transcode)h+=' · 源动作与 complete 为审查转码；上方 SHA 适用于Stage3 网页转码，complete 原文件 SHA 对应原文件';
  byId('runMeta').innerHTML=h;
}
function filterRuns(){
  var c=byId('caseFilter').value, arr=data.runs.filter(function(r){return c==='all'||r.case===c});
  byId('runFilter').innerHTML=arr.map(function(r){return '<option value="'+esc(r.id)+'">'+esc(r.id)+'</option>'}).join('');
  setRun();
}
function renderVariants(){
  var c=byId('caseFilter').value,arr=data.variants.filter(function(v){return c==='all'||v.case===c});
  byId('variantList').innerHTML=arr.map(function(v){return '<button type="button" data-id="'+esc(v.id)+'">'+esc(v.id)+'</button>'}).join('');
  byId('variantList').querySelectorAll('button').forEach(function(b){b.onclick=function(){
    var v=data.variants.find(function(x){return x.id===b.dataset.id});if(!v)return;
    byId('variantList').querySelectorAll('button').forEach(function(t){t.classList.remove('active')});b.classList.add('active');
    byId('variantTitle').textContent=v.id+' · Stage3 网页转码 · 原始文件 SHA256 '+v.native_sha256;
    byId('variantVideo').src=v.native_url;byId('variantViewer').hidden=false;byId('variantViewer').scrollIntoView({block:'nearest',behavior:'smooth'});
  }});
}
function renderExperiments(){
  var q=byId('expSearch').value.toLowerCase().trim(),g=byId('groupFilter').value;
  var arr=data.experiments.filter(function(e){return (g==='all'||e.group===g)&&(!q||JSON.stringify(e).toLowerCase().includes(q))});
  byId('experimentList').innerHTML=arr.map(function(e){
    var links=e.evidence.map(function(path){return '<a target="_blank" href="'+fileUrl(path)+'">'+esc(path.split('/').slice(-2).join('/'))+' ↗</a>'}).join('');
    return '<details class="experiment"><summary><span class="tag">'+esc(e.group)+'<br>'+esc(e.case)+'</span><span class="exp-title">'+esc(e.title)+'</span><span class="status">'+esc(e.status)+'</span></summary><div class="exp-body"><b>实际做法：</b>'+esc(e.method)+'<br><b>整段结果：</b>'+esc(e.result)+'<div class="links">'+links+'</div></div></details>';
  }).join('')||'<p>没有符合条件的实验。</p>';
}
function renderPapers(){
  byId('paperList').innerHTML=data.literature.map(function(p){return '<article class="paper"><div class="paper-top"><a href="'+esc(p.url)+'" target="_blank" rel="noopener noreferrer">'+esc(p.name)+' ↗</a><span>'+esc(p.status)+'</span></div><p><b>论文原方法：</b>'+esc(p.original)+'</p><p><b>本地实际实现：</b>'+esc(p.transfer)+'</p></article>'}).join('');
}
function renderAssets(){
  var q=byId('assetSearch').value.toLowerCase().trim(),t=byId('assetType').value;
  var arr=data.assets.filter(function(a){var type=a.type==='jpeg'?'jpg':a.type;return (t==='all'||type===t||t==='md'&&['md','txt','csv'].includes(type))&&(!q||a.name.toLowerCase().includes(q))});
  byId('assetCount').textContent='匹配 '+arr.length+' / 收录 '+data.assets.length+' 个文件；当前显示 '+Math.min(visibleAssets,arr.length);
  byId('assetList').innerHTML=arr.slice(0,visibleAssets).map(function(a){return '<div class="asset"><a href="'+esc(a.url)+'" target="_blank">'+esc(a.name)+'</a><span>'+esc(a.type.toUpperCase())+'</span><span>'+size(a.bytes)+'</span></div>'}).join('')||'<div class="asset">没有匹配文件。</div>';
  byId('moreAssets').hidden=visibleAssets>=arr.length;
}
fetch('./catalog-20261106-nlf.json').then(function(r){if(!r.ok)throw Error('catalog HTTP '+r.status);return r.json()}).then(function(c){
  data=c;
  var groups=[...new Set(data.experiments.map(function(x){return x.group}))];
  byId('groupFilter').innerHTML='<option value="all">全部类别</option>'+groups.map(function(g){return '<option value="'+esc(g)+'">'+esc(g)+'</option>'}).join('');
  byId('caseFilter').onchange=function(){filterRuns();renderVariants()};
  byId('runFilter').onchange=setRun;
  byId('groupFilter').onchange=renderExperiments;byId('expSearch').oninput=renderExperiments;
  byId('assetType').onchange=function(){visibleAssets=60;renderAssets()};byId('assetSearch').oninput=function(){visibleAssets=60;renderAssets()};
  byId('moreAssets').onclick=function(){visibleAssets+=100;renderAssets()};
  byId('limits').textContent=c.limits;
  filterRuns();renderVariants();renderExperiments();renderPapers();renderAssets();
}).catch(function(e){byId('experimentList').textContent='数据加载失败：'+e.message});
})();
