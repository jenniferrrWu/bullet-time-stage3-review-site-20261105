(function(){
var data,currentGroup=null,currentVariant=null,frame=0,playing=false;
var id=function(x){return document.getElementById(x)};
var esc=function(v){return String(v==null?'':v).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]})};
var vids=function(){return [id('cmpSource'),id('cmpBaseline'),id('cmpVariant')]};
function pause(){playing=false;vids().forEach(function(v){v.pause()});id('syncPlay').textContent='同步播放'}
function seek(n){
 if(!currentGroup||!currentVariant)return;
 var max=Math.min(currentGroup.source.timing.frames,currentGroup.baseline.timing.frames,currentVariant.timing.frames)-1;
 frame=Math.max(0,Math.min(max,Math.round(n)));
 id('frameSlider').max=max;id('frameSlider').value=frame;id('frameCounter').textContent=frame+' / '+max;
 var timing=[currentGroup.source.timing,currentGroup.baseline.timing,currentVariant.timing];
 vids().forEach(function(v,i){try{v.currentTime=frame/timing[i].fps+0.001}catch(e){}});
}
function renderPipeline(caseId){
 var p=data.pipelines.find(function(x){return x.id===caseId});if(!p)return;
 id('pipelineTabs').innerHTML=data.pipelines.map(function(x){return '<button type="button" data-id="'+esc(x.id)+'" class="'+(x.id===caseId?'active':'')+'">'+esc(x.title)+'</button>'}).join('');
 id('pipelineTabs').querySelectorAll('button').forEach(function(b){b.onclick=function(){renderPipeline(b.dataset.id)}});
 id('pipelineIntro').innerHTML='<b>'+esc(p.description)+'</b><br>'+esc(p.verdict);
 id('pipelineCards').innerHTML=p.candidates.map(function(c){return '<article class="pipeline-card"><div class="card-head"><div class="candidate-name">'+esc(c.label)+'</div><div class="candidate-verdict">'+esc(c.verdict)+'</div></div><video controls playsinline preload="none" src="'+esc(c.complete)+'"></video><div class="stage-buttons"><button class="active" type="button" data-url="'+esc(c.complete)+'">完整三阶段</button><button type="button" data-url="'+esc(c.stage1)+'">Stage1</button><button type="button" data-url="'+esc(c.stage2)+'">Stage2</button><button type="button" data-url="'+esc(c.stage3)+'">Stage3 结果</button></div><div class="card-foot"><a href="'+esc(c.source)+'" target="_blank">源动作 ↗</a> · <a href="'+esc(c.record)+'" target="_blank">运行记录 ↗</a></div></article>'}).join('');
 id('pipelineCards').querySelectorAll('.pipeline-card').forEach(function(card){
  var v=card.querySelector('video');
  card.querySelectorAll('.stage-buttons button').forEach(function(b){b.onclick=function(){
   v.pause();v.src=b.dataset.url;
   card.querySelectorAll('.stage-buttons button').forEach(function(x){x.classList.remove('active')});b.classList.add('active');
  }});
 });
}
function renderVariant(variantId){
 pause();
 currentVariant=currentGroup.variants.find(function(v){return v.id===variantId})||currentGroup.variants[0];
 id('variantButtons').innerHTML=currentGroup.variants.map(function(v){return '<button type="button" data-id="'+esc(v.id)+'" class="'+(v.id===currentVariant.id?'active':'')+'">'+esc(v.label)+'</button>'}).join('');
 id('variantButtons').querySelectorAll('button').forEach(function(b){b.onclick=function(){renderVariant(b.dataset.id)}});
 var ev=currentVariant.evidence?'<a target="_blank" href="'+esc(currentVariant.evidence)+'">查看结果记录 ↗</a>':'';
 var hd=currentVariant.hd_url?'<a target="_blank" href="'+esc(currentVariant.hd_url)+'">改进版高清审查转码 ↗</a>':'';
 var sourcehd=currentGroup.source_hd_url?'<a target="_blank" href="'+esc(currentGroup.source_hd_url)+'">原片高清审查转码 ↗</a>':'';
 var basehd=currentGroup.baseline_hd_url?'<a target="_blank" href="'+esc(currentGroup.baseline_hd_url)+'">第一版高清审查转码 ↗</a>':'';
 id('comparisonVerdict').innerHTML='<b>'+esc(currentVariant.label)+'：</b> '+esc(currentVariant.verdict)+ev+' '+sourcehd+' '+basehd+' '+hd;
 var v=id('cmpVariant');v.src=currentVariant.url;
 frame=0;seek(0);
}
function renderGroup(groupId){
 pause();
 currentGroup=data.comparison_groups.find(function(x){return x.id===groupId})||data.comparison_groups[0];
 id('compareGroup').value=currentGroup.id;
 id('compareIntro').innerHTML='<b>'+esc(currentGroup.title)+'</b><br>'+esc(currentGroup.description);
 id('frameNote').textContent=currentGroup.frame_note+' 源动作 '+currentGroup.source.timing.fps+' fps；第一版 '+currentGroup.baseline.timing.fps+' fps。播放按帧率换算同步。';
 id('cmpSource').src=currentGroup.source.url;id('cmpBaseline').src=currentGroup.baseline.url;
 
 renderVariant(currentGroup.variants[0].id);
}
function play(){
 if(!currentGroup||!currentVariant)return;
 pause();var timing=[currentGroup.source.timing,currentGroup.baseline.timing,currentVariant.timing];
 var base=currentVariant.timing.fps;
 vids().forEach(function(v,i){v.playbackRate=timing[i].fps/base});
 playing=true;id('syncPlay').textContent='播放中…';
 vids().forEach(function(v){v.play().catch(function(){pause()})});
}
fetch('./review_data-20261106-flashcombo.json').then(function(r){if(!r.ok)throw Error('review data HTTP '+r.status);return r.json()}).then(function(d){
 data=d;
 id('montageChapters').querySelectorAll('button').forEach(function(b){b.onclick=function(){var v=id('montageVideo');v.currentTime=Number(b.dataset.time);v.play()}});
 renderPipeline('case15');
 id('compareGroup').innerHTML=data.comparison_groups.map(function(g){return '<option value="'+esc(g.id)+'">'+esc(g.title)+'</option>'}).join('');
 id('compareGroup').onchange=function(){renderGroup(this.value)};
 id('frameSlider').oninput=function(){pause();seek(this.value)};
 id('frameBack').onclick=function(){pause();seek(frame-1)};
 id('frameNext').onclick=function(){pause();seek(frame+1)};
 id('syncPlay').onclick=play;id('syncPause').onclick=pause;
 id('cmpVariant').ontimeupdate=function(){
  if(!playing)return;
  var max=Math.min(currentGroup.source.timing.frames,currentGroup.baseline.timing.frames,currentVariant.timing.frames)-1;
  frame=Math.min(max,Math.round(this.currentTime*currentVariant.timing.fps));
  id('frameSlider').value=frame;id('frameCounter').textContent=frame+' / '+max;
  if(frame>=max)pause();
 };
 renderGroup('case15-upstream');
}).catch(function(e){id('pipelineIntro').textContent='视频审查数据加载失败：'+e.message});
})();
