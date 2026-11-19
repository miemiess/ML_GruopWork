"use strict";
const el = id => document.getElementById(id);
const experiment = window.EMG_DEVELOPMENT;
const gestures = ["静息","握拳","腕屈","腕伸","桡偏","尺偏"];
const modelNames = {logistic:"逻辑回归",knn:"kNN",forest:"随机森林"};
const colors = ["#92d7be","#bef792","#eac885","#c4b5fa","#79bdfa","#ed9fa7"];
const mint = "#bef792", muted = "#a4b0c1", bright = "#eaf0f7";
const percent = value => (100 * value).toFixed(1) + "%";
const gesture = label => gestures[label-1];

if (experiment) startDevelopment(experiment);
else el("devMissing").hidden = false;

function startDevelopment(data) {
  const row = data.sample;
  const votes = gestures.map((_,i)=>data.neighbors.points.filter(p=>p.label===i+1).length);
  const example = "验证用户 "+row.subject+" · "+gesture(row.label);
  const result = (id,pred)=>[
    [data.best[id].parameter+" · "+data.best[id].features, "该模型在验证集上的最佳配置"],
    [data.best[id].macro_f1.toFixed(4), "验证 macro-F1"],
    [gesture(pred), "同一验证窗口的预测"]
  ];
  const stages = [
    {tab:"准备数据", title:"先按受试者分组", description:"同一受试者只进入一个集合。提取每个窗口的时域特征后，训练集拟合模型，验证集比较方案，测试集评估获选模型。",
     caption:"三组用户互斥。后面的三个模型都观察同一个验证窗口。",
     facts:[["200 点 / 100 点","窗长 / 步长"],["8 路 → 16 或 24 个特征","RMS、MAV，可选 WL"],[example,"当前对比窗口的真实标签"]], draw:drawSplit},
    {tab:"提取特征", title:"把一段波形变成特征", description:"从同一个验证窗口选一条通道。RMS 是波形的有效幅度，更强调尖峰；MAV 是幅值绝对值的平均。可选的 WL 汇总所有相邻采样点的绝对差。",
     caption:"曲线是所选通道真实的 200 个采样点。虚线标出 RMS 与 MAV 对应的幅度。开启 WL 后，图中高亮一对变化较大的相邻点。",
     facts:[["RMS","平方 → 求均值 → 开方"],["MAV","取绝对值 → 求均值"],["WL（可选）","相邻点作差 → 取绝对值 → 求和"]], draw:drawExtraction},
    {tab:"逻辑回归", title:"一个特征能把概率推多远？", description:"模型给六个动作分别算分，再用 Softmax 换算概率。拖动滑块，改变一个标准化特征，其余特征固定在同一个验证窗口的实际值。",
     caption:"概率曲线来自训练后的六分类模型。竖线是滑块位置，实心点标出原窗口该特征的值。",
     facts:result("logistic",data.curve.prediction), draw:drawLogistic},
    {tab:"kNN", title:"最近的邻居怎么投票？", description:"每个训练窗口是一条带动作标签的特征向量。按标准化特征空间的欧氏距离排序，从前 k 个训练窗口中取票数最多的动作。",
     caption:"横轴是到同一验证窗口的真实距离，圆点按动作分行，右侧是每一类的票数。",
     facts:result("knn",data.neighbors.prediction), draw:drawKnn},
    {tab:"随机森林", title:"从一棵树走到一片森林", description:"树按特征阈值依次判断。当前窗口在每个节点选择左边或右边，走到叶子得到一组类别概率。森林平均 100 棵树的概率，再选出最高类。",
     caption:"图展示第 1 棵树的前两次分裂，亮线是这个窗口的实际路径。概率来自完整路径。",
     facts:result("forest",data.forest.prediction), draw:drawForest},
    {tab:"模型选型", title:"把验证集成绩放在一起", description:"每种模型在同一批验证用户上比较参数及特征组合。验证 macro-F1 最高的方案被选中，再用留出的测试用户评估。",
     caption:"柱长从 0 开始，显示各模型最佳验证 macro-F1。展开表格可检查完整的 12 组配置。",
     facts:[[data.best.forest.features+" · 100 棵树","最终配置"],[(100*data.test.accuracy).toFixed(2)+"%","留出测试准确率"],[data.test.macro_f1.toFixed(4),"留出测试 macro-F1"]], draw:drawScores}
  ];
  const canvas=el("devCanvas"), context=canvas.getContext("2d"), slider=el("devFeature");
  const channel=el("devChannel"), includeWL=el("devIncludeWL");
  data.sample.waveform?.forEach((_,i)=>channel.add(new Option("CH "+(i+1),i)));
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  let step=0, animation=0, progress=1;
  function render(amount){
    progress=amount;
    const dpr=Math.min(devicePixelRatio||1,2);
    const width=Math.round(canvas.clientWidth*dpr),height=Math.round(canvas.clientHeight*dpr);
    if(width<1 || height<1) return;
    if(canvas.width!==width)canvas.width=width;
    if(canvas.height!==height)canvas.height=height;
    context.setTransform(width/760,0,0,height/335,0,0);
    context.clearRect(0,0,760,335);
    stages[step].draw(context,data,amount,Number(slider.value),votes,
      {channel:Number(channel.value), wl:includeWL.checked});
  }
  const buttons=stages.map((stage,i)=>{
    const b=document.createElement("button");
    b.type="button";b.innerHTML="<span>"+String(i+1).padStart(2,"0")+"</span><span>"+stage.tab+"</span>";
    b.onclick=()=>show(i);el("devSteps").append(b);return b;
  });
  function show(i){
    step=Math.max(0,Math.min(stages.length-1,i)); const stage=stages[step];
    cancelAnimationFrame(animation);
    buttons.forEach((button,n)=>button.setAttribute("aria-current",n===step?"step":"false"));
    el("devStageLabel").textContent=stage.tab;
    el("devTitle").textContent=stage.title;
    el("devDescription").textContent=stage.description;
    el("devCaption").textContent=stage.caption;
    canvas.setAttribute("aria-label",stage.title+" · "+stage.caption);
    el("devFacts").innerHTML=stage.facts.map(([v,k])=>'<div class="devFact"><strong>'+v+"</strong><small>"+k+"</small></div>").join("");
    el("devSweep").hidden=stage.draw!==drawLogistic;
    el("devExtraction").hidden=stage.draw!==drawExtraction;
    el("devPrev").disabled=step===0;
    el("devNext").hidden=step===stages.length-1;
    el("devCandidates").hidden=step!==stages.length-1;
    el("devFinish").hidden=step!==stages.length-1;
    el("devPosition").textContent=(step+1)+" / "+stages.length;
    if(stage.draw===drawLogistic)slider.value=Math.floor(data.curve.x.length/2);
    if(stage.draw===drawExtraction)updateCount();
    const start=performance.now();
    function tick(now){
      const t=reduced?1:Math.min(1,Math.max(0,(now-start)/720));
      render(1-(1-t)**2);
      if(t<1)animation=requestAnimationFrame(tick);
    }
    tick(start);
  }
  el("devPrev").onclick=()=>show(step-1);
  el("devNext").onclick=()=>show(step+1);
  slider.oninput=()=>{cancelAnimationFrame(animation);render(1)};
  function updateCount() {
    const formula=includeWL.checked ? "8 通道 × 3 特征 → 24 维" : "8 通道 × 2 特征 → 16 维";
    el("devFeatureCount").textContent=formula;
  }
  channel.onchange=()=>{cancelAnimationFrame(animation);render(1)};
  includeWL.onchange=()=>{updateCount();cancelAnimationFrame(animation);render(1)};
  document.addEventListener("keydown",event=>{
    if(event.target.closest("input, textarea, select") || !["ArrowLeft","ArrowRight"].includes(event.key))return;
    event.preventDefault();
    show(step+(event.key==="ArrowRight"?1:-1));
  });
  el("devTable").innerHTML="<table><thead><tr><th>模型</th><th>特征</th><th>参数</th><th>验证 macro-F1</th><th>验证准确率</th></tr></thead><tbody>"+
    [...data.scores].sort((a,b)=>b.macro_f1-a.macro_f1).map((entry,i)=>'<tr class="'+(i===0?"bestConfig":"")+'"><td>'+modelNames[entry.model]+"</td><td>"+entry.features+"</td><td>"+entry.parameter+"</td><td>"+entry.macro_f1.toFixed(4)+"</td><td>"+percent(entry.accuracy)+"</td></tr>").join("")+"</tbody></table>";
  el("devPlayer").hidden=false;
  show(0);
  new ResizeObserver(()=>render(progress)).observe(canvas);
}

function text(ctx,x,y,value,fill=bright,size=17,align="left"){
  ctx.font=size+'px "Segoe UI","Microsoft YaHei",sans-serif';ctx.fillStyle=fill;
  ctx.textAlign=align;ctx.textBaseline="middle";ctx.fillText(value,x,y);
}
function line(ctx,points,stroke,width=1){
  ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.beginPath();
  points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();
}
function box(ctx,x,y,w,h,fill){ctx.fillStyle=fill;ctx.fillRect(x,y,w,h)}
function dot(ctx,x,y,r,fill){ctx.fillStyle=fill;ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fill()}
function drawSplit(ctx,data,p){
  const users=Object.values(data.users).reduce((sum,n)=>sum+n,0);
  text(ctx,30,35,"从 "+users+" 名受试者到三组窗口",bright,20);
  const groups=[["train","训练集","拟合模型"],["validation","验证集","比较候选"],["test","测试集","评估获选模型"]];
  const max=Math.max(...Object.values(data.windows));
  groups.forEach(([id,name,role],i)=>{
    const y=77+i*79;
    text(ctx,30,y+8,name,bright,18);text(ctx,30,y+34,role,muted,14);
    box(ctx,145,y,475,22,"#273343");
    box(ctx,145,y,475*data.windows[id]/max*p,22,[mint,"#a990e6","#74b8f0"][i]);
    text(ctx,145,y+40,data.windows[id]+" 窗口",muted,15);
    text(ctx,718,y+10,data.users[id]+" 人",bright,20,"right");
  });
}
function drawExtraction(ctx,data,p,_point,_votes,{channel,wl}){
  if (!data.sample.waveform) {
    text(ctx,35,80,"请重新生成演示数据，以加载窗口波形。",muted,19);
    return;
  }
  const signal=data.sample.waveform[channel];
  const [rms,mav,length]=data.sample.features[channel];
  const left=58,right=730,center=134,half=95;
  const limit=Math.max(...signal.map(Math.abs),1e-8);
  const x=i=>left+(right-left)*i/(signal.length-1);
  const y=v=>center-half*v/limit;
  const count=Math.max(2,Math.ceil(signal.length*p));

  text(ctx,left,19,"CH "+(channel+1)+" · 原始肌电",bright,17);
  line(ctx,[[left,center],[right,center]],"#425065");
  ctx.setLineDash([5,5]);
  line(ctx,[[left,y(rms)],[right,y(rms)]],"#a990e6");
  line(ctx,[[left,y(-rms)],[right,y(-rms)]],"#a990e6");
  line(ctx,[[left,y(mav)],[right,y(mav)]],"#d1b679");
  line(ctx,[[left,y(-mav)],[right,y(-mav)]],"#d1b679");
  ctx.setLineDash([]);
  line(ctx,signal.slice(0,count).map((v,i)=>[x(i),y(v)]),mint,2);
  text(ctx,left,249,"0",muted,14);
  text(ctx,right,249,"199 采样点",muted,14,"right");

  if(wl){
    let edge=1;
    for(let i=2;i<count;i++){
      if(Math.abs(signal[i]-signal[i-1])>Math.abs(signal[edge]-signal[edge-1]))edge=i;
    }
    line(ctx,[[x(edge-1),y(signal[edge-1])],[x(edge),y(signal[edge])]],"#79bdfa",4);
    dot(ctx,x(edge-1),y(signal[edge-1]),4,"#79bdfa");
    dot(ctx,x(edge),y(signal[edge]),4,"#79bdfa");
    text(ctx,left,271,"WL 累加 199 个相邻差｜高亮一项 Δ = "+Math.abs(signal[edge]-signal[edge-1]).toPrecision(3),muted,14);
  }

  const metrics=[["RMS",rms,"#bda8f0"],["MAV",mav,"#e3c789"]];
  if(wl)metrics.push(["WL",length,"#79bdfa"]);
  metrics.forEach(([label,value,color],i)=>{
    const width=(right-left)/metrics.length;
    const start=left+width*i;
    box(ctx,start,287,width-7,36,"#253144");
    text(ctx,start+10,305,label+"   "+value.toPrecision(4),color,17);
  });
}
function drawLogistic(ctx,data,p,point){
  const c=data.curve,left=78,right=725,top=31,bottom=222;
  const x=i=>left+(right-left)*i/(c.x.length-1),y=v=>bottom-(bottom-top)*v;
  for(const value of [0,.25,.5,.75,1]){
    line(ctx,[[left,y(value)],[right,y(value)]],"#2d394b");
    text(ctx,left-10,y(value),Math.round(value*100)+"%",muted,14,"right");
  }
  gestures.forEach((_,i)=>line(ctx,c.p.slice(0,Math.max(2,Math.ceil(c.x.length*p))).map((v,j)=>[x(j),y(v[i])]),colors[i],i+1===c.prediction?3.2:1.6));
  const middle=Math.floor(c.x.length/2);
  dot(ctx,x(middle),y(c.p[middle][c.prediction-1]),6,colors[c.prediction-1]);
  const winner=c.p[point].indexOf(Math.max(...c.p[point]));
  line(ctx,[[x(point),top],[x(point),bottom]],"#a4b0c1");
  dot(ctx,x(point),y(c.p[point][winner]),4,colors[winner]);
  text(ctx,left,bottom+21,c.x[0].toFixed(2),muted,14);
  text(ctx,right,bottom+21,c.x.at(-1).toFixed(2),muted,14,"right");
  text(ctx,(left+right)/2,bottom+23,c.feature+" · 标准化值 "+c.x[point].toFixed(2)+" → "+gestures[winner]+" "+percent(c.p[point][winner]),bright,16,"center");
  gestures.forEach((label,i)=>{const px=95+i*105;dot(ctx,px,290,4,colors[i]);text(ctx,px+10,290,label,muted,15)});
}
function drawKnn(ctx,data,p,_point,votes){
  const points=data.neighbors.points,max=points.at(-1).distance,left=106,right=630;
  const xp=distance=>left+(right-left)*distance/max;
  text(ctx,left,30,"离当前窗更近",muted,15);text(ctx,right,30,"更远",muted,15,"right");text(ctx,700,30,"票数",muted,15,"center");
  gestures.forEach((label,i)=>{
    const y=62+i*38;
    line(ctx,[[left,y],[right,y]],"#344252");
    text(ctx,71,y,label,bright,17,"right");
    text(ctx,700,y,String(votes[i]),i+1===data.neighbors.prediction?mint:muted,20,"center");
  });
  points.slice(0,Math.ceil(points.length*p)).forEach(({distance,label},i)=>{
    const x=xp(distance),y=62+(label-1)*38+(i%3-1)*7;
    dot(ctx,x,y,5.2,colors[label-1]);
  });
  text(ctx,left,304,"0",muted,14);
  text(ctx,right,304,max.toFixed(2)+" 欧氏距离",muted,14,"right");
}
function drawForest(ctx,data,p){
  const forest=data.forest;
  const position=slot=>{
    const depth=Math.floor(Math.log2(slot+1)),cells=2**depth;
    return [(slot-cells+1+.5)*760/cells,47+depth*74];
  };
  const visible=forest.nodes.filter(n=>Math.floor(Math.log2(n.slot+1))<=Math.floor(p*3));
  for(const n of visible){
    if(n.slot===0)continue;
    const [x,y]=position(n.slot),[a,b]=position(Math.floor((n.slot-1)/2));
    line(ctx,[[a,b+22],[x,y-22]],n.path?mint:"#48566a",n.path?3:1);
    text(ctx,(a+x)/2,(b+y)/2-5,n.slot%2?"≤":">",n.path?mint:muted,15,"center");
  }
  for(const n of visible){
    const [x,y]=position(n.slot);
    box(ctx,x-78,y-21,156,42,n.path?"#395844":"#2a3546");
    text(ctx,x,y-7,n.feature??"叶子",n.path?mint:bright,15,"center");
    text(ctx,x,y+11,n.feature?"阈值 "+n.threshold.toFixed(2):"输出类别概率",muted,13,"center");
  }
  const single=forest.single.indexOf(Math.max(...forest.single));
  const root=forest.nodes[0];
  text(ctx,26,236,"当前窗 · "+root.feature+" = "+root.value.toFixed(2)+(root.value<=root.threshold?" ≤ ":" > ")+root.threshold.toFixed(2),muted,15);
  text(ctx,26,271,"第 1 棵树 · 路径 "+forest.depth+" 层 · "+gestures[single]+" "+percent(forest.single[single]),bright,17);
  text(ctx,26,307,"100 棵树平均 · "+gesture(forest.prediction)+" "+percent(forest.probabilities[forest.prediction-1]),mint,19);
}
function drawScores(ctx,data,p){
  text(ctx,28,30,"最佳验证 macro-F1",bright,20);
  ["logistic","knn","forest"].forEach((id,i)=>{
    const score=data.best[id].macro_f1,y=68+i*72;
    text(ctx,28,y+13,modelNames[id],bright,17);
    box(ctx,143,y,500,25,"#283544");
    box(ctx,143,y,500*score*p,25,id==="forest"?mint:"#a493d4");
    text(ctx,725,y+13,score.toFixed(4),id==="forest"?mint:bright,20,"right");
  });
  text(ctx,143,298,"0.0",muted,15);text(ctx,643,298,"1.0",muted,15,"right");
}
