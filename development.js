"use strict";
const el = id => document.getElementById(id);
const experiment = window.EMG_DEVELOPMENT;
const gestures = ["静息","握拳","腕屈","腕伸","桡偏","尺偏"];
const modelNames = {logistic:"逻辑回归",knn:"kNN",forest:"随机森林"};
const colors = ["#92d7be","#bef792","#eac885","#c4b5fa","#79bdfa","#ed9fa7"];
const mint = "#bef792", muted = "#a4b0c1", bright = "#eaf0f7";
const percent = value => (100 * value).toFixed(1) + "%";
const gesture = label => gestures[label-1];
const GROUPS = [["train","训练集","拟合模型"],["validation","验证集","比较候选"],["test","测试集","评估获选模型"]];

if (experiment?.matrix && experiment.curve.logits) startDevelopment(experiment);
else el("devMissing").hidden = false;

function startDevelopment(data) {
  const row = data.sample;
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
     caption:"点 RMS、MAV 或 WL，重播采样点到特征的计算。拖动波形可检查任意前缀，WL 会连起最新的一对相邻点。",
     facts:[["RMS","平方 → 求均值 → 开方"],["MAV","取绝对值 → 求均值"],["WL（可选）","相邻点作差 → 取绝对值 → 求和"]], draw:drawExtraction},
    {tab:"训练矩阵", title:"每个窗口变成一行", description:"对训练组的每个窗口重复提取。每行是一个窗，每列是一条通道的一种特征，旁边的标签组成 y_train。标准化使用各列在训练集上的均值和标准差。",
     caption:"显示最终 24 维方案的六个真实训练窗口和其中四列。点击「应用标准化」，查看同一矩阵怎样逐列转换。",
     facts:[[data.matrix.shape[0]+" × "+data.matrix.shape[1],"完整 X_train 的形状"],["16 维 / 24 维","不含 WL / 含 WL 两组候选"],["z = (x − μ) / σ","μ 和 σ 只由训练集拟合"]], draw:drawTrainingMatrix},
    {tab:"逻辑回归", title:"从特征权重到分类概率", description:"先把标准化特征与每类权重相乘，再加偏置，得到六个分数。Softmax 把这些分数转换成总和为 1 的概率。最后可以单独改变一个特征，观察预测怎么变。",
     caption:"三种视图都从同一个验证窗出发。线性打分展示拟合后的真实参数，曲线视图固定其余特征。",
     facts:result("logistic",data.curve.prediction), draw:drawLogisticDetails},
    {tab:"kNN", title:"最近的邻居怎么投票？", description:"每个训练窗口是一条带动作标签的特征向量。按标准化特征空间的欧氏距离排序，从前 k 个训练窗口中取票数最多的动作。",
     caption:"每个圆点是一个真实训练窗口。滑动 k 改变参与投票的邻居数，较暗的点不计票。实验最终比较的是 k = 5 和 k = 15。",
     facts:result("knn",data.neighbors.prediction), draw:drawKnn},
    {tab:"随机森林", title:"从一棵树走到一片森林", description:"树按特征阈值依次判断。当前窗口在每个节点选择左边或右边，走到叶子得到一组类别概率。森林平均 100 棵树的概率，再选出最高类。",
     caption:"点击前两次判断，观察当前特征如何跨过阈值。点击「汇总结果」看第 1 棵树和完整森林的输出。",
     facts:result("forest",data.forest.prediction), draw:drawForest},
    {tab:"模型选型", title:"把验证集成绩放在一起", description:"每种模型在同一批验证用户上比较参数及特征组合。验证 macro-F1 最高的方案被选中，再用留出的测试用户评估。",
     caption:"柱长从 0 开始，显示各模型最佳验证 macro-F1。展开表格可检查完整的 12 组配置。",
     facts:[[data.best.forest.features+" · 100 棵树","最终配置"],[(100*data.test.accuracy).toFixed(2)+"%","留出测试准确率"],[data.test.macro_f1.toFixed(4),"留出测试 macro-F1"]], draw:drawScores},
    {tab:"产物与参数", title:"训练之后留下了什么？", description:"查看训练集的特征与类别关系、标准化参数，以及三个模型保存的信息。浏览器从训练产物复现分窗后的分类过程。",
     caption:"热图显示各类别的平均标准化特征。可切到标准化参数、逻辑回归权重或获选森林的第 1 棵树。",
     facts:[["24 列 μ 与 σ","获选方案的标准化参数"],["权重、邻居与决策树","三种模型各自保留的训练信息"],["100 棵树","随网页导出的获选森林"]], draw:drawArtifacts}
  ];
  const canvas=el("devCanvas"), context=canvas.getContext("2d"), slider=el("devFeature");
  const channel=el("devChannel"), includeWL=el("devIncludeWL");
  const samples=el("devSample"), neighbors=el("devKnn"), compare=el("devCompare");
  const artifactMetric=el("devArtifactMetric"), artifactClass=el("devArtifactClass");
  const views={metric:"rms",matrix:"raw",logistic:"scores",artifact:"relation"};
  const viewButtons=[...document.querySelectorAll("button[data-view]")];
  const groups=[...el("devGroups").querySelectorAll("button")];
  const branches=[...el("devBranches").querySelectorAll("button")];
  const controls=["devGroups","devExtraction","devMatrix","devLogic","devNeighbors","devBranches","devComparison","devArtifacts"];
  const hints=["点击数据组","点 RMS、MAV 或 WL 播放计算","切换组装矩阵与标准化","依次查看打分、概率和单特征曲线","拖动 k，观察投票如何累积","点击判断步骤，跟着窗口走","切换特征组合，比较验证分数","选择要查看的训练产物"];
  data.sample.waveform?.forEach((_,i)=>channel.add(new Option("CH "+(i+1),i)));
  gestures.forEach((name,i)=>artifactClass.add(new Option(name,i)));
  artifactClass.value=data.curve.prediction-1;
  neighbors.max=data.neighbors.points.length;
  neighbors.value=neighbors.max;
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  let step=0, animation=0, progress=1, group="train", depth=0, drag=null;
  function settings(){
    const k=Number(neighbors.value);
    const votes=gestures.map((_,i)=>data.neighbors.points.slice(0,k).filter(p=>p.label===i+1).length);
    const scores=compare.value==="all"?data.best:Object.fromEntries(Object.keys(modelNames).map(id=>[
      id,data.scores.filter(row=>row.model===id&&row.features===compare.value).reduce((a,b)=>a.macro_f1>=b.macro_f1?a:b)
    ]));
    return {group, depth, channel:Number(channel.value), wl:includeWL.checked, count:Number(samples.value),
      point:Number(slider.value), k, votes, scores, ...views,
      artifactMetric:Number(artifactMetric.value), artifactClass:Number(artifactClass.value)};
  }
  function readout(){
    const state=settings();
    const currentGroup=GROUPS.find(([id])=>id===group);
    const winner=state.votes.indexOf(Math.max(...state.votes));
    const selectedPoint=data.curve.p[state.point];
    const row=selectedPoint.indexOf(Math.max(...selectedPoint));
    const scoreOrder=Object.entries(state.scores).sort((a,b)=>b[1].macro_f1-a[1].macro_f1);
    const values=[
      currentGroup[1]+" · "+data.users[group]+" 人 · "+data.windows[group]+" 个窗口",
      "前 "+state.count+" 点 · "+state.metric.toUpperCase()+" 计算 · "+(state.wl?"24 维方案":"16 维方案"),
      state.matrix==="raw" ? "X_train · 每个窗一行，每个特征一列" : "逐列应用训练集的参数 · z = (x − μ) / σ",
      state.logistic==="curve" ? data.curve.feature+" · "+gesture(row+1)+" 概率最高 "+percent(selectedPoint[row])
        : state.logistic==="scores" ? "标准化特征 × 权重 + 偏置 → 六个原始分数"
        : gesture(data.curve.prediction)+" 概率最高 · "+percent(data.curve.probabilities[data.curve.prediction-1]),
      "当前 k = "+state.k+" · "+gestures[winner]+" 得票最多（"+state.votes[winner]+" 票）",
      ["从第 1 棵树的根节点开始","完成第 1 次阈值判断","完成第 2 次阈值判断","合并 100 棵树的概率"][depth],
      "当前比较领先 · "+modelNames[scoreOrder[0][0]]+" · "+scoreOrder[0][1].macro_f1.toFixed(4),
      {relation:"按训练标签聚合 · 显示每类在各通道的标准化均值",scale:"各列的 μ 和 σ 来自训练窗口",
        weights:"逻辑回归权重 · kNN 的 k · 随机森林树数",tree:"获选森林 · 第 1 棵树的前两层分裂"}[state.artifact]
    ];
    el("devLive").textContent=values[step];
    el("devSampleCount").textContent=state.count;
    el("devKnnCount").textContent=state.k;
  }
  function render(amount){
    progress=amount;
    const dpr=Math.min(devicePixelRatio||1,2);
    const width=Math.round(canvas.clientWidth*dpr),height=Math.round(canvas.clientHeight*dpr);
    if(width<1 || height<1) return;
    if(canvas.width!==width)canvas.width=width;
    if(canvas.height!==height)canvas.height=height;
    context.setTransform(width/760,0,0,height/335,0,0);
    context.clearRect(0,0,760,335);
    const state=settings();
    stages[step].draw(context,data,amount,state);
  }
  const buttons=stages.map((stage,i)=>{
    const b=document.createElement("button");
    b.type="button";b.innerHTML="<span>"+String(i+1).padStart(2,"0")+"</span><span>"+stage.tab+"</span>";
    b.onclick=()=>show(i);el("devSteps").append(b);return b;
  });
  function animate(duration=720){
    cancelAnimationFrame(animation);
    const start=performance.now();
    function tick(now){
      const t=reduced?1:Math.min(1,Math.max(0,(now-start)/duration));
      render(1-(1-t)**2);
      if(t<1)animation=requestAnimationFrame(tick);
    }
    tick(start);
  }
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
    controls.forEach((id,n)=>el(id).hidden=n!==step);
    el("devInteraction").textContent=hints[step];
    updateViews();
    el("devPrev").disabled=step===0;
    el("devNext").hidden=step===stages.length-1;
    el("devCandidates").hidden=stage.draw!==drawScores;
    el("devFinish").hidden=step!==stages.length-1;
    el("devPosition").textContent=(step+1)+" / "+stages.length;
    if(stage.draw===drawLogisticDetails)slider.value=Math.floor(data.curve.x.length/2);
    if(stage.draw===drawExtraction)updateCount();
    readout();
    if(!reduced)el("devTitle").animate([{opacity:.35,transform:"translateY(7px)"},{opacity:1,transform:"none"}],{duration:300,easing:"ease-out"});
    animate();
  }
  function scrub(event){
    const rect=canvas.getBoundingClientRect();
    const x=(event.clientX-rect.left)*760/rect.width;
    const [input,left,right,low,high]=step===1?[samples,58,730,1,200]:[slider,78,725,0,data.curve.x.length-1];
    const fraction=Math.max(0,Math.min(1,(x-left)/(right-left)));
    input.value=low+Math.round(fraction*(high-low));
    refresh();
  }
  canvas.addEventListener("pointerdown",event=>{
    if(isScrubbable()){
      drag=event.pointerId;canvas.setPointerCapture(drag);scrub(event);
    }else if(step===0){
      const y=(event.clientY-canvas.getBoundingClientRect().top)*335/canvas.clientHeight;
      const index=Math.floor((y-65)/79);
      if(GROUPS[index])selectGroup(GROUPS[index][0]);
    }
  });
  canvas.addEventListener("pointermove",event=>{if(drag===event.pointerId && isScrubbable())scrub(event)});
  canvas.addEventListener("lostpointercapture",()=>drag=null);
  function selectGroup(id){
    group=id;
    groups.forEach(b=>b.setAttribute("aria-pressed",b.dataset.group===id));
    readout();animate(380);
  }
  function selectDepth(value){
    depth=value;
    branches.forEach(b=>b.setAttribute("aria-pressed",Number(b.dataset.depth)===depth));
    readout();animate(450);
  }
  function refresh(){
    cancelAnimationFrame(animation);
    readout();render(1);
  }
  el("devPrev").onclick=()=>show(step-1);
  el("devNext").onclick=()=>show(step+1);
  el("devReplay").onclick=()=>animate();
  function isScrubbable(){
    const drawing=stages[step].draw;
    return drawing===drawExtraction || (drawing===drawLogisticDetails && views.logistic==="curve");
  }
  function updateViews(){
    viewButtons.forEach(b=>b.setAttribute("aria-pressed",views[b.dataset.view]===b.dataset.value));
    el("devSweep").hidden=views.logistic!=="curve";
    el("devMetricControl").hidden=views.artifact!=="relation" && views.artifact!=="scale";
    el("devClassControl").hidden=views.artifact!=="weights";
    canvas.classList.toggle("scrubbable",isScrubbable());
  }
  viewButtons.forEach(button=>button.onclick=()=>{
    const {view,value}=button.dataset;
    views[view]=value;
    if(view==="metric" && value==="wl"){
      includeWL.checked=true;
      updateCount();
    }
    updateViews();readout();animate(view==="metric" ? 1350 : 780);
  });
  groups.forEach(b=>{
    b.setAttribute("aria-pressed",b.dataset.group===group);
    b.onclick=()=>selectGroup(b.dataset.group);
  });
  branches.forEach(b=>{
    b.setAttribute("aria-pressed",Number(b.dataset.depth)===depth);
    b.onclick=()=>selectDepth(Number(b.dataset.depth));
  });
  slider.oninput=samples.oninput=neighbors.oninput=refresh;
  compare.onchange=()=>{readout();animate(550)};
  function updateCount() {
    const formula=includeWL.checked ? "8 通道 × 3 特征 → 24 维" : "8 通道 × 2 特征 → 16 维";
    el("devFeatureCount").textContent=formula;
  }
  channel.onchange=()=>{readout();animate(430)};
  includeWL.onchange=()=>{
    if(!includeWL.checked && views.metric==="wl")views.metric="rms";
    updateViews();updateCount();refresh();
  };
  artifactMetric.onchange=artifactClass.onchange=()=>{readout();animate(480)};
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
function drawSplit(ctx,data,p,{group}){
  const users=Object.values(data.users).reduce((sum,n)=>sum+n,0);
  text(ctx,30,35,"从 "+users+" 名受试者到三组窗口",bright,20);
  const max=Math.max(...Object.values(data.windows));
  GROUPS.forEach(([id,name,role],i)=>{
    const y=77+i*79;
    if(id===group)box(ctx,16,y-11,721,65,"#233349");
    text(ctx,30,y+8,name,bright,18);text(ctx,30,y+34,role,muted,14);
    box(ctx,145,y,475,22,"#273343");
    box(ctx,145,y,475*data.windows[id]/max*(id===group?p:1),22,[mint,"#a990e6","#74b8f0"][i]);
    text(ctx,145,y+40,data.windows[id]+" 窗口",muted,15);
    text(ctx,718,y+10,data.users[id]+" 人",bright,20,"right");
  });
}
function drawExtraction(ctx,data,p,{channel,metric,count:requested}){
  if (!data.sample.waveform) {
    text(ctx,35,80,"请重新生成演示数据，以加载窗口波形。",muted,19);
    return;
  }
  const signal=data.sample.waveform[channel];
  const left=58,right=730,center=115,half=78;
  const limit=Math.max(...signal.map(Math.abs),1e-8);
  const x=i=>left+(right-left)*i/(signal.length-1);
  const y=v=>center-half*v/limit;
  const count=Math.max(1,Math.ceil(requested*p));
  const visible=signal.slice(0,count);
  const rms=Math.sqrt(visible.reduce((sum,value)=>sum+value**2,0)/count);
  const mav=visible.reduce((sum,value)=>sum+Math.abs(value),0)/count;
  const length=visible.slice(1).reduce((sum,value,i)=>sum+Math.abs(value-visible[i]),0);

  text(ctx,left,19,"CH "+(channel+1)+" · 原始肌电",bright,17);
  line(ctx,[[left,center],[right,center]],"#425065");
  ctx.setLineDash([5,5]);
  if(metric!=="wl"){
    const amplitude=metric==="rms"?rms:mav;
    const color=metric==="rms"?"#a990e6":"#d1b679";
    line(ctx,[[left,y(amplitude)],[right,y(amplitude)]],color);
    line(ctx,[[left,y(-amplitude)],[right,y(-amplitude)]],color);
  }
  ctx.setLineDash([]);
  line(ctx,signal.map((v,i)=>[x(i),y(v)]),"#465966");
  line(ctx,visible.map((v,i)=>[x(i),y(v)]),mint,2);
  line(ctx,[[x(count-1),34],[x(count-1),198]],"#91b5a6");
  dot(ctx,x(count-1),y(signal[count-1]),4,mint);
  text(ctx,left,213,"0",muted,14);
  text(ctx,right,213,"199 采样点",muted,14,"right");

  const latest=signal[count-1];
  const delta=count>1 ? latest-signal[count-2] : 0;
  if(metric==="wl" && count>1){
    const edge=count-1;
    line(ctx,[[x(edge-1),y(signal[edge-1])],[x(edge),y(signal[edge])]],"#79bdfa",4);
    dot(ctx,x(edge-1),y(signal[edge-1]),4,"#79bdfa");
    dot(ctx,x(edge),y(signal[edge]),4,"#79bdfa");
  }
  const stages={
    rms:[["采样值 x",latest],["当前点平方 x²",latest**2],["前缀平方均值",rms**2],["开方 → RMS",rms]],
    mav:[["采样值 x",latest],["当前绝对值 |x|",Math.abs(latest)],["平均 → MAV",mav]],
    wl:[["最近相邻差 Δ",delta],["相邻差绝对值 |Δ|",Math.abs(delta)],["累计 → WL",length]]
  }[metric];
  const terms=metric==="wl" ? count-1+" 个相邻差" : count+" 个点";
  text(ctx,left,242,"已纳入 "+terms,muted,15);
  calculationCards(ctx,stages,p,left,right);
}
function drawLogistic(ctx,data,p,{point}){
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
function drawKnn(ctx,data,p,{k}){
  const points=data.neighbors.points,max=points.at(-1).distance,left=106,right=630;
  const shown=Math.ceil(k*p);
  const votes=gestures.map((_,i)=>points.slice(0,shown).filter(point=>point.label===i+1).length);
  const winner=shown ? votes.indexOf(Math.max(...votes)) : -1;
  const xp=distance=>left+(right-left)*distance/max;
  text(ctx,left,30,"离当前窗更近",muted,15);text(ctx,right,30,"更远",muted,15,"right");text(ctx,700,30,"票数",muted,15,"center");
  gestures.forEach((label,i)=>{
    const y=62+i*38;
    line(ctx,[[left,y],[right,y]],"#344252");
    text(ctx,71,y,label,bright,17,"right");
    text(ctx,700,y,String(votes[i]),i===winner?mint:muted,20,"center");
  });
  points.forEach(({distance,label},i)=>{
    const x=xp(distance),y=62+(label-1)*38+(i%3-1)*7;
    const included=i<shown;
    dot(ctx,x,y,included?5.2:3,included?colors[label-1]:"#465363");
  });
  text(ctx,left,304,"0",muted,14);
  text(ctx,right,304,max.toFixed(2)+" 欧氏距离",muted,14,"right");
}
function drawForest(ctx,data,p,{depth}){
  const forest=data.forest;
  const position=slot=>{
    const depth=Math.floor(Math.log2(slot+1)),cells=2**depth;
    return [(slot-cells+1+.5)*760/cells,47+depth*74];
  };
  const shown=Math.min(2,Math.floor(depth*p+.05));
  const visible=forest.nodes.filter(n=>Math.floor(Math.log2(n.slot+1))<=shown);
  for(const n of visible){
    if(n.slot===0)continue;
    const [x,y]=position(n.slot),[a,b]=position(Math.floor((n.slot-1)/2));
    line(ctx,[[a,b+22],[x,y-22]],n.path?mint:"#48566a",n.path?3:1);
    text(ctx,(a+x)/2,(b+y)/2-5,n.slot%2?"≤":">",n.path?mint:muted,15,"center");
  }
  for(const n of visible){
    const [x,y]=position(n.slot);
    ctx.globalAlpha=.6+.4*p;
    box(ctx,x-78,y-21,156,42,n.path?"#395844":"#2a3546");
    text(ctx,x,y-7,n.feature??"叶子",n.path?mint:bright,15,"center");
    text(ctx,x,y+11,n.feature?"阈值 "+n.threshold.toFixed(2):"输出类别概率",muted,13,"center");
    ctx.globalAlpha=1;
  }
  const single=forest.single.indexOf(Math.max(...forest.single));
  const decisions=forest.nodes.filter(n=>n.path && n.feature).sort((a,b)=>a.slot-b.slot);
  const node=decisions[Math.max(0,shown-1)];
  text(ctx,26,236,"当前窗 · "+node.feature+" = "+node.value.toFixed(2)+(node.value<=node.threshold?" ≤ ":" > ")+node.threshold.toFixed(2),muted,15);
  if(depth===3 && p>.7){
    text(ctx,26,271,"第 1 棵树 · 路径 "+forest.depth+" 层 · "+gestures[single]+" "+percent(forest.single[single]),bright,17);
    text(ctx,26,307,"100 棵树平均 · "+gesture(forest.prediction)+" "+percent(forest.probabilities[forest.prediction-1]),mint,19);
  }
}
function drawScores(ctx,_data,p,{scores}){
  text(ctx,28,30,"最佳验证 macro-F1",bright,20);
  const winner=Object.keys(scores).reduce((a,b)=>scores[a].macro_f1>=scores[b].macro_f1?a:b);
  ["logistic","knn","forest"].forEach((id,i)=>{
    const score=scores[id].macro_f1,y=68+i*72;
    text(ctx,28,y+13,modelNames[id],bright,17);
    box(ctx,143,y,500,25,"#283544");
    box(ctx,143,y,500*score*p,25,id===winner?mint:"#a493d4");
    text(ctx,725,y+13,score.toFixed(4),id===winner?mint:bright,20,"right");
  });
  text(ctx,143,298,"0.0",muted,15);text(ctx,643,298,"1.0",muted,15,"right");
}
