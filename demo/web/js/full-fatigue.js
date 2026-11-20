"use strict";
(() => {
  const host=document.getElementById("fullLab"), data=window.fullFatigue;
  if(!data){host.textContent="缺少疲劳实验快照，请运行 build_pages.py。";return;}
  const m=data.metrics, labels=["无疲劳","向疲劳过渡","疲劳"];
  const methods={four:"同侧四路肌电",agonist:"单路主动力肌",four_time:"四路肌电 + 运动时长",elapsed:"仅运动时长",majority:"多数类"};
  const v1=data.metrics_v1;
  const records=new Map();
  for(const row of data.test_predictions){
    const id=row.subject+" / "+row.trial;
    if(!records.has(id))records.set(id,[]);
    records.get(id).push(row);
  }
  let id=records.has("10 / 11")?"10 / 11":records.keys().next().value,index=0,timer,step=0;
  const $=id=>document.getElementById(id);
  const table=(headers,rows)=>'<div class="scrollTable"><table><thead><tr>'+headers.map(h=>`<th>${h}</th>`).join("")+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>`<td>${v}</td>`).join("")+'</tr>').join("")+'</tbody></table></div>';
  const fmt=n=>n.toFixed(3);
  // 有序三档沿用三档力量的配色：暗 → 青绿 → 薄荷绿。
  const colors=["#344052","#4f8f7c","#b3f690"];
  function stop(){clearInterval(timer);timer=null;if($("fullPlay"))$("fullPlay").textContent="▶ 播放";}
  function compare(){
    const head=["输入","固定测试 F1","四折合并 F1","逐人平均 F1",...(v1?["第一版四折 F1"]:[])];
    return table(head,Object.entries(methods).map(([k,n])=>[n,fmt(m.test[k].macro_f1),fmt(m.grouped_cv[k].macro_f1),fmt(data.per_subject.reduce((s,r)=>s+r[k],0)/data.per_subject.length),
      ...(v1?[v1.grouped_cv[k]?fmt(v1.grouped_cv[k].macro_f1):"—"]:[])]));
  }
  function replay(){
    const rows=records.get(id), last=rows.length-1, lo=rows[0].seconds, hi=rows[last].seconds+4;
    host.innerHTML=`<section class="caseGrid labReveal"><article class="modelCard"><div class="caseTools"><div><p class="eyebrow">HELD-OUT RECORDING</p><h2>评分与预测时间线</h2></div><label>用户 / 试验 <select id="fullCase"></select></label></div><div class="bandLegend">${labels.map((l,i)=>`<span><i style="background:${colors[i]}"></i>${l}</span>`).join("")}</div><svg class="casePlot" viewBox="0 0 680 182" role="img" aria-label="参与者评分、四路肌电预测与加入运动时长的预测，空白表示缺少评估窗口"></svg><p class="demoDetail">按窗起点标记，空白处缺少可评估窗口。评分在每场试验中只升不降，模型预测也按时间取累计最大值。</p><div class="transport"><button id="fullPlay">▶ 播放</button><label class="slider"><input id="fullSeek" aria-label="选择窗口" type="range" min="0" max="${last}" value="${index}"></label><output id="fullClock"></output></div></article><article class="modelCard"><p class="eyebrow">MODEL OUTPUT</p><h2>当前窗口预测</h2><div class="demoValue" id="fullValue"></div><div id="fullDetail"></div></article></section>`;
    const x=t=>80+580*(t-lo)/(hi-lo);
    host.querySelector("svg").innerHTML=[["label","评分"],["four","四路肌电"],["four_time","肌电+时长"]].map(([k,name],j)=>`<text x="72" y="${44+j*41}" text-anchor="end">${name}</text>`+
      rows.map(r=>`<rect x="${x(r.seconds)}" y="${26+j*41}" width="${x(r.seconds+2)-x(r.seconds)}" height="25" fill="${colors[r[k]]}"/>`).join("")).join("")+
      `<line id="fullCursor" y1="12" y2="148" stroke="white" stroke-width="1.5"/><text x="80" y="170">${lo.toFixed(1)} s</text><text x="660" y="170" text-anchor="end">${hi.toFixed(1)} s</text>`;
    for(const key of records.keys())$("fullCase").add(new Option(key,key));
    $("fullCase").value=id;
    $("fullCase").onchange=e=>{stop();id=e.target.value;index=0;replay();$("fullCase").focus();};
    function update(){const r=rows[index];$("fullSeek").value=index;$("fullValue").textContent=labels[r.four];$("fullClock").textContent=r.seconds.toFixed(1)+" s";
      $("fullCursor").setAttribute("x1",x(r.seconds));$("fullCursor").setAttribute("x2",x(r.seconds));
      $("fullDetail").innerHTML=table(["对照","当前窗"],[["参与者评分",labels[r.label]],["四路 + 运动时长",labels[r.four_time]],["主动力肌模型",labels[r.agonist]],["仅用运动时长",labels[r.elapsed]]]);}
    $("fullSeek").oninput=e=>{stop();index=+e.target.value;update();};
    $("fullPlay").onclick=()=>{if(timer)return stop();if(index===last)index=0;update();$("fullPlay").textContent="Ⅱ 暂停";timer=setInterval(()=>{index=Math.min(index+1,last);update();if(index===last)stop();},500);};
    update();
  }
  const stages=[
    ["数据与评分","按用户和试验配对","左右臂共覆盖八块肌肉，每场同步记录同侧四路。沿时间戳对齐独立评分，保留评分一致的完整窗口。"],
    ["特征","把每窗信号转换成 60 个特征","每路计算三项对数幅值、平均频率和中位频率（4 s 窗、2 s 步长）。第二版先在过去约 30 秒内因果平滑，再减去本场开头约 10 秒的个人基线，并加入 30 秒变化趋势；全部只用当前及过去的窗口。"],
    ["时间约束","评分只升不降","156 场试验的评分全部单调不减。逐窗分类器的输出会来回跳动，因此按时间对预测取累计最大值：一旦判为更疲劳，之后不再回落。"],
    ["训练选型","在验证用户上选择分类器","固定划分中，四路、单路和“四路 + 运动时长”三种输入各自比较类别平衡逻辑回归及两种叶节点大小的极端随机树，按验证用户平均 macro-F1 选择。特征设计只在 11 名开发用户上比较确定，测试用户不参与。"],
    ["跨用户评估","逐人检查泛化能力","四折外层按人留出；每折只在训练人员中选型，再预测该折未见用户。下表每人只出现一次。"]
  ];
  function addConfusionMatrix(){
    const section=document.createElement("section");
    section.className="modelCard labReveal";
    section.innerHTML='<h2>从误判回到试验</h2><p>行是参与者评分，列是四路模型预测。点击计数跳转到该组合的第一个测试窗口。</p>';
    const counts=labels.map(()=>labels.map(()=>[]));
    for(const [key,rows] of records){
      rows.forEach((row,i)=>counts[row.label][row.four].push({key,i}));
    }
    section.insertAdjacentHTML("beforeend",table(["评分 / 预测",...labels],counts.map((row,i)=>[
      labels[i],...row.map((items,j)=>`<button data-true="${i}" data-pred="${j}" ${items.length?"":"disabled"} aria-label="${labels[i]}预测为${labels[j]}，${items.length}窗">${items.length}</button>`)
    ])));
    section.addEventListener("click",event=>{
      const button=event.target.closest("button[data-true]");
      if(!button)return;
      const first=counts[button.dataset.true][button.dataset.pred][0];
      id=first.key;
      index=first.i;
      location.hash="full/replay";
    });
    host.append(section);
  }
  function stageDetails(i){
    if(i===0)return table(["划分","人员编号"],Object.entries(m.groups).map(([k,v])=>[k,v.join("、")]));
    if(i===1)return table(["采样率","窗长 / 步长","滤波","平滑 / 基线 / 趋势"],[["1259 Hz","4 s / 2 s","20–450 Hz","15 / 5 / 15 窗"]]);
    if(i===2)return table(["场次","评分单调不减"],[[m.records,"全部"]]);
    if(i===3)return table(["输入","候选","验证 F1"],data.validation.map(r=>[methods[r.scope],r.model,fmt(r.validation_subject_macro_f1)]));
    return table(["用户","四路 F1","四路+时长 F1","单路 F1","时长 F1"],data.per_subject.map(r=>[r.subject,fmt(r.four),fmt(r.four_time),fmt(r.agonist),fmt(r.elapsed)]));
  }
  function render(){
    stop();const view=modelRoute().view;
    document.querySelectorAll("[data-view]").forEach(a=>{if(a.dataset.view===view)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current");});
    $("sourceMode").textContent="Zenodo / "+({training:"训练过程",development:"开发过程",replay:"离线回放"})[view];
    if(view==="replay")return replay();
    if(view==="training"){
      const wins=data.per_subject.filter(r=>r.four>r.elapsed).length, both=data.per_subject.filter(r=>r.four_time>r.elapsed).length;
      host.innerHTML=`<div id="fullTraining"></div><section class="modelCard labReveal"><p class="eyebrow">HELD-OUT USERS</p><h2>未见用户上的完整对照</h2><p>${m.records} 场试验 · ${m.windows.toLocaleString()} 个窗口 · 获选四路模型 ${m.selected}</p>${compare()}<p class="demoDetail">四路模型在 ${wins}/${data.per_subject.length} 人上超过运动时长基线，加入运动时长后为 ${both}/${data.per_subject.length}。第二版明显改善了纯肌电模型，但仍不如只看运动时长；现有数据不足以证明肌电能比运动时长更好地预测新用户的自感疲劳。</p></section>`;
      if(window.TrainingView)window.TrainingView.mount($("fullTraining"),"fatigue_full");
      addConfusionMatrix();return;
    }
    const item=stages[step];
    host.innerHTML=`<section class="modelCard labReveal"><p class="eyebrow">DEVELOPMENT</p><h2>四路疲劳模型的开发过程</h2><nav class="demoNav" id="steps" aria-label="开发阶段"></nav><div class="demoScene"><div><h3>${item[1]}</h3><p class="demoDetail">${item[2]}</p></div>${stageDetails(step)}</div><div class="sceneControls"><button id="back">← 上一步</button><output>${step+1} / ${stages.length}</output><button id="forward">下一步 →</button></div></section>`;
    stages.forEach(([name],i)=>{const b=document.createElement("button");b.textContent=name;if(i===step)b.setAttribute("aria-current","step");b.onclick=()=>{step=i;render();};$("steps").append(b);});
    $("back").disabled=step===0;$("forward").disabled=step===stages.length-1;
    $("back").onclick=()=>{step--;render();};$("forward").onclick=()=>{step++;render();};
  }
  addEventListener("hashchange",render);addEventListener("pagehide",stop);document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});render();
})();
