"use strict";
(() => {
  const host = document.getElementById("modelDemo");
  const demos = window.modelDemos;
  const names = {levels:"力度三档",change:"两日力量变化",fatigue:"单路肌肉疲劳"};
  const classes = {levels:["静息","轻","重"],fatigue:["无疲劳","向疲劳过渡","疲劳"]};
  const f = (value, places=2) => Number(value).toFixed(places);
  const signed = value => (value>0?"+":"")+f(value,1);
  const percent = value => f(value*100,1)+"%";
  const $ = id => document.getElementById(id);
  const state = {};
  let key, data, s, rows, pair, timer;
  const scenes = {
    levels:[
      ["动作标签","从实测力量定义三档","先扣除静息零点与慢漂移，取第二强的传感器读数除以 MVC。当前窗的实测比例决定训练标签。","targets.py · level_labels"],
      ["信号特征","24 路肌电进入同一个窗口","每窗 200 ms，步长 100 ms。每通道取 RMS、MAV、WL，以本记录前四秒静息特征为参照，在训练用户上拟合标准化器。","preprocessing.py · record_features"],
      ["训练选型","按验证用户比较分类器","比较逻辑回归与随机森林的特征和参数组合。验证 macro-F1 越高越好。","levels.py · 训练候选"],
      ["窗口推理","三档概率如何变成结果","对森林输出的类别概率作因果平滑，概率最大的档位成为当前窗口的预测。","model.py · ForceLevelModel.predict_proba"],
      ["留出评估","在未见用户上比较三种方法","对比专门训练的三档模型、连续回归后分档和多数类基线。macro-F1 给每一档相同的权重。","levels.py · 测试评估"]
    ],
    change:[
      ["动作取样","为两天选择同一种保持动作","只取第四组单独出力的稳定段。20% 与 40% 是相对于各次记录 MVC 的指令幅度，两天分别按同一档位比较。","change_experiment.py · collect"],
      ["十路特征","一次保持动作压缩成十个数","冻结第一阶段随机森林。只保留窗口及平滑历史完全落在保持段内的预测，再对每一路取中位数。","change.py · trial_features"],
      ["训练选型","学习十路特征如何合成力量","用拟合用户的实测 FORCE_7–10 去静息后的均值构造参考。训练标准化岭回归，在独立验证用户上选跨日 MAE 最低的参数。","change_experiment.py · evaluate"],
      ["跨日输出","先取每日中位数，再相减","每次动作产生一个估计值。以复测日中位数减去基线日中位数，得到最终有符号的变化值。","change_predict.py · run"],
      ["留出评估","检查幅度与变化方向","九名测试用户各完成两个档位。一个人贡献两个比较，误差的单位沿用设备刻度。个人牛顿输出另需同步测力计标定。","change_experiment.py · scores"]
    ],
    fatigue:[
      ["用户划分","保留独立的测试参与者","选取 trial_5 的右肱二头肌。训练、验证和测试按人分开，避免相邻重叠窗口跨集合。","single_channel.py · 用户划分"],
      ["滤波分窗","把主观评分对齐到肌电窗口","20–450 Hz 因果带通后，按约四秒窗、两秒步长切分。只评估整个窗口被同一评分覆盖的片段。","windowing.py · 纯标签窗口"],
      ["训练选型","用幅值与频谱区分疲劳状态","比较 kNN 与岭分类器。当前获选模型同时使用三项对数幅值特征、平均频率与中位频率。","single_channel.py · 验证候选"],
      ["窗口推理","给当前窗选择一个疲劳等级","训练集统计量标准化五项特征后，岭模型计算三类得分，取最高分。可与当时参与者自己的评分对照。","single_channel.py · predict"],
      ["留出评估","比较肌电与仅用时间的基线","测试仅有两人。按 macro-F1 衡量，单路肌电超过多数类基线，低于仅用运动时长的基线。","single_channel.py · 测试指标"]
    ]
  };
  function table(headers, entries) {
    return '<div class="scrollTable"><table><thead><tr>'+headers.map(v=>`<th>${v}</th>`).join("")+
      '</tr></thead><tbody>'+entries.map(r=>'<tr>'+r.map(v=>`<td>${v}</td>`).join("")+'</tr>').join("")+'</tbody></table></div>';
  }
  function bars(items, max=1) {
    return items.map(([label,n,caption])=>`<div class="barRow"><span>${label}</span><i><b style="--width:${Math.min(100,Math.abs(n)/max*100)}%"></b></i><output>${caption}</output></div>`).join("");
  }
  function rowIndex() {return Math.max(0,Math.min(s.index,rows.length-1));}
  function daily() {
    return [...new Set(rows.map(r=>r.date))].sort().map(date=>{
      const values=rows.filter(r=>r.date===date).map(r=>r.prediction).sort((a,b)=>a-b);
      const mid=Math.floor(values.length/2);
      return [date,values.length,f((values[mid]+values[Math.floor((values.length-1)/2)])/2,2)];
    });
  }
  function details() {
    const r=rows[rowIndex()];
    if(key==="levels")return bars(classes.levels.map((v,i)=>[v,r["prob_"+i],percent(r["prob_"+i])]));
    if(key==="change")return table(["日期","重复","预测中位数"],daily());
    return table(["该窗特征","数值"],[
      ["log RMS",f(r.log_rms,3)],["log MAV",f(r.log_mav,3)],["log 平均 WL",f(r.log_mean_waveform_length,3)],
      ["平均频率",f(r.mean_frequency_hz,1)+" Hz"],["中位频率",f(r.median_frequency_hz,1)+" Hz"]]);
  }
  function evidence(step) {
    const m=data.metrics,r=rows[rowIndex()];
    if(step===2) {
      const metric=key==="change"?"mae":key==="levels"?"macro_f1":"validation_macro_f1";
      const ordered=[...data.validation].sort((a,b)=>(a[metric]-b[metric])*(key==="change"?1:-1));
      return table(["候选与配置",key==="change"?"验证 MAE ↓":"验证 macro-F1 ↑"],ordered.map(v=>[
        key==="change"?"岭回归 · α "+v.alpha:key==="levels"?`${v.model} · ${v.features} · ${v.scale} · ${v.parameter} · ${v.smoothing} 窗`:`${v.kind} · ${v.parameter} · ${v.subset}`,f(v[metric],3)]));
    }
    if(step===4) {
      if(key==="change")return table(["模型","变化 MAE","方向正确"],[["十路融合","ten"],["仅 FORCE_8","force8"],["始终报零","zero"]].map(([n,k])=>[n,f(m.test[k].mae),m.test[k].direction_correct+" / "+m.test[k].pairs]));
      const comparison=key==="fatigue"?[["肌电模型",m.test],["时间基线",m.elapsed_time_baseline],["多数类",m.majority_baseline]]:
        [["三档分类",m.test.model],["回归后分档",m.test.regression_to_levels],["多数类",m.test.majority]];
      return table(["方案","准确率","macro-F1"],comparison.map(([n,v])=>[n,percent(v.accuracy),f(v.macro_f1,3)]));
    }
    if(step===3 && key==="fatigue"){
      const values=classes.fatigue.map((name,i)=>[name,r["score_"+i],f(r["score_"+i],3)]);
      return "<h4>三类线性得分</h4>"+bars(values,Math.max(...values.map(v=>Math.abs(v[1])),.001))+`<p class="demoDetail">最高分对应 ${classes.fatigue[r.predicted_label]}。</p>`;
    }
    if(step===3 && key==="change")return details()+`<p class="demoDetail">复测 − 基线</p><strong class="demoValue">${signed(pair.prediction)}</strong> <small>putEMG 原始刻度</small>`;
    if(step===3)return details();
    if(key==="levels" && step===1 && window.forceData?.metrics.curve_example.record===m.curve_example.record){
      const i=window.forceData.time.findIndex(t=>Math.abs(t-r.time_s)<.001);
      if(i>=0)return table(["EMG_1","窗口特征","静息参照","对数差"],["RMS","MAV","WL"].map((n,j)=>{
        const v=window.forceData.features[i][j],rest=window.forceData.rest[j];
        return [n,f(v,4),f(rest,4),f(Math.log(Math.max(v,1e-6))-Math.log(Math.max(rest,1e-6)),3)];
      }))+'<p class="demoDetail">以第一路肌电为例，其余 23 路使用相同的特征计算。</p>';
    }
    if(key==="levels")return step===0?table(["实测/MVC","训练标签"],[["< 12.5%","静息"],["12.5% 至 < 37.5%","轻"],["≥ 37.5%","重"],[percent(r.relative_force),classes.levels[r.label]]]):
      table(["数据集","受试者","肌电窗口"],Object.entries(m.subjects).map(([group,ids])=>[group,ids.length,m.label_share[group].reduce((a,b)=>a+b,0).toLocaleString()]));
    if(key==="fatigue")return step===0?table(["集合","参与者","纯标签窗"],Object.entries(m.split_counts).map(([k,v])=>[k,v.subjects.length,v.windows])):
      table(["当前测试窗","值"],[["起点",f(r.start_seconds)+" s"],["终点",f(r.end_seconds)+" s"],["参与者评分",classes.fatigue[r.label]]]);
    return step===0?table(["选中记录","保持段"],[[r.date,`${f(r.start_s)}–${f(r.end_s)} s`],[r.protocol,r.patient+" 号参与者"]]):
      table(["通道","保持段中位数","岭回归权重","加权贡献"],data.weights.map((w,i)=>{
        const value=r[`FORCE_${i+1}_pred`];return ["FORCE_"+(i+1),f(value),f(w,3),signed(w*value)];
      }))+`<p class="demoDetail">十项贡献之和 + 截距 ${f(data.intercept)} → 本次力量 ${f(r.prediction)}。</p>`;
  }
  function renderScene() {
    const steps=scenes[key], item=steps[s.step];
    $("sceneTitle").textContent=item[1]; $("sceneCopy").textContent=item[2]; $("sceneSrc").textContent=item[3];
    $("modelEvidence").innerHTML=evidence(s.step);
    // 训练选型与留出评估两步嵌入与训练过程视图相同的动画图，表格保留在下方。
    if([2,4].includes(s.step)&&window.TrainingView){
      const holder=document.createElement("div");holder.className="devChart";
      $("modelEvidence").prepend(holder);
      window.TrainingView.chart(holder,key,s.step===2?"search":"heldout");
    }
    if([0,1,3].includes(s.step)){
      const label=document.createElement("label");
      label.className="exampleScrub";
      const unit=key==="change"?"动作":"窗口";
      label.innerHTML=`<span>${unit} <output>${rowIndex()+1} / ${rows.length}</output></span><input type="range" min="0" max="${rows.length-1}" value="${rowIndex()}" aria-label="开发案例${unit}">`;
      $("sceneSrc").after(label);
      label.querySelector("input").oninput=e=>{
        s.index=Number(e.target.value);
        label.querySelector("output").textContent=(rowIndex()+1)+" / "+rows.length;
        $("modelEvidence").innerHTML=evidence(s.step);
      };
    }
    $("sceneCount").textContent=(s.step+1)+" / "+steps.length;
    $("stageBack").disabled=s.step===0;$("stageForward").disabled=s.step===steps.length-1;
    [...$("modelStages").children].forEach((b,i)=>i===s.step?b.setAttribute("aria-current","step"):b.removeAttribute("aria-current"));
  }
  // 档位类任务画成上下两条色带（实测 / 预测），两日变化仍画折线。
  const ramp=["#344052","#4f8f7c","#b3f690"], truthColor="#8b96a8", modelColor="#b3f690";
  function drawChart() {
    const longitudinal=key==="change", fatigue=key==="fatigue", svg=$("modelPlot");
    const time=rows.map((r,i)=>longitudinal?i+1:fatigue?r.start_seconds:r.time_s);
    const x=i=>52+620*(time[i]-time[0])/(time.at(-1)-time[0]||1);
    const axis=`<text x="52" y="194">${f(time[0],longitudinal?0:1)} ${longitudinal?"次":"s"}</text><text x="672" y="194" text-anchor="end">${f(time.at(-1),longitudinal?0:1)} ${longitudinal?"次":"s"}</text>`;
    if(!longitudinal){
      const pred=r=>fatigue?r.predicted_label:r.prediction;
      // 合并相邻同档窗口为一段，窗口右端取下一窗起点；疲劳记录的空档保持空白。
      const track=(value,top,height)=>{
        let out="",start=0;
        for(let i=1;i<=rows.length;i++){
          const gap=fatigue&&i<rows.length&&rows[i].start_seconds>rows[i-1].end_seconds+1e-6;
          if(i===rows.length||gap||value(rows[i])!==value(rows[start])){
            const x0=x(start), end=i<rows.length&&!gap?x(i):fatigue?52+620*(rows[i-1].end_seconds-time[0])/(time.at(-1)-time[0]||1):672;
            out+=`<rect x="${x0.toFixed(1)}" y="${top}" width="${Math.max(.6,end-x0).toFixed(1)}" height="${height}" fill="${ramp[value(rows[start])]}"/>`;
            start=i;
          }
        }
        return out;
      };
      const agree=rows.map((r,i)=>r.label===pred(r)?"":`<rect x="${x(i).toFixed(1)}" y="80" width="${Math.max(.6,(i+1<rows.length?x(i+1):672)-x(i)).toFixed(1)}" height="6" fill="#d95926"/>`).join("");
      svg.innerHTML=`<text x="44" y="46" text-anchor="end">${fatigue?"评分":"实测"}</text><text x="44" y="126" text-anchor="end">预测</text>`+
        track(r=>r.label,22,48)+track(pred,102,48)+`<text x="44" y="86" text-anchor="end" fill="#d95926">不一致</text>`+agree+
        `<line id="modelCursor" y1="14" y2="158" stroke="white" stroke-width="1.5"/>`+axis;
      return x;
    }
    const series=rows.map(r=>[r.reference_raw,r.prediction]);
    const values=series.flat(),lo=Math.min(...values),hi=Math.max(...values);
    const y=v=>165-132*(v-lo)/(hi-lo||1);
    const ticks=[lo,(lo+hi)/2,hi].map(v=>`<path d="M50 ${y(v)}h622" stroke="#333c47"/><text x="43" y="${y(v)+4}" text-anchor="end">${f(v,0)}</text>`).join("");
    const lines=[truthColor,modelColor].map((color,j)=>{
      const path=series.map((p,i)=>(i===0||rows[i-1].date!==rows[i].date?"M":"L")+`${x(i)},${y(p[j])}`).join("");
      const dots=series.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p[j])}" r="3" fill="${color}"/>`).join("");
      return `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`+dots;
    }).join("");
    const boundary=rows.findIndex(r=>r.date!==rows[0].date);
    const dates=boundary>0?`<path d="M${(x(boundary-1)+x(boundary))/2} 20v148" stroke="#72808f" stroke-dasharray="2 5"/><text x="52" y="16">基线 · ${rows[0].date}</text><text x="670" y="16" text-anchor="end">复测 · ${rows.at(-1).date}</text>`:"";
    svg.innerHTML=ticks+dates+lines+`<line id="modelCursor" y1="19" y2="171" stroke="white" stroke-dasharray="3 3"/>`+axis;
    return x;
  }
  function stop() {clearInterval(timer);timer=0;if($("modelPlay"))$("modelPlay").textContent="▶ 播放";}
  function renderCase(){
    const x=drawChart();
    function update(){
      const i=rowIndex(),r=rows[i],longitudinal=key==="change";
      $("modelCursor").setAttribute("x1",x(i));$("modelCursor").setAttribute("x2",x(i));
      $("modelSeek").value=i;
      $("sampleClock").textContent=longitudinal?`动作 ${i+1} / ${rows.length}`:f(r.time_s??r.start_seconds,1)+" s";
      $("modelValue").textContent=longitudinal?signed(pair.prediction):classes[key][r.predicted_label??r.prediction];
      $("caseNote").textContent=longitudinal?`putEMG 原始刻度 · 实测变化 ${signed(pair.reference_raw)}。当前动作 ${r.date}，估计 ${f(r.prediction)}，实测 ${f(r.reference_raw)}。`:
        key==="levels"?`实测档位 ${classes.levels[r.label]} · 力量 / MVC ${percent(r.relative_force)}`:`参与者评分 ${classes.fatigue[r.label]} · 窗口 ${f(r.start_seconds,1)}–${f(r.end_seconds,1)} s`;
      $("caseEvidence").innerHTML=details();
    }
    $("modelSeek").max=rows.length-1;
    $("modelSeek").oninput=e=>{stop();s.index=Number(e.target.value);update();};
    $("modelPlay").onclick=()=>{
      if(timer)return stop();
      if(rowIndex()===rows.length-1)s.index=0;
      $("modelPlay").textContent="Ⅱ 暂停";update();
      timer=setInterval(()=>{s.index++;update();if(s.index>=rows.length-1)stop();},key==="levels"?100:680);
    };
    update();
  }
  function render(){
    stop();
    const route=modelRoute();key=route.model;
    document.querySelectorAll("[data-model]").forEach(a=>{
      const name=a.dataset.model;a.href="#"+name+"/"+route.view;
      if(name===key)a.setAttribute("aria-current","true");else a.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-view]").forEach(a=>{
      a.href="#"+key+"/"+a.dataset.view;
      if(a.dataset.view===route.view)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current");
    });
    const source=document.getElementById("sourceMode");
    if(source)source.textContent=source.textContent.split(" / ")[0]+" / "+({replay:"离线回放",training:"训练过程",development:"开发过程"})[route.view];
    if($("error"))$("error").hidden=key!=="continuous"||Boolean(window.forceData);
    host.hidden=key==="continuous"||route.training;if(host.hidden)return;
    if(!demos?.[key]){host.textContent="缺少实验快照，请重新运行 build_pages.py。";return;}
    data=demos[key];s=state[key]??={index:0,step:0,case:0};
    const cases=key==="levels"?["测试用户 "+data.metrics.curve_example.subject]:key==="fatigue"?
      [...new Set(data.rows.map(r=>r.subject_id))]:data.pairs.map(r=>r.patient+" · "+r.protocol);
    pair=data.pairs?.[s.case];
    rows=key==="levels"?data.rows:key==="fatigue"?data.rows.filter(r=>r.subject_id===cases[s.case]):
      data.rows.filter(r=>r.patient===pair.patient&&r.protocol===pair.protocol).sort((a,b)=>a.date.localeCompare(b.date)||a.start_s-b.start_s);
    const picker=cases.length>1?`<label>测试案例 <select id="modelCase">${cases.map((c,i)=>`<option value="${i}">${key==="fatigue"?"用户 ":""}${c}</option>`).join("")}</select></label>`:`<small>${cases[0]}</small>`;
    if(route.development){
      host.innerHTML=`<section class="modelCard labReveal"><div class="caseTools"><div><p class="eyebrow">DEVELOPMENT</p><h2>${names[key]}的开发过程</h2></div>${picker}</div><nav class="demoNav" id="modelStages" aria-label="开发阶段"></nav><div class="demoScene"><div><h3 id="sceneTitle"></h3><p class="demoDetail" id="sceneCopy"></p><pre id="sceneSrc"></pre></div><div id="modelEvidence"></div></div><div class="sceneControls"><button id="stageBack">← 上一步</button><output id="sceneCount"></output><button id="stageForward">下一步 →</button></div></section>`;
      function show(i){s.step=i;host.querySelector(".exampleScrub")?.remove();renderScene();if(!matchMedia("(prefers-reduced-motion: reduce)").matches)$("modelEvidence").animate([{opacity:.2,transform:"translateY(4px)"},{opacity:1,transform:"none"}],220);}
      scenes[key].forEach((item,i)=>{const b=document.createElement("button");b.textContent=item[0];b.onclick=()=>show(i);$("modelStages").append(b);});
      $("stageBack").onclick=()=>show(s.step-1);$("stageForward").onclick=()=>show(s.step+1);
      renderScene();
    }else{
      const change=key==="change";
      host.innerHTML=`<section class="caseGrid labReveal"><article class="modelCard"><div class="caseTools"><div><p class="eyebrow">TEST RECORDING</p><h2>${change?"两日重复动作":names[key]+"时间线"}</h2></div>${picker}</div>${change?'<div class="legend"><span>实测</span><span>模型预测</span></div>':`<div class="bandLegend">${classes[key].map((c,i)=>`<span><i style="background:${ramp[i]}"></i>${c}</span>`).join("")}<span><i style="background:#d95926;height:4px"></i>预测与${key==="fatigue"?"评分":"实测"}不一致</span></div>`}<svg class="casePlot" id="modelPlot" viewBox="0 0 690 202" role="img" aria-label="实测与预测对照"></svg><div class="transport"><button id="modelPlay">▶ 播放</button><label class="slider"><input type="range" id="modelSeek" aria-label="案例位置" min="0" step="1"></label><output id="sampleClock"></output></div></article><article class="modelCard"><p class="eyebrow">MODEL OUTPUT</p><h2>${change?"相对基线的力量变化":"当前窗口的预测"}</h2><div class="demoValue" id="modelValue"></div><p class="demoDetail" id="caseNote"></p><div id="caseEvidence"></div></article></section>`;
      renderCase();
    }
    if($("modelCase")){
      $("modelCase").value=s.case;
      $("modelCase").onchange=e=>{s.case=Number(e.target.value);s.index=0;render();$("modelCase").focus();};
    }
  }
  window.addEventListener("hashchange",render);
  window.addEventListener("pagehide",stop);
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
  render();
})();
