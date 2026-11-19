"use strict";
const $ = id => document.getElementById(id);
const labels = ["未标注","静息","握拳","腕屈","腕伸","桡偏","尺偏","手掌伸展"];
const poses = [null,[0,0,.06],[0,0,1],[-1.1,0,.08],[.95,0,.05],[0,-.46,.06],[0,.56,.06],[0,0,0]];
const data = window.EMG_SESSION;
if (!data) {
  $("missing").hidden=false;
  document.querySelectorAll("button,select,input").forEach(el=>el.disabled=true);
} else init();

function init() {
  const clips=data.clips, picker=$("case"), slider=$("time"), wave=$("wave");
  const traceRows=[...$("trace").children];
  let clip,time,traceStep=-1,playing=false,speed=1,last=0,frame,background;
  let simulation=null, simId=0, simLast=0;
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  const inputHand=poseAnimator($("truthHand"), 162, reduced);
  const outputHand=poseAnimator($("hand"), 253, reduced);
  const channel=document.createElement("select");
  channel.id="channel";
  channel.setAttribute("aria-label","观察通道");
  $("caseChannel").append(channel);
  const walkthrough=createWalkthrough(stop,channel);
  clips.forEach((c,i)=>picker.add(new Option(labels[c.label]+(c.label===c.pred?" · 正确样例":" → "+labels[c.pred]+" · 误判"),i)));
  for(let i=0;i<8;i++) channel.add(new Option("CH "+String(i+1).padStart(2,"0"),i));
  const controls=$("gestureControls");
  const model=window.EMG_SIMULATOR;
  for (const label of model?.classes ?? []) {
    const button=document.createElement("button");
    button.textContent=labels[label];
    button.setAttribute("aria-pressed","false");
    button.onclick=()=>selectGesture(label);
    controls.append(button);
  }
  $("simMode").onclick=()=>switchMode(true);
  $("replayMode").onclick=()=>switchMode(false);
  channel.onchange=redraw;
  picker.onchange=()=>choose(+picker.value);
  slider.oninput=()=>{stop(); walkthrough.reset(); time=+slider.value; render(true)};
  wave.onclick=event=>{
    if(simulation)return;
    stop();
    walkthrough.reset();
    const fraction=Math.max(0,Math.min(1,(event.offsetX-46)/(wave.clientWidth-52)));
    time=clip.t[0]+fraction*(clip.t.at(-1)-clip.t[0]);render(true);
  };
  $("speed").onclick=()=>{speed=speed===1?.5:speed===.5?2:1;$("speed").textContent=speed+"×"};
  $("play").onclick=()=>{
    if(playing){stop();render();return}
    walkthrough.reset();
    if(time>=clip.t.at(-1)) time=clip.t[0];
    playing=true;last=performance.now();$("play").textContent="Ⅱ 暂停";playId=requestAnimationFrame(tick);
  };
  function stop(){playing=false;$("play").textContent="▶ 播放";cancelAnimationFrame(playId)}
  let playId=0;
  function tick(now){
    if(!playing)return;
    time=Math.min(clip.t.at(-1),time+(now-last)*speed);last=now;
    if(time>=clip.t.at(-1))stop();else playId=requestAnimationFrame(tick);
    render();
  }
  function choose(i){
    stop();clip=clips[i];picker.value=i;time=clip.focus;frame=null;
    walkthrough.reset();
    slider.min=clip.t[0];slider.max=clip.t.at(-1);$("user").textContent="测试用户 / "+clip.subject;
    cacheWave();render(true);
  }
  function selectGesture(label){
    simulation.select(label);
    [...controls.children].forEach((button,i)=>button.setAttribute("aria-pressed", model.classes[i]===label));
    comparePoses(label,frame?.pred || 0);
  }
  function switchMode(simulating){
    if(simulating===Boolean(simulation))return;
    if(simulating && !model){$("simMissing").hidden=false;return}
    $("simMissing").hidden=true;
    stop();cancelAnimationFrame(simId);
    walkthrough.reset();
    frame=null;
    document.body.classList.toggle("simulating",simulating);
    $("simMode").setAttribute("aria-pressed",simulating);
    $("replayMode").setAttribute("aria-pressed",!simulating);
    controls.hidden=!simulating;
    $("casePicker").hidden=simulating;
    $("testResults").hidden=simulating;
    $("caseFlow").hidden=simulating;
    $("compactTrace").hidden=!simulating;
    $(simulating ? "simChannel" : "caseChannel").append(channel);
    setText($("sourceMode"),simulating ? "UCI / 交互模拟" : "UCI / 离线回放");
    setText($("waveTitle"),simulating ? "模拟肌电波形" : "前臂肌电波形");
    setText($("inputTitle"),simulating ? "INPUT · 模拟手" : "INPUT · 真实动作示意");
    setText($("poseNote"),simulating ? "模拟信号由 UCI 训练窗口的模板合成，换手势时平滑过渡。" : "左侧按 UCI 标签模拟动作，未标注时显示自然姿势。右侧显示肌电窗口的分类结果。");
    setText($("traceNote"),simulating ? "浏览器按 200 点分窗，以 C 实验选出的模型持续推理。" : "高亮按回放节奏重现处理顺序，数值来自当前窗口。");
    if(simulating){
      simulation=createSimulation(model);
      simulation.advance(2);
      selectGesture(1);
      simLast=performance.now();
      redraw();simId=requestAnimationFrame(simTick);
    }else{
      simulation=null;
      choose(+picker.value);
    }
  }
  function simTick(now){
    simulation.advance(Math.min(100,now-simLast));
    simLast=now;
    renderSimulation();
    simId=requestAnimationFrame(simTick);
  }
  function redraw(){
    if(simulation)renderSimulation(true);
    else {cacheWave();render(true)}
  }
  function renderSimulation(force=false){
    clip=simulation.snapshot();
    time=clip.t.at(-1);
    const changed=frame!==simulation.frame;
    frame=simulation.frame;
    cacheWave();paintWave(frame);
    setText($("user"),"虚拟 1 kHz · "+(time/1000).toFixed(2)+" s");
    updateTrace(frame ? time-frame.b : 0, changed||force);
    comparePoses(simulation.gesture,frame?.pred || 0);
  }
  function cacheWave(){
    const w=wave.clientWidth,h=wave.clientHeight,r=Math.min(devicePixelRatio||1,2);
    if(wave.width!==Math.round(w*r))wave.width=Math.round(w*r);
    if(wave.height!==Math.round(h*r))wave.height=Math.round(h*r);
    if(!background)background=document.createElement("canvas");
    if(background.width!==wave.width)background.width=wave.width;
    if(background.height!==wave.height)background.height=wave.height;
    const g=background.getContext("2d");
    g.setTransform(1,0,0,1,0,0);g.clearRect(0,0,background.width,background.height);g.scale(r,r);
    const t0=clip.t[0],dt=clip.t.at(-1)-t0;
    const limit=simulation?.peak ?? Math.max(...clip.channels.map(c=>Math.max(...c.map(Math.abs))),1e-12);
    for(let i=0;i<=10;i++){const x=46+(w-52)*i/10;g.fillStyle="#2a313e";g.fillRect(x,8,1,h-20)}
    clip.channels.forEach((signal,ch)=>{
      const y=10+(ch+.5)*(h-22)/8;
      g.fillStyle="#353e4c";g.fillRect(46,y,w-52,1);
      g.fillStyle="#9baab6";g.font="11px Consolas,monospace";g.fillText("CH "+String(ch+1).padStart(2,"0"),0,y+4);
      g.strokeStyle=ch===+channel.value?"#d2ff9d":"#84c8be";g.lineWidth=1.15;g.beginPath();
      signal.forEach((v,i)=>{const x=46+(w-52)*(clip.t[i]-t0)/dt,py=y-v/limit*(h-22)/16;i?g.lineTo(x,py):g.moveTo(x,py)});
      g.stroke();
    });
    g.fillStyle="#a4b0c1";g.font="10px Consolas,monospace";
    for(let i=0;i<=4;i++){
      g.textAlign=i===4?"right":i===0?"left":"center";
      g.fillText(((t0+dt*i/4)/1000).toFixed(2)+" s",46+(w-52)*i/4,h-1);
    }
  }
  function paintWave(selected){
    const g=wave.getContext("2d"),r=wave.width/wave.clientWidth;
    // 缓存底图含透明区域，重画前清除旧游标和窗口高亮。
    g.clearRect(0,0,wave.width,wave.height);
    g.drawImage(background,0,0);
    const px=t=>r*(46+(wave.clientWidth-52)*(Math.max(clip.t[0],t)-clip.t[0])/(clip.t.at(-1)-clip.t[0]));
    if(selected){g.fillStyle="rgba(181,245,143,.14)";g.fillRect(px(selected.a),0,px(selected.b)-px(selected.a),wave.height)}
    g.fillStyle="#e3ffc7";g.fillRect(px(time),0,1.5*r,wave.height);
  }
  function render(force=false){
    const sample=clip.t.findLastIndex(t=>t<=time);
    const candidate=clip.frames.findLast(f=>f.b<=time);
    // 超过一个步长或离开标注段后，不再显示上一窗的结果。
    const samplesSinceWindow=candidate ? sample-clip.t.findLastIndex(t=>t<=candidate.b) : 0;
    const selected=candidate && clip.truth[sample]===candidate.label && samplesSinceWindow<100 ? candidate:null;
    paintWave(selected);
    slider.value=time;$("clock").textContent=(time/1000).toFixed(2)+" s";
    const changed=frame!==selected;frame=selected;
    updateTrace(samplesSinceWindow,changed||force);
    comparePoses(clip.truth[sample] || 0, frame?.pred || 0);
  }
  function comparePoses(truth, prediction){
    // 未标注时用静息手模作为自然姿势，标签仍为未标注。
    inputHand.show(truth || 1);
    outputHand.show(prediction);
    const comparison=$("comparison");
    const match=truth && prediction ? (truth===prediction ? "match" : "error") : "pending";
    const message=match==="match" ? "识别一致" : match==="error" ? (simulation ? "预测与当前输入不同" : "误判 · 预测与标注不同") : "";
    setText($("truth"), labels[truth]);
    setText($("name"), prediction ? labels[prediction] : "等待有效窗口");
    setText(comparison, message);
    if(comparison.dataset.state!==match)comparison.dataset.state=match;
  }
  function updateTrace(samplesSinceWindow, refreshValues){
    // 重放已完成窗口的阶段。窗口间隔以原始采样点计算。
    const active=frame ? (playing || simulation ? Math.min(3,1+Math.floor(samplesSinceWindow/30)) : 3) : 0;
    if(!simulation){walkthrough.update(frame,active,playing,refreshValues);return}
    if(traceStep!==active){
      traceStep=active;
      traceRows.forEach((row,i)=>{
        row.dataset.state=i===active ? "active" : i<active ? "done" : "waiting";
        if(i===active)row.setAttribute("aria-current","step");
        else row.removeAttribute("aria-current");
      });
    }
    if(!refreshValues)return;
    const ch=+channel.value;
    const describe=values=>["RMS","MAV","WL"].map((name,i)=>name+" "+values[ch*3+i].toPrecision(4)).join("  ·  ");
    setText($("traceWindow"),frame ? "起点 "+frame.start+" · 结束 "+(frame.b/1000).toFixed(3)+" s" : "等待有效窗口");
    setText($("traceFeatures"),frame ? describe(frame.f) : "—");
    setText($("traceScale"),frame ? describe(frame.z) : "—");
    setText($("traceResult"),frame ? labels[frame.pred]+" · 类别概率 "+(100*frame.p[frame.pred-1]).toFixed(0)+"%" : "—");
  }
  function cell(text){const s=document.createElement("span");s.textContent=text;$("matrix").append(s)}
  cell(""); labels.slice(1,1+data.matrix.length).forEach(cell);
  data.matrix.forEach((row,i)=>{
    cell(labels[i+1]); row.forEach((n,j)=>{
      const b=document.createElement("button");b.textContent=n;b.disabled=!n;
      b.title=labels[i+1]+" → "+labels[j+1]+" · "+n+" 窗";
      if(n){b.style.background=i===j?"#375347":"#655343";b.onclick=()=>choose(clips.findIndex(c=>c.label===i+1&&c.pred===j+1))}
      $("matrix").append(b);
    });
  });
  choose(clips.findIndex(c=>c.label===2&&c.pred===2));
  new ResizeObserver(redraw).observe(wave);
}

function setText(element, content){
  if(element.textContent!==content)element.textContent=content;
}

function poseAnimator(canvas, hue, reduced){
  let id=-1, pose=null, animation=0;
  const paint=()=>{if(pose)drawPose(canvas,pose,hue)};
  new ResizeObserver(paint).observe(canvas);
  return {
    show(next){
      if(id===next)return;
      id=next;
      cancelAnimationFrame(animation);
      const target=poses[id];
      canvas.style.opacity=target ? "1" : "0";
      if(!target || !pose || reduced){pose=target;paint();return}

      const from=pose, start=performance.now();
      function tick(now){
        const progress=Math.min(1,(now-start)/240);
        const eased=1-(1-progress)**3;
        pose=from.map((value,i)=>value+(target[i]-value)*eased);
        paint();
        if(progress<1)animation=requestAnimationFrame(tick);
      }
      animation=requestAnimationFrame(tick);
    }
  };
}
