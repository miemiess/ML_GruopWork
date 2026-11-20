"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const data = window.forceData;
  if (!data) { $("error").hidden = false; return; }
  const m = data.metrics, names = m.targets;
  const fixed = (x, n = 2) => Number(x).toFixed(n);
  const tile = (label, value, caption = "") =>
    `<div class="tile"><small>${label}</small><strong>${value}</strong><small>${caption}</small></div>`;
  const tiles = items => '<div class="tiles">' + items.join("") + '</div>';
  const table = (headers, rows) => '<div class="scrollTable"><table><thead><tr>' +
    headers.map(h => `<th>${h}</th>`).join("") + '</tr></thead><tbody>' +
    rows.map(row => '<tr>' + row.map(v => `<td>${v}</td>`).join("") + '</tr>').join("") + '</tbody></table></div>';
  let target = 6, index = 0, stage = 0, timer = 0, scale = m.selected.scale;
  let lo, hi;
  const last = data.time.length - 1;
  // 展示第四组出力阶段，保留完整记录供拖动回看。
  const initialWindow = data.time.findIndex(t => t >= 220);
  index = Math.max(0, initialWindow);
  const x = i => 46 + 581 * (data.time[i] - data.time[0]) / (data.time[last] - data.time[0]);
  const y = value => 192 - 174 * (value - lo) / (hi - lo);
  const rms = data.features.map(row => row[0]);
  const rmsPeak = Math.max(...rms) || 1;
  const rmsPath = rms.map((value,i) => (i ? "L" : "M") + fixed(x(i),1) + " " + fixed(58-50*value/rmsPeak,1)).join("");
  $("inputTrace").innerHTML = `<path d="${rmsPath}" stroke="#9ae3cb" fill="none"/><line id="inputCursor" y1="4" y2="60" stroke="#fff"/>`;
  $("seek").max = last;
  $("caseInfo").textContent = `测试用户 ${m.curve_example.subject} · ${m.trial} · 每点为一个窗口的中心时刻`;
  names.forEach((name, j) => {
    $("target").add(new Option(name, j));
    const b = document.createElement("button");
    b.innerHTML = name + "<strong></strong>";
    b.onclick = () => selectTarget(j);
    $("channels").append(b);
  });
  function selectTarget(j) {
    target = j;
    $("target").value = String(j);
    const truth = data.truth.map(r => r[j]), pred = data.prediction.map(r => r[j]);
    lo = Math.min(...truth, ...pred, data.base[j]);
    hi = Math.max(...truth, ...pred, data.base[j], lo + 1);
    const path = series => series.map((v,i) => (i ? "L" : "M") + fixed(x(i),1) + " " + fixed(y(v),1)).join("");
    const grid = [lo, (lo+hi)/2, hi].map(v =>
      `<path d="M46 ${y(v)}h581" stroke="#343b48"/><text x="39" y="${y(v)+4}" text-anchor="end">${fixed(v,0)}</text>`).join("");
    $("plot").innerHTML = grid + `<text x="46" y="217">${fixed(data.time[0],1)} s</text><text x="627" y="217" text-anchor="end">${fixed(data.time[last],1)} s</text>
      <path d="M46 ${y(data.base[j])}h581" stroke="#a7afbc" stroke-dasharray="5 5"/>
      <path d="${path(truth)}" fill="none" stroke="#8b96a8" stroke-width="1.4"/>
      <path d="${path(pred)}" fill="none" stroke="#b3f690" stroke-width="1.7"/>
      <line id="cursor" y1="12" y2="192" stroke="#fff" opacity=".7"/><circle id="dot" r="4" fill="#b3f690"/>`;
    refreshWindow();
  }
  function refreshWindow() {
    $("seek").value = index;
    $("clock").textContent = fixed(data.time[index],1) + " s";
    $("cursor").setAttribute("x1",x(index)); $("cursor").setAttribute("x2",x(index));
    $("inputCursor").setAttribute("x1",x(index)); $("inputCursor").setAttribute("x2",x(index));
    const p = data.prediction[index][target], t = data.truth[index][target];
    $("dot").setAttribute("cx",x(index)); $("dot").setAttribute("cy",y(p));
    [...$("channels").children].forEach((b,j) => {
      b.setAttribute("aria-pressed",j === target);
      b.lastChild.textContent = fixed(data.prediction[index][j],1);
    });
    $("difference").textContent = `${names[target]}　实测 ${fixed(t)}　预测 ${fixed(p)}　绝对误差 ${fixed(Math.abs(t-p))}`;
    $("liveSteps").innerHTML = tile("① EMG_1 · 窗口 RMS",fixed(data.features[index][0],3),"24 通道 × 三项特征") +
      tile("② EMG_1 · RMS 标准分",fixed(data.z[index][0],3),"相对静息取对数，再标准化") +
      tile("③ 森林直接输出",fixed(data.raw[index][target]),names[target] + " · 100 棵树取平均") +
      tile("④ 因果平滑",fixed(p),`本记录最近 ${Math.min(index+1,m.selected.smoothing)} 窗平均`);
  }
  function stop() {clearInterval(timer); timer = 0; $("play").textContent = "▶ 播放";}
  $("play").onclick = () => {
    if (timer) return stop();
    if (index === last) index = 0;
    refreshWindow();
    $("play").textContent = "Ⅱ 暂停";
    timer = setInterval(() => {index++; refreshWindow(); if (index === last) stop();},100);
  };
  $("target").onchange = e => selectTarget(Number(e.target.value));
  $("seek").oninput = e => {stop(); index = Number(e.target.value); refreshWindow();};
  const stages = [
    ["用户划分","先把人分开，再比较模型","同一位参与者两天的记录进入同一组。训练用户用于拟合，验证用户用于选型，测试用户留到最后评估。","split_subject_ids(subjects, seed=42)","experiment.py · prepare_data"],
    ["校准分窗","先给力传感器调零","检查前四秒的出力指令均为零，保存各力量通道的静息中位数。肌电整段经过因果带通，再在同一记录内滑窗。校准窗从训练与评估中排除。","target = mean(force_window) − median(rest_force)","preprocessing.py · calibration_is_rest / record_features"],
    ["特征矩阵","把一个肌电窗口压缩成 72 个数","每路提取 RMS、平均绝对值 MAV、波形长度 WL。获选方案对各特征取对数并减去本记录静息对数特征，随后用训练集统计量标准化。下表跟随案例当前窗口。","z = scaler.transform(log(feature) − log(rest_feature))","model.py · transform / ForceModel.predict"],
    ["训练选型","让验证误差决定模型","岭回归学习各特征的加权和，随机森林对多棵回归树的十维输出取平均。切换尺度可比较同一批验证用户上的候选，按十路平均 RMSE 从低到高排列。","choice = validation.loc[validation.mean_rmse.idxmin()]","experiment.py · model_candidates / fit_predict / select_model"],
    ["连续预测","把最近五窗连成一个输出","每个窗口先输出十路连续值，再在记录内部对当前及之前的预测取平均。记录开头按已有窗数计算。下表可对照回放中所选通道和时刻。","force[t] = mean(raw[max(0, t−4) : t+1])","model.py · ForceModel.predict / smooth"],
    ["留出评估","新用户的力量能预测到什么程度","均值基线始终输出训练目标的平均值。RMSE 衡量预测偏差，越低越好。R² 衡量相对测试数据自身均值的解释能力，负值表示误差超过该参照。","target_metrics(y_true, y_pred) → MAE, RMSE, R², Pearson r","targets.py · target_metrics"]
  ];
  function windowDiagram() {
    return `<figure class="diagram"><svg viewBox="0 0 520 94" role="img" aria-label="三个相邻肌电窗口，每个覆盖二百毫秒，并与下一窗重叠一半">
      <g fill="#9ae3cb"><rect x="16" y="9" width="240" height="16" rx="4"/>
      <rect x="136" y="34" width="240" height="16" rx="4" opacity=".8"/>
      <rect x="256" y="59" width="240" height="16" rx="4" opacity=".6"/></g></svg>
      <figcaption>校准结束后，每隔 100 ms 开始下一窗。相邻窗口共享 512 个采样点。</figcaption></figure>`;
  }
  function smoothingDiagram() {
    const begin = Math.max(0,index-49), rows = data.raw.slice(begin,index+1);
    const direct = rows.map(r=>r[target]), averaged = data.prediction.slice(begin,index+1).map(r=>r[target]);
    const min = Math.min(...direct,...averaged), range = Math.max(...direct,...averaged)-min || 1;
    const points = values => values.map((v,j)=>`${10+j*500/Math.max(1,rows.length-1)},${84-(v-min)*72/range}`).join(" ");
    return `<figure class="diagram"><svg viewBox="0 0 520 96" role="img" aria-label="同一段记录平滑前后的力量预测">
      <polyline fill="none" stroke="#8b96a8" stroke-width="2" points="${points(direct)}"/>
      <polyline fill="none" stroke="#b3f690" stroke-width="3" points="${points(averaged)}"/></svg>
      <figcaption>${names[target]} · ${fixed(data.time[begin],1)}–${fixed(data.time[index],1)} s　灰色为森林直接输出，薄荷绿为五窗平滑结果。</figcaption></figure>`;
  }
  function renderStage() {
    const s = stages[stage], ev = $("evidence");
    $("stageTitle").textContent = s[1]; $("story").textContent = s[2];
    $("code").textContent = s[3]; $("source").textContent = "关键步骤示意 · src/emginsight/putemg_force/" + s[4];
    $("counter").textContent = `${stage+1} / ${stages.length}`;
    $("prev").disabled = stage === 0; $("next").disabled = stage === stages.length-1;
    [...$("stages").children].forEach((b,j) => b.setAttribute("aria-pressed",j===stage));
    if (stage === 0) {
      ev.innerHTML = tiles(["train","validation","test"].map((g,i) => tile(["训练","验证","测试"][i],
        m.subjects[g].length+" 人",m.records[g]+" 条记录 · "+m.windows[g].toLocaleString()+" 窗")));
    } else if (stage === 1) {
      ev.innerHTML = tiles([tile("因果带通",m.window.filter_hz.join("–")+" Hz","名义采样率 5120 Hz"),
        tile("窗长 / 步长","1024 / 512","约 200 / 100 ms"),tile(names[target]+" · 静息零点",fixed(data.tare[target]),"案例记录的前四秒")]) + windowDiagram();
    } else if (stage === 2) {
      ev.innerHTML = table(["EMG_1","当前窗口","静息特征","标准分"],["RMS","MAV","WL"].map((f,j) =>
        [f,fixed(data.features[index][j],4),fixed(data.rest[j],4),fixed(data.z[index][j],3)]));
    } else if (stage === 3) {
      const winner = m.selected;
      ev.innerHTML = '<div id="devChart" class="devChart"></div>' + tiles([tile("全局最低验证 RMSE",fixed(winner.mean_rmse),
        winner.scale+" · "+winner.features+" · "+winner.parameter+" · "+winner.smoothing+" 窗")]) +
        '<label>比较特征尺度 <select id="scale"><option>raw</option><option>log</option><option>log_rest</option></select></label>' +
        table(["模型","特征","参数","平滑窗","验证 RMSE"],data.validation.filter(r=>r.scale===scale)
          .sort((a,b)=>a.mean_rmse-b.mean_rmse).map(r=>[r.model==="ridge"?"岭回归":"随机森林",r.features,r.parameter,r.smoothing,fixed(r.mean_rmse)]));
      window.TrainingView?.chart($("devChart"), "continuous", "search");
      $("scale").value = scale; $("scale").onchange = e => {scale=e.target.value; renderStage(); $("scale").focus();};
    } else if (stage === 4) {
      const start = Math.max(0,index-m.selected.smoothing+1);
      ev.innerHTML = smoothingDiagram() + table(["窗口中心","通道","森林直接输出"],data.raw.slice(start,index+1)
        .map((r,j)=>[fixed(data.time[start+j],2)+" s",names[target],fixed(r[target])])) +
        tiles([tile("平滑后预测",fixed(data.prediction[index][target])),tile("同期实测",fixed(data.truth[index][target]))]);
    } else {
      ev.innerHTML = tiles([tile("测试平均 RMSE",fixed(m.test.model.rmse),"基线 "+fixed(m.test.baseline.rmse)),
        tile("测试平均 MAE",fixed(m.test.model.mae),"基线 "+fixed(m.test.baseline.mae)),
        tile("测试平均 R²",fixed(m.test.model.r2,3),"基线 "+fixed(m.test.baseline.r2,3))]) +
        '<div id="devChart" class="devChart"></div><label>查看结果 <select id="by"><option>按力量通道</option><option>按测试用户</option></select></label><div id="scores"></div>';
      const scores = () => {
        const byUser = $("by").selectedIndex === 1;
        $("scores").innerHTML = table([byUser?"用户":"目标","模型 RMSE","基线 RMSE","模型 R²"],data[byUser?"per_user":"per_target"]
          .map(r=>[byUser?r.subject:r.target,fixed(r.rmse_model),fixed(r.rmse_baseline),fixed(r.r2_model,3)]));
      };
      $("by").onchange=scores; scores();
      window.TrainingView?.chart($("devChart"), "continuous", "users");
    }
  }
  function setStage(i) {
    stage = i; renderStage();
    $("scene").classList.remove("enter"); void $("scene").offsetWidth; $("scene").classList.add("enter");
  }
  stages.forEach((s,j) => {const b=document.createElement("button"); b.textContent=s[0]; b.onclick=()=>setStage(j); $("stages").append(b);});
  $("prev").onclick=()=>setStage(stage-1); $("next").onclick=()=>setStage(stage+1);
  function view() {
    const {model,development:dev,training}=modelRoute(), continuous=model==="continuous"&&!training;
    stop(); $("development").hidden=!continuous||!dev; $("replay").hidden=!continuous||dev; $("pipeline").hidden=!continuous||dev;
    if (continuous && dev) renderStage();
  }
  selectTarget(target); view(); window.addEventListener("hashchange",view);
})();
