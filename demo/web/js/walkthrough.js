"use strict";

const FLOW_STAGES = [
  {
    title: "从连续信号取一个窗口",
    code: "windows = window_gesture_record(\n    record, window_size=200,\n    step_size=100, max_gap_ms=20\n)",
    source: "gesture/windowing.py",
    knowledge: "相邻窗口共享一半采样点。因此每新增 100 点，就能形成一个新的 200 点窗口。离线评估保留纯标签窗口，并先按用户划分数据，避免同一受试者同时进入训练和测试。"
  },
  {
    title: "把波形压缩成 24 个特征",
    code: "rms = sqrt(mean(window ** 2))\nmav = mean(abs(window))\nwl  = sum(abs(diff(window)))",
    source: "features/time_domain.py",
    knowledge: "RMS 和 MAV 描述信号幅度，RMS 对较大的尖峰更敏感。WL 累加相邻采样点的变化量。同样的三个计算应用到八个通道，再按通道拼成一个向量。"
  },
  {
    title: "用训练集的刻度转换",
    code: "scaler = Standardizer().fit(X_train)\nz = scaler.transform(X_window)",
    source: "models/standardization.py",
    knowledge: "每一列减去训练集均值，再除以训练集标准差。标准化为正表示高于训练均值，为负表示低于训练均值。测试窗口只使用训练时保存的参数。"
  },
  {
    title: "从六类概率得到手势",
    code: "probs = forest.predict_proba(z)\ngesture = forest.classes_[argmax(probs)]",
    source: "gesture/experiment.py",
    knowledge: "每棵树沿着特征阈值走到一个叶子，得到一组类别概率。各树的概率取平均，最大的一类就是预测动作。树数和特征组合依据验证集表现选定。"
  }
];

function createWalkthrough(pause, channel) {
  const panel = $("caseFlow");
  const buttons = [...$("flowSteps").children];
  const launch = $("walkWindow");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stageDuration = 850;
  let frame = null, selected = 3, mode = "follow", animation = 0;
  let displayedStage = -1;

  function showStage() {
    const stage = FLOW_STAGES[selected];
    const changed = displayedStage !== selected;
    if (changed) {
      displayedStage = selected;
      setText($("flowStageTitle"), stage.title);
      setText($("flowCode"), stage.code);
      setText($("flowKnowledge"), stage.knowledge);
      const sourceLink = $("flowSource");
      sourceLink.href = sourceLink.dataset.sourceBase + stage.source;
      $("flowChannelControl").hidden = selected !== 1 && selected !== 2;
      buttons.forEach((button,i) => button.setAttribute("aria-pressed", i === selected));
    }
    const evidence = $("flowEvidence");
    showEvidence(evidence, selected, frame, Number(channel.value));
    if (changed && !reducedMotion) {
      evidence.animate([{opacity: .55, transform: "translateY(5px)"}, {opacity: 1, transform: "none"}], 190);
    }
  }

  function mark(step, progress = 1) {
    buttons.forEach((button, i) => {
      const state = i === step ? "active" : i < step ? "done" : "waiting";
      if (button.dataset.state !== state) {
        button.dataset.state = state;
        if (i === step) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      }
    });
    panel.style.setProperty("--flow-progress", reducedMotion ? 1 : progress);
  }

  function reset() {
    cancelAnimationFrame(animation);
    mode = "follow";
    setText(launch, "▶ 逐步演示");
  }

  buttons.forEach((button, i) => button.onclick = () => {
    pause();
    reset();
    mode = "inspect";
    selected = i;
    mark(i);
    showStage();
    setText($("flowStatus"), "查看当前窗口");
  });

  launch.onclick = () => {
    if (!frame) return;
    if (mode === "running") {
      cancelAnimationFrame(animation);
      mode = "inspect";
      setText(launch, "▶ 重新演示");
      setText($("flowStatus"), "逐步演示已暂停");
      return;
    }
    pause();
    mode = "running";
    setText(launch, "Ⅱ 暂停演示");
    setText($("flowStatus"), "固定窗口 · 逐步演示");
    const start = performance.now();
    function tick(now) {
      const elapsed = Math.max(0, now - start);
      const next = Math.min(FLOW_STAGES.length - 1, Math.floor(elapsed / stageDuration));
      if (selected !== next || elapsed === 0) {
        selected = next;
        showStage();
      }
      mark(next, Math.min(1, elapsed / stageDuration - next));
      if (elapsed < FLOW_STAGES.length * stageDuration) animation = requestAnimationFrame(tick);
      else {
        mode = "inspect";
        setText(launch, "▶ 重新演示");
        setText($("flowStatus"), "逐步演示完成");
      }
    }
    tick(start);
  };

  return {
    reset,
    update(current, step, playing, refresh) {
      frame = current;
      launch.disabled = !frame;
      if (mode === "follow") {
        mark(step);
        // 播放时保留阅读面板，左侧单独跟随实际回放高亮。
        if (!playing) selected = frame ? 3 : 0;
        setText($("flowStatus"), frame ? (playing ? "跟随回放" : "窗口已就绪") : "等待有效窗口");
      }
      if (refresh || displayedStage !== selected) showStage();
    }
  };
}

function showEvidence(container, stage, frame, channel) {
  if (!frame) {
    container.innerHTML = '<p class="flowEmpty">等待一个已完成的有效窗口。可以拖动波形或切换观察案例。</p>';
    return;
  }
  const seconds = time => (time / 1000).toFixed(3) + " s";
  if (stage === 0) {
    container.innerHTML = `
      <div class="windowRail" aria-label="当前 200 点窗口与后续窗口重叠 100 点">
        <span class="currentSpan">当前窗 · 200 点</span><span class="nextSpan">下一候选窗</span>
      </div>
      <small>${seconds(frame.a)} → ${seconds(frame.b)} · 起点 #${frame.start} · 重叠 100 点</small>`;
    return;
  }
  if (stage === 1 || stage === 2) {
    const metrics = ["RMS", "MAV", "WL"];
    container.innerHTML = '<div class="flowMetrics">' + metrics.map((name,i)=>{
      const index = channel * 3 + i;
      const raw = frame.f[index].toPrecision(4);
      const value = stage === 1 ? raw : frame.z[index].toFixed(3);
      return `<div><small>${name}</small><strong>${value}</strong>${stage === 2 ? `<small>原值 ${raw}</small>` : ""}</div>`;
    }).join("") + "</div>";
    return;
  }
  container.innerHTML = '<div class="probabilities">' + frame.p.map((value,i)=>{
    const percent = value * 100;
    return `<div class="probability" data-winner="${i+1===frame.pred}"><span>${labels[i+1]}</span>
      <meter min="0" max="1" value="${value}" aria-label="${labels[i+1]} 概率"></meter><small>${percent.toFixed(1)}%</small></div>`;
  }).join("") + `</div><small>预测 ${labels[frame.pred]} · UCI 标注 ${labels[frame.label]}</small>`;
}
