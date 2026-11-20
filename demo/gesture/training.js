"use strict";
// 训练过程视图：按人划分 → 训练人数 → 候选搜索 → 留出评估 → 逐用户。
// 所有数值来自 training-data.js（由 training_export.py 从 results/ 导出），网页不重新训练。
window.TrainingView = (() => {
  const W = 760;
  const color = {
    train: "#3987e5", validation: "#d95926", test: "#199e70",
    family: ["#9085e9", "#c98500", "#d55181"],
    model: "#b3f690", base: "#8b96a8", grid: "#2b3240", surface: "#191e29",
  };
  const groupName = {train: "训练", validation: "验证", test: "测试"};
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches || /[?&]still\b/.test(location.search);
  if (reduced) document.documentElement.classList.add("tvStill");
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
  const median = v => { const s = [...v].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

  function format(item, value) {
    if (item.percent) return (value * 100).toFixed(1) + "%";
    return Number(value).toFixed(item.digits ?? 2);
  }
  // 刻度值去掉多余的零：0.250 → 0.25，60.00 → 60。
  const tick = (item, value) => item.percent ? +(value * 100).toFixed(1) + "%" : String(+Number(value).toFixed(item.digits ?? 2));
  // 纵轴一律"越往上越好"：越低越好的指标翻转坐标，刻度仍显示真实数值。
  function scale(values, from, to, better, pad = .08) {
    let lo = Math.min(...values), hi = Math.max(...values);
    const span = hi - lo || Math.abs(hi) || 1;
    lo -= span * pad; hi += span * pad;
    const map = v => better === "low" ? from + (to - from) * (v - lo) / (hi - lo) : to - (to - from) * (v - lo) / (hi - lo);
    map.ticks = nice(lo, hi, 4);
    return map;
  }
  // 取 1、2、2.5、5 × 10ⁿ 的整齐刻度。
  function nice(lo, hi, count = 5) {
    const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw || 1));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(v => v >= raw);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }
  function yAxis(y, item, x0 = 64, x1 = W - 24) {
    return y.ticks.map(t => `<path d="M${x0} ${y(t).toFixed(1)}H${x1}" class="tvGrid"/>` +
      `<text x="${x0 - 8}" y="${(y(t) + 4).toFixed(1)}" text-anchor="end" class="tvTick">${tick(item, t)}</text>`).join("");
  }

  // ① 按人划分：所有参与者先排成一池，再按编号落入三组。
  function splitStep(d) {
    const groups = ["train", "validation", "test"];
    const all = groups.flatMap(g => d.split.subjects[g].map(id => ({id, g})));
    const pool = [...all].sort((a, b) => a.id.localeCompare(b.id, "zh", {numeric: true}));
    const perRow = Math.min(16, pool.length), gap = 40;
    const left = (W - (perRow - 1) * gap) / 2;
    const start = new Map(pool.map((p, i) => [p.id, [left + (i % perRow) * gap, 34 + Math.floor(i / perRow) * 34]]));
    const colX = [130, 380, 630], cols = 5, step = 34;
    const top = 74 + Math.ceil(pool.length / perRow) * 34;
    let svg = `<text x="${W / 2}" y="16" text-anchor="middle" class="tvLabel tvPool">${pool.length} 名参与者</text>`;
    const rows = [];
    groups.forEach((g, k) => {
      const ids = d.split.subjects[g];
      svg += `<g class="tvFade" style="--d:700ms"><text x="${colX[k]}" y="${top}" text-anchor="middle" class="tvHead" fill="${color[g]}">${groupName[g]} · ${ids.length} 人</text>` +
        `<text x="${colX[k]}" y="${top + 18}" text-anchor="middle" class="tvTick">${esc(d.split.counts[g] ?? "")}</text></g>`;
      ids.forEach((id, i) => {
        const [x0, y0] = start.get(id);
        const x1 = colX[k] + ((i % cols) - (cols - 1) / 2) * step, y1 = top + 46 + Math.floor(i / cols) * step;
        svg += `<g class="tvMove" data-tip="参与者 ${esc(id)} → ${groupName[g]}集" style="--x0:${x0}px;--y0:${y0}px;--x1:${x1}px;--y1:${y1}px;--d:${500 + i * 28 + k * 90}ms">` +
          `<circle r="13" fill="${color[g]}" stroke="${color.surface}" stroke-width="2"/><text y="4" text-anchor="middle" class="tvId">${esc(id)}</text></g>`;
      });
      rows.push([groupName[g], ids.length, d.split.counts[g] ?? "", ids.join("、")]);
    });
    const height = top + 46 + Math.ceil(d.split.subjects.train.length / cols) * step + 4;
    return {
      nav: "按人划分", title: "先把人分开，再谈训练",
      lead: `${pool.length} 名参与者按编号随机分成三组：<b>${d.split.subjects.train.length} 人</b>用于拟合，<b>${d.split.subjects.validation.length} 人</b>用于选模型，<b>${d.split.subjects.test.length} 人</b>留到最后只评估一次。`,
      why: "同一个人的所有记录只进入一组。若按窗口随机划分，相邻窗口几乎相同，模型会\"认出人\"而不是学会规律，分数会虚高。",
      svg, height, legend: groups.map(g => [color[g], groupName[g] + "集"]),
      table: [["集合", "人数", "规模", "参与者编号"], rows],
    };
  }

  // ② 训练人数：固定获选配置，只改变参与训练的用户数，在验证用户上评分。
  function learningStep(d) {
    const pts = d.learning, item = d.metric, users = [...new Set(pts.map(p => p.users))].sort((a, b) => a - b);
    const maxU = users.at(-1), H = 330, x = u => 90 + 600 * Math.log2(u) / Math.log2(maxU);
    const med = users.map(u => [u, median(pts.filter(p => p.users === u).map(p => p.score)), median(pts.filter(p => p.users === u).map(p => p.baseline))]);
    // 基线远离模型时不画进坐标，避免把模型曲线压扁，改在图上注明数值。
    const span = Math.max(...pts.map(p => p.score)) - Math.min(...pts.map(p => p.score));
    const near = Math.min(...med.map(m => Math.abs(m[2] - m[1]))) < span * 1.5;
    const y = scale(pts.flatMap(p => near ? [p.score, p.baseline] : [p.score]), 30, H - 50, item.better);
    const line = k => med.map(([u, ...v], i) => (i ? "L" : "M") + x(u).toFixed(1) + " " + y(v[k]).toFixed(1)).join("");
    let svg = yAxis(y, item) + users.map(u => `<text x="${x(u)}" y="${H - 26}" text-anchor="middle" class="tvTick">${u} 人</text>`).join("") +
      `<text x="${W / 2}" y="${H - 6}" text-anchor="middle" class="tvTick">参与训练的用户数（对数刻度）</text>` +
      (near ? `<path d="${line(1)}" class="tvDraw" pathLength="1" fill="none" stroke="${color.base}" stroke-width="2" style="--d:200ms"/>` :
        `<text x="${W - 24}" y="${H - 58}" text-anchor="end" class="tvTick tvFade" style="--d:1500ms">基线（多数类）约 ${format(item, median(med.map(m => m[2])))}，远低于图示范围</text>`) +
      `<path d="${line(0)}" class="tvDraw tvSolid" pathLength="1" fill="none" stroke="${color.model}" stroke-width="2.5" style="--d:200ms"/>`;
    const seen = {};
    pts.forEach(p => {
      const i = users.indexOf(p.users), j = seen[p.users] = (seen[p.users] ?? -1) + 1;
      svg += `<circle class="tvFade tvDot" cx="${(x(p.users) + (j % 5 - 2) * 3).toFixed(1)}" cy="${y(p.score)}" r="4.5" fill="${color.model}" fill-opacity=".55" stroke="${color.surface}" stroke-width="2" style="--d:${250 + i * 260}ms" data-tip="${p.users} 名训练用户 · ${item.label} ${format(item, p.score)}（基线 ${format(item, p.baseline)}）"/>`;
    });
    const [first, last] = [med[0], med.at(-1)];
    svg += `<g class="tvFade" style="--d:1700ms"><text x="${x(maxU) + 8}" y="${y(last[1]) + 4}" class="tvDirect" fill="${color.model}">获选模型</text>` +
      (near ? `<text x="${x(maxU) + 8}" y="${y(last[2]) + 4}" class="tvDirect" fill="${color.base}">基线</text>` : "") + "</g>";
    const spread = pts.filter(p => p.users === users[0]).map(p => p.score);
    return {
      nav: "训练人数", title: "更多的人，比更多的数据更重要",
      lead: `固定获选模型，只改变训练用户数。仅用 <b>1 人</b>训练时，验证${item.label}中位数为 <b>${format(item, first[1])}</b>（随抽到的人在 ${format(item, Math.min(...spread))}–${format(item, Math.max(...spread))} 间波动）；用全部 <b>${maxU} 人</b>为 <b>${format(item, last[1])}</b>。`,
      why: "每个人的皮肤、脂肪厚度和电极位置都不同。见过的人越多，模型越能学到跨人共通的规律，抽样波动也越小——这就是\"跨用户泛化\"。每个人数随机抽 5 组用户，点为单次结果，实线为中位数。",
      svg, height: H, legend: [[color.model, "获选模型（验证集）"], [color.base, "同样训练用户的基线", true]],
      axis: item.better === "low" ? "纵轴已翻转：越往上误差越小" : "越往上越好",
      table: [["训练用户数", `模型 ${item.label}（中位）`, `基线 ${item.label}（中位）`, "抽样次数"],
        med.map(([u, s, b]) => [u, format(item, s), format(item, b), pts.filter(p => p.users === u).length])],
    };
  }

  // ③ 候选搜索：按实验中的评估顺序逐个点亮，薄荷绿阶梯线记录"当前最佳"。
  function searchStep(d) {
    const c = d.candidates, item = d.metric, n = c.length, H = 330;
    const better = (a, b) => item.better === "low" ? a < b : a > b;
    const x = i => 80 + (n === 1 ? 300 : 640 * i / (n - 1));
    const y = scale(c.map(v => v.score), 24, H - 44, item.better);
    let best = 0, path = "";
    c.forEach((v, i) => {
      if (better(v.score, c[best].score)) { path += `H${x(i).toFixed(1)}V${y(v.score).toFixed(1)}`; best = i; }
      if (!i) path = `M${x(0).toFixed(1)} ${y(v.score).toFixed(1)}`;
    });
    path += `H${x(n - 1).toFixed(1)}`;
    const total = reduced ? 0 : Math.min(2600, 500 + n * 45);
    const ranked = [...c].map((v, i) => [v, i]).sort((a, b) => better(a[0].score, b[0].score) ? -1 : 1);
    const rank = new Map(ranked.map(([, i], r) => [i, r + 1]));
    let svg = `<defs><clipPath id="tvScan"><rect class="tvScanRect" x="60" y="0" width="${W - 60}" height="${H}" style="--t:${total}ms"/></clipPath></defs>` + yAxis(y, item) +
      `<text x="${W / 2}" y="${H - 8}" text-anchor="middle" class="tvTick">候选编号（按实验评估顺序，共 ${n} 个）</text>` +
      `<g clip-path="url(#tvScan)"><path d="${path}" fill="none" stroke="${color.model}" stroke-width="2"/>`;
    // 候选按实验循环顺序排列，同一组（如特征尺度）相邻；画出分组边界帮助比较。
    const runs = [];
    c.forEach((v, i) => { if (v.group !== undefined && (!runs.length || runs.at(-1).g !== v.group)) runs.push({g: v.group, i}); });
    const groups = runs.length > 1 && runs.length <= 6 ? runs.map((r, k) => {
      const end = (runs[k + 1]?.i ?? n) - 1, xa = k ? (x(r.i - 1) + x(r.i)) / 2 : 66;
      return (k ? `<path d="M${xa.toFixed(1)} 14V${H - 40}" stroke="#3a4352" stroke-dasharray="3 4"/>` : "") +
        `<text x="${((x(r.i) + x(end)) / 2).toFixed(1)}" y="14" text-anchor="middle" class="tvTick tvGroup">${esc(r.g)}</text>`;
    }).join("") : "";
    svg = svg.replace("<g clip-path", groups + "<g clip-path");
    c.forEach((v, i) => {
      const fill = color.family[Math.max(0, d.families.indexOf(v.family))];
      svg += `<circle cx="${x(i).toFixed(1)}" cy="${y(v.score).toFixed(1)}" r="5" fill="${fill}" stroke="${color.surface}" stroke-width="2" data-tip="#${i + 1} ${esc(v.label)}｜${item.label} ${format(item, v.score)}｜第 ${rank.get(i)} 名"/>`;
    });
    svg += `</g><line class="tvScanLine" x1="0" x2="0" y1="18" y2="${H - 38}" style="--x0:60px;--x1:${W - 4}px;--t:${total}ms"/>`;
    const bx = x(best), by = y(c[best].score), right = bx > W - 260;
    svg += `<g class="tvWin" style="--d:${total + 100}ms"><circle cx="${bx}" cy="${by}" r="11" fill="none" stroke="${color.model}" stroke-width="2"/>` +
      `<text x="${right ? bx - 18 : bx + 18}" y="${by + 4}" text-anchor="${right ? "end" : "start"}" class="tvCallout">获选 · ${format(item, c[best].score)}</text></g>`;
    const worst = ranked.at(-1)[0];
    return {
      nav: "候选搜索", title: "让验证集替我们挑模型",
      lead: `在训练用户上拟合 <b>${n} 个候选</b>（模型 × 参数 × 特征${c.some(v => /平滑/.test(v.label)) ? " × 平滑" : ""}），逐个在验证用户上打分。最佳为 <b>${esc(c[best].label)}</b>，${item.label} <b>${format(item, c[best].score)}</b>；最差 ${format(item, worst.score)}。`,
      why: "训练集上的分数总会偏乐观，所以选型只看没参与拟合的验证用户。测试用户此时仍然不可见，保证最终成绩没有被\"挑\"过。阶梯线是扫描到当前为止的最佳分数。",
      svg, height: H, scanMs: total,
      legend: d.families.map((f, i) => [color.family[i], f]).concat([[color.model, "当前最佳", "line"]]),
      axis: item.better === "low" ? "纵轴已翻转：越往上误差越小" : "越往上越好",
      table: [["名次", "候选", `验证 ${item.label}`], ranked.map(([v], r) => [r + 1, v.label, format(item, v.score)])],
    };
  }

  // ④ 留出评估：获选模型只在测试用户上跑一次，与基线并排比较。
  function heldoutStep(d) {
    const item = d.metric, rows = d.heldout, H = 70 + rows.length * 62;
    const max = Math.max(...rows.map(r => r.value)) * 1.12, x = v => 190 + 500 * v / max;
    let svg = `<path d="M190 20V${H - 30}" class="tvGrid"/>`;
    rows.forEach((r, i) => {
      const yy = 34 + i * 62, model = r.role === "model";
      svg += `<text x="178" y="${yy + 21}" text-anchor="end" class="tvRowLabel${model ? " tvStrong" : ""}">${esc(r.name)}</text>` +
        `<rect class="tvGrow" x="190" y="${yy + 4}" width="${(x(r.value) - 190).toFixed(1)}" height="26" rx="4" fill="${model ? color.model : "#4a5466"}" style="--d:${200 + i * 180}ms"/>` +
        `<text class="tvFade tvValue" x="${x(r.value) + 8}" y="${yy + 22}" style="--d:${700 + i * 180}ms">${format(item, r.value)}</text>`;
    });
    svg += `<text x="190" y="${H - 8}" class="tvTick">${item.label} · ${item.better === "low" ? "越短越好" : "越长越好"}</text>`;
    const model = rows.find(r => r.role === "model"), bases = rows.filter(r => r.role !== "model");
    const bestBase = bases.reduce((a, b) => (item.better === "low" ? b.value < a.value : b.value > a.value) ? b : a);
    const delta = item.better === "low" ? (bestBase.value - model.value) / bestBase.value : (model.value - bestBase.value) / Math.abs(bestBase.value || 1);
    return {
      nav: "留出评估", title: "在从没见过的人身上考一次",
      lead: `测试用户的${item.label}：获选模型 <b>${format(item, model.value)}</b>，最强对照\"${esc(bestBase.name)}\" ${format(item, bestBase.value)}，${delta >= 0 ? "相对改善" : "相对变差"} <b>${Math.abs(delta * 100).toFixed(1)}%</b>。`,
      why: "基线回答\"不看肌电能做到多好\"：始终报训练均值、始终猜最多的类别或只用时间。模型必须显著超过它，才说明肌电里确实有可学到的信息。",
      svg, height: H, legend: [[color.model, "获选模型"], ["#4a5466", "对照方法"]],
      table: [["方法", `测试 ${item.label}`], rows.map(r => [r.name, format(item, r.value)])],
    };
  }

  // ⑤ 逐用户：每一行是一名测试用户，点从基线位置移动到模型位置。
  function perUserStep(d) {
    const p = d.per_user, rows = [...p.rows], item = {...d.metric, percent: p.percent, digits: p.percent ? 1 : d.metric.digits};
    const has = rows.every(r => r.baseline !== undefined);
    const lowBetter = d.metric.better === "low";
    rows.sort((a, b) => lowBetter ? a.model - b.model : b.model - a.model);
    const rowH = rows.length > 12 ? 22 : 30, H = 50 + rows.length * rowH;
    const vals = rows.flatMap(r => has ? [r.model, r.baseline] : [r.model]);
    const vmin = Math.min(...vals), vmax = Math.max(...vals), pad = (vmax - vmin) * .12 || .05;
    const ticks = nice(Math.max(0, vmin - pad), vmax + pad, 5), lo = ticks[0], hi = ticks.at(-1);
    const x = v => 180 + 500 * (v - lo) / (hi - lo || 1);
    let svg = ticks.map(v => { return `<path d="M${x(v)} 14V${H - 30}" class="tvGrid"/><text x="${x(v)}" y="${H - 14}" text-anchor="middle" class="tvTick">${tick(item, v)}</text>`; }).join("");
    let wins = 0;
    rows.forEach((r, i) => {
      const yy = 26 + i * rowH, xm = x(r.model), xb = has ? x(r.baseline) : xm;
      const won = has && (lowBetter ? r.model < r.baseline : r.model > r.baseline);
      wins += won;
      svg += `<text x="168" y="${yy + 4}" text-anchor="end" class="tvRowLabel">${esc(r.subject)}</text>`;
      if (has) svg += `<line x1="${xb}" x2="${xm}" y1="${yy}" y2="${yy}" stroke="#5b6576" stroke-width="2" class="tvFade" style="--d:${300 + i * 40}ms"/>` +
        `<circle cx="${xb}" cy="${yy}" r="5" fill="${color.surface}" stroke="${color.base}" stroke-width="2"/>`;
      svg += `<g class="tvMove" style="--x0:${xb - xm}px;--y0:0px;--x1:0px;--y1:0px;--d:${300 + i * 40}ms"><circle cx="${xm}" cy="${yy}" r="6" fill="${color.model}" stroke="${color.surface}" stroke-width="2" data-tip="${esc(r.subject)}｜模型 ${format(item, r.model)}${has ? "｜对照 " + format(item, r.baseline) : ""}"/></g>`;
      svg += `<text x="${Math.max(xm, xb) + 12}" y="${yy + 4}" class="tvFade tvTick" style="--d:${900 + i * 40}ms">${has ? (won ? "✓ " : "✗ ") : ""}${format(item, r.model)}</text>`;
    });
    return {
      nav: "逐个用户", title: "平均分背后，每个人差多少",
      lead: has ? `${rows.length} 个测试对象中，<b>${wins} 个</b>模型优于对照。最好 ${esc(rows[0].subject)}：${format(item, rows[0].model)}；最差 ${esc(rows.at(-1).subject)}：${format(item, rows.at(-1).model)}。` :
        `只有 ${rows.length} 名测试参与者，结果差异很大：${rows.map(r => `${esc(r.subject)} 号 ${format(item, r.model)}`).join("，")}。`,
      why: "平均值会掩盖个体差异。逐人检查能发现模型对哪些人基本失效，这是部署前必须知道的风险，也是报告里\"失败案例分析\"的来源。",
      svg, height: H, legend: has ? [[color.model, "获选模型"], [color.base, "对照（空心）", "ring"]] : [[color.model, "获选模型"]],
      axis: p.label,
      table: [["对象", "模型", ...(has ? ["对照", "是否更好"] : [])], rows.map(r => [r.subject, format(item, r.model), ...(has ? [format(item, r.baseline), (lowBetter ? r.model < r.baseline : r.model > r.baseline) ? "是" : "否"] : [])])],
    };
  }

  function steps(d) {
    return [splitStep(d), d.learning ? learningStep(d) : null, searchStep(d), heldoutStep(d), perUserStep(d)].filter(Boolean);
  }
  function legendHtml(items) {
    return items.map(([c, label, kind]) => `<span class="tvKey"><i class="${kind === true || kind === "dash" ? "dash" : kind || "dot"}" style="--c:${c}"></i>${esc(label)}</span>`).join("");
  }
  function tableHtml([head, rows]) {
    return `<div class="scrollTable tvTable"><table><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>` +
      rows.map(r => `<tr>${r.map(v => `<td>${esc(v)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
  }
  // 画一张图并在下一帧开始过渡；重播只需再次调用。
  function play(figure, s) {
    figure.innerHTML = `<svg viewBox="0 0 ${W} ${s.height}" role="img" aria-label="${esc(s.title)}">${s.svg}</svg>`;
    const svg = figure.firstChild;
    if (reduced) svg.classList.add("play"); else requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.add("play")));
    return svg;
  }
  function tooltip(figure, tip) {
    figure.onpointermove = e => {
      const t = e.target.closest("[data-tip]");
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.dataset.tip; tip.hidden = false;
      const box = (tip.offsetParent ?? figure).getBoundingClientRect();
      const left = Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 4);
      tip.style.transform = `translate(${Math.max(4, left)}px, ${e.clientY - box.top + 14}px)`;
    };
    figure.onpointerleave = () => { tip.hidden = true; };
  }

  // 挂载完整的分步视图。
  function mount(host, key) {
    const d = window.trainingData?.[key];
    if (!d) { host.innerHTML = '<p class="tvMissing">缺少训练过程快照，请运行 <code>python demo/gesture/training_export.py</code> 后刷新。</p>'; return; }
    const list = steps(d);
    let index = Math.min(host._step ?? 0, list.length - 1), showTable = false;
    host.innerHTML = `<section class="trainView" aria-label="${esc(d.title)}的训练过程">
      <ol class="tvSteps">${list.map((s, i) => `<li><button type="button" data-i="${i}"><span>${String(i + 1).padStart(2, "0")}</span>${s.nav}</button></li>`).join("")}</ol>
      <div class="tvStage">
        <div class="tvCopy"><p class="tvCount"></p><h2 class="tvTitle"></h2><p class="tvLead"></p><p class="tvWhy"></p></div>
        <div class="tvFigureWrap">
          <div class="tvBar"><div class="tvLegend"></div><div class="tvActions"><button type="button" class="tvReplay">↻ 重播</button><button type="button" class="tvToggle" aria-pressed="false">表格</button></div></div>
          <div class="tvFigure"></div><div class="tvTip" hidden></div><p class="tvAxis"></p>
        </div>
      </div>
      <div class="tvNav"><button type="button" class="tvPrev">← 上一步</button><div class="tvProgress"><i></i></div><button type="button" class="tvNext">下一步 →</button></div>
    </section>`;
    const q = sel => host.querySelector(sel), figure = q(".tvFigure");
    tooltip(figure, q(".tvTip"));
    function render() {
      const s = list[index];
      host._step = index;
      q(".tvCount").textContent = `${d.title} · 第 ${index + 1} 步 / 共 ${list.length} 步`;
      q(".tvTitle").textContent = s.title;
      q(".tvLead").innerHTML = s.lead; q(".tvWhy").textContent = s.why;
      q(".tvLegend").innerHTML = legendHtml(s.legend);
      q(".tvAxis").textContent = s.axis ?? "";
      q(".tvToggle").setAttribute("aria-pressed", showTable); q(".tvToggle").textContent = showTable ? "图形" : "表格";
      q(".tvReplay").disabled = showTable;
      if (showTable) figure.innerHTML = tableHtml(s.table); else play(figure, s);
      host.querySelectorAll(".tvSteps button").forEach((b, i) => {
        if (i === index) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
        b.dataset.done = i < index;
      });
      q(".tvProgress i").style.width = `${100 * (index + 1) / list.length}%`;
      q(".tvPrev").disabled = index === 0; q(".tvNext").disabled = index === list.length - 1;
      q(".tvStage").animate?.(reduced ? [] : [{opacity: .35, transform: "translateY(4px)"}, {opacity: 1, transform: "none"}], {duration: 220, easing: "ease-out"});
    }
    // 地址记录当前步骤，方便课堂上直接打开某一步。
    const go = i => {
      index = Math.max(0, Math.min(list.length - 1, i)); render();
      const parts = location.hash.slice(1).split("/").filter(p => !/^\d+$/.test(p));
      history.replaceState(null, "", "#" + parts.concat(index + 1).join("/"));
    };
    host.querySelectorAll(".tvSteps button").forEach(b => b.onclick = () => go(+b.dataset.i));
    q(".tvPrev").onclick = () => go(index - 1); q(".tvNext").onclick = () => go(index + 1);
    q(".tvReplay").onclick = () => play(figure, list[index]);
    q(".tvToggle").onclick = () => { showTable = !showTable; render(); };
    host._keys && document.removeEventListener("keydown", host._keys);
    host._keys = e => {
      if (host.hidden || e.target.closest("input,select,textarea")) return;
      if (e.key === "ArrowRight") go(index + 1); else if (e.key === "ArrowLeft") go(index - 1);
    };
    document.addEventListener("keydown", host._keys);
    render();
  }

  // 单张图：供开发过程视图嵌入候选搜索或留出评估。
  function chart(el, key, kind) {
    const d = window.trainingData?.[key];
    if (!d) return false;
    const s = {search: searchStep, heldout: heldoutStep, users: perUserStep}[kind](d);
    el.innerHTML = `<div class="tvEmbed"><div class="tvLegend">${legendHtml(s.legend)}</div><div class="tvFigure"></div><div class="tvTip" hidden></div><p class="tvAxis">${esc(s.axis ?? "")}</p></div>`;
    const figure = el.querySelector(".tvFigure");
    tooltip(figure, el.querySelector(".tvTip"));
    play(figure, s);
    return true;
  }

  // 根据当前页面和地址决定是否显示训练过程视图。
  function route() {
    const host = document.getElementById("trainingView");
    if (!host) return;
    const lab = document.querySelector(".labHeader")?.dataset.lab;
    const parts = location.hash.slice(1).split("/"), [first, second] = parts, step = Number(parts.at(-1));
    const active = first === "training" || second === "training";
    host.hidden = !active;
    if (!active) { host._key = null; return; }
    const key = lab === "gesture" ? "gesture" : lab === "fatigue" ? "fatigue" : ["levels", "change"].includes(first) ? first : "continuous";
    if (host._key !== key) { host._step = 0; host._key = key; }
    if (Number.isInteger(step) && step > 0) host._step = step - 1;
    mount(host, key);
  }
  window.addEventListener("hashchange", route);
  document.addEventListener("DOMContentLoaded", route);
  if (document.readyState !== "loading") setTimeout(route);
  return {mount, chart};
})();
