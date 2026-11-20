"use strict";

const SIM_WINDOW = 200;
const SIM_STEP = 100;
const SIM_HISTORY = 1200;

// 对应 time_domain.py、Standardizer.transform 和 sklearn 的森林概率平均。
function inferGesture(window, model) {
  const features = [];
  for (let channel = 0; channel < 8; channel++) {
    let squares = 0, absolute = 0, length = 0;
    for (let i = 0; i < window.length; i++) {
      const value = window[i][channel];
      squares += value ** 2;
      absolute += Math.abs(value);
      if (i) length += Math.abs(value - window[i - 1][channel]);
    }
    features.push(Math.sqrt(squares / window.length), absolute / window.length, length);
  }
  const standardized = model.columns.map((column, i) => (features[column] - model.mean[i]) / model.scale[i]);
  const input = Float32Array.from(standardized);
  const probabilities = Array(model.classes.length).fill(0);
  for (const nodes of model.trees) {
    let node = nodes[0];
    while (node.length === 4) {
      const [left, right, feature, threshold] = node;
      node = nodes[input[feature] <= threshold ? left : right];
    }
    node.forEach((p, i) => probabilities[i] += p);
  }
  const p = probabilities.map(total => total / model.trees.length);
  const pred = model.classes[p.indexOf(Math.max(...p))];
  return {pred, f: features, z: standardized, p};
}

function createSimulation(model) {
  const buffer = Array(SIM_HISTORY);
  let gesture = 1, count = 0, elapsed = 0, frame = null;
  const weights = model.classes.map(label => Number(label === gesture));
  const peak = 1.04 * Math.max(...Object.values(model.templates).flat(2).map(Math.abs), 1e-12);
  const history = size => Array.from({length: size}, (_, i) => buffer[(count - size + i) % SIM_HISTORY]);

  return {
    peak,
    get gesture() { return gesture; },
    get frame() { return frame; },
    select(label) { gesture = label; },
    advance(ms) {
      elapsed += Math.max(0, ms);
      // 虚拟时钟每毫秒产生一个采样点，切换手势时约 40 点过渡。
      while (count < Math.floor(elapsed)) {
        const sample = Array(8).fill(0);
        const gain = 1 + .04 * Math.sin(count / 240);
        model.classes.forEach((label, i) => {
          weights[i] += (Number(label === gesture) - weights[i]) / 14;
          const template = model.templates[label];
          const row = template[count % template.length];
          for (let ch = 0; ch < 8; ch++) sample[ch] += row[ch] * weights[i] * gain;
        });
        buffer[count % SIM_HISTORY] = sample;
        count++;
        if (count >= SIM_WINDOW && (count - SIM_WINDOW) % SIM_STEP === 0) {
          frame = {a: count - SIM_WINDOW, b: count - 1, start: count - SIM_WINDOW,
                   ...inferGesture(history(SIM_WINDOW), model)};
        }
      }
    },
    snapshot() {
      const size = Math.min(count, SIM_HISTORY);
      const rows = history(size);
      return {
        t: Array.from({length: size}, (_, i) => count - size + i),
        channels: Array.from({length: 8}, (_, ch) => rows.map(row => row[ch]))
      };
    }
  };
}
