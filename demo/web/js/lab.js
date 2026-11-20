"use strict";
const labHeader = document.querySelector(".labHeader");
const currentLab = labHeader.dataset.lab;
const labRoutes = [
  ["force", "force.html", "指力预测", "M2 16h3v-4h3v-5h3v9h3V4h3v12h3"],
  ["gesture", "index.html", "手势信号", "M1 12h4l3-7 3 14 3-12 2 5h5"],
  ["fatigue", "fatigue.html", "肌肉疲劳", "M1 6l3 9 3-10 3 11 3-8 3 6 2-3h3"],
];
labHeader.innerHTML = '<div class="brand"><b>∿</b> EMGInsight</div><nav class="labNav" aria-label="切换实验台">' +
  labRoutes.map(([id, url, name, d]) => `<a href="${url}" ${id === currentLab ? 'aria-current="page"' : ""}><svg viewBox="0 0 22 22" aria-hidden="true"><path d="${d}"/></svg>${name}</a>`).join("") +
  '</nav><small id="sourceMode">' + ({force:"putEMG / 离线回放",gesture:"UCI / 离线回放",fatigue:"Zenodo / 离线回放"})[currentLab] + '</small>';

// 保留原有 force.html#development 与空哈希入口，模型和视图均可分享。
function modelRoute() {
  const [key, view] = location.hash.slice(1).split("/");
  const model = currentLab === "fatigue" ? "fatigue" : ["levels","change"].includes(key) ? key : "continuous";
  const training = key === "training" || view === "training";
  return {model, training, development: !training && (key === "development" || view === "development"), view: training ? "training" : (key === "development" || view === "development") ? "development" : "replay"};
}
