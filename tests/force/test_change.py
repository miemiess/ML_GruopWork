"""用合成参考牛顿读数验证标定、跨日比较和患者输出格式。"""

import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from emginsight.force.change import calibrate, daily_changes, main, read_trials
from emginsight.force.change_predict import run as predict_change
from emginsight.force.model import ForceModel, save_force_model
from emginsight.force.targets import TARGETS


## @brief 写入小型第一阶段模型、三天原始肌电与同步的合成参考读数。
@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 72))
    scaler = StandardScaler().fit(X)
    reg = LinearRegression().fit(scaler.transform(X), X[:, :10])
    model = ForceModel(reg, scaler, "raw", "RMS+MAV+WL", 2, TARGETS,
                       {"window_size": 128, "step_size": 64, "calibration_seconds": 0.05}, ())
    model_path = tmp_path / "model.joblib"
    save_force_model(model, model_path)
    rows = []
    for day, loads in enumerate(([1, 3, 5], [4, 4], [2, 2]), start=1):
        time = np.arange(5120) / 5120
        wave = np.sin(2 * np.pi * 120 * time)
        amp = np.ones(len(time))
        for i, load in enumerate(loads):
            start = 0.15 + 0.25 * i
            amp[(time >= start) & (time < start + 0.2)] = load
            rows.append(dict(patient="001", protocol="left-pinch", date=f"2026-06-0{day}",
                             record=f"{day}.csv", start_s=start + 0.04, end_s=start + 0.18,
                             reference_n=load * 10, split="fit" if day == 1 else "check"))
        pd.DataFrame({f"EMG_{i}": amp * wave for i in range(1, 25)}).to_csv(tmp_path / f"{day}.csv", index=False)
    return model_path, pd.DataFrame(rows)


## @brief 验证原始 CSV 到患者单值的完整流程，随访无需测力计读数。
def test_calibrated_change(inputs, tmp_path, capsys):
    model, rows = inputs
    manifest, artifact = tmp_path / "trials.csv", tmp_path / "personal.json"
    args = ["--trials", str(manifest), "--calibration", str(artifact), "--model", str(model)]
    rows.to_csv(manifest, index=False)
    assert main(["fit", *args]) == 0
    calibration = json.loads(artifact.read_text(encoding="utf-8"))
    assert calibration["patient"] == "001" and len(calibration["weights"]) == 10
    validation = calibration["validation"]
    assert validation["measured_changes_n"] == [-20] and validation["delta_mae_n"] < 2
    assert validation["no_change_mae_n"] == 20
    capsys.readouterr()
    visits = rows[rows.split == "check"].drop(columns=["reference_n", "split"]).iloc[::-1]
    visits.to_csv(manifest, index=False)
    assert main(["change", *args]) == 0
    line = capsys.readouterr().out.strip()
    assert re.fullmatch(r"本次动作力量较基线的估计变化\s+-\d+\.\d N", line)
    assert f"{validation['estimated_changes_n'][0]:+.1f} N" in line

    visits.assign(patient="002").to_csv(manifest, index=False)
    with pytest.raises(SystemExit):
        main(["change", *args])
    assert "患者" in capsys.readouterr().err
    visits.to_csv(manifest, index=False)
    calibration["validation"]["delta_mae_n"] = 21
    artifact.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["change", *args])
    assert "误差" in capsys.readouterr().err


## @brief 检查独立参考、整日留出以及动作片段重叠的约束。
def test_invalid_calibration_manifest(inputs, tmp_path):
    _, rows = inputs
    path = tmp_path / "trials.csv"
    for invalid, message in (
        (rows.drop(columns="reference_n"), "reference_n"),
        (rows.assign(reference_n=np.inf), "有限"),
        (rows.assign(reference_n=1), "不同的测力计"),
        (rows[rows.date != "2026-06-03"], "两个不同日期"),
        (pd.concat([rows, rows.iloc[:1]]), "重叠"),
    ):
        invalid.to_csv(path, index=False)
        with pytest.raises(ValueError, match=message):
            read_trials(path, True)


## @brief 验证留出日真值不影响标定权重，各次动作先按日期独立取中位数。
def test_validation_never_fits_and_daily_aggregation(inputs):
    _, rows = inputs
    X = np.random.default_rng(5).normal(size=(len(rows), 10))
    before = calibrate(rows, X, "test")
    rows.loc[rows.split == "check", "reference_n"] *= 2
    after = calibrate(rows, X, "test")
    assert before["weights"] == after["weights"]
    assert before["validation"]["measured_changes_n"] != after["validation"]["measured_changes_n"]
    assert daily_changes(["a", "a", "a", "b"], [1, 2, 100, 6]).tolist() == [4]


## @brief 回读真实序列化的个人模型，核对部署输出、变化符号及模型卡校验。
def test_personal_artifact_and_outputs(inputs, tmp_path):
    stage1, rows = inputs
    manifest = tmp_path / "trials.csv"
    artifact = tmp_path / "personal.joblib"
    card = artifact.with_suffix(".card.json")
    output = tmp_path / "out"
    rows.to_csv(manifest, index=False)
    main(["fit", "--trials", str(manifest), "--calibration", str(artifact), "--model", str(stage1)])
    rows[rows.split == "check"].drop(columns=["reference_n", "split"]).to_csv(manifest, index=False)
    summary = predict_change(manifest, stage1, artifact, card, output)
    assert summary["unit"] == "N" and -22 < summary["change"] < -18
    assert (output / "patient_result.txt").read_text(encoding="utf-8").count("\n") == 1
    days = pd.read_csv(output / "daily.csv")
    assert days["median"].diff().iloc[-1] == pytest.approx(summary["change"])
    assert len(pd.read_csv(output / "trials.csv")) == 4 and (output / "force_change.png").stat().st_size
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    card.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        predict_change(manifest, stage1, artifact, card, output)
