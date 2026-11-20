"""导出 C 任务模型选择演示所需的少量可核对数据。"""

import numpy as np

from emginsight.features import TIME_DOMAIN_FEATURES
from emginsight.models import KNNClassifier, LogisticRegression, Standardizer
from emginsight.uci_gesture import load_uci_gesture_record
from emginsight.uci_gesture.experiment import FEATURE_SETS


def feature_name(column):
    channel, feature = divmod(int(column), 3)
    return f"CH{channel + 1} {TIME_DOMAIN_FEATURES[feature]}"


def describe_tree(forest, vector, columns):
    estimator = forest.estimators_[0]
    tree = estimator.tree_
    path = set(estimator.decision_path(vector).indices.tolist())
    nodes = []

    def visit(node, slot, depth):
        if node < 0 or depth > 2:
            return
        feature = tree.feature[node]
        nodes.append(dict(slot=slot, path=node in path, feature=feature_name(columns[feature]) if feature >= 0 else None,
                          threshold=float(tree.threshold[node]), value=float(vector[0, feature]) if feature >= 0 else None))
        visit(tree.children_left[node], 2 * slot + 1, depth + 1)
        visit(tree.children_right[node], 2 * slot + 2, depth + 1)

    visit(0, 0, 0)
    return dict(nodes=nodes, depth=len(path) - 1, single=estimator.predict_proba(vector)[0].tolist(),
                probabilities=forest.predict_proba(vector)[0].tolist(), prediction=int(forest.predict(vector)[0]))


def export_development(data, scores, forest, forest_scaler, forest_columns, summary, raw):
    """在同一个验证窗口展示三个模型，并保留全部真实选型分数。"""
    X_train, train = data["train"]
    X_val, validation = data["validation"]
    fist = np.flatnonzero(validation.label.to_numpy() == 2)
    distances = np.linalg.norm(forest_scaler.transform(X_val[fist][:, forest_columns]), axis=1)
    row = int(fist[np.argsort(distances)[len(fist) // 2]])
    sample = X_val[[row]]
    reference = validation.iloc[row]
    truth = int(reference.label)
    start = int(reference.start_sample)
    record = load_uci_gesture_record(raw / reference.subject / (reference.record + ".txt"))
    waveform = record.signal[start:start + 200]
    ranked = scores.sort_values("macro_f1", ascending=False, kind="stable")
    best = ranked.drop_duplicates("model").set_index("model")

    def inputs(model):
        config = best.loc[model]
        cols = np.flatnonzero(np.isin(np.arange(X_train.shape[1]) % 3, FEATURE_SETS[config.features]))
        scale = Standardizer().fit(X_train[:, cols])
        return cols, scale.transform(X_train[:, cols]), scale.transform(sample[:, cols])

    columns, training, vector = inputs("logistic")
    logistic = LogisticRegression(l2=float(best.loc["logistic", "parameter"].split("=")[1])).fit(training, train.label)
    x = np.linspace(-3, 3, 81)
    # 比较一维切片的变化范围，选择能清晰展示目标类别变化的特征。
    variations = []
    for feature in range(vector.shape[1]):
        sweep = np.repeat(vector, len(x), axis=0)
        sweep[:, feature] += x
        variations.append(logistic.predict_proba(sweep))
    feature = int(np.argmax([np.ptp(p[:, truth - 1]) for p in variations]))
    contributions = logistic.coef_ * vector[0]
    logits = contributions.sum(axis=1) + logistic.intercept_
    curve = dict(x=(x + vector[0, feature]).tolist(), p=variations[feature].tolist(),
                 feature=feature_name(columns[feature]), current=float(vector[0, feature]),
                 prediction=int(logistic.predict(vector)[0]), probabilities=logistic.predict_proba(vector)[0].tolist(),
                 logits=logits.tolist(), contributions=contributions.tolist(),
                 weights=logistic.coef_.tolist(), intercept=logistic.intercept_.tolist(),
                 columns=[feature_name(column) for column in columns])

    _, training, vector = inputs("knn")
    k = int(best.loc["knn", "parameter"].split("=")[1])
    knn = KNNClassifier(k=k).fit(training, train.label)
    distance = np.linalg.norm(training - vector, axis=1)
    nearest = np.argsort(distance, kind="stable")[:k]
    labels = train.label.to_numpy()[nearest]
    prediction = int(knn.predict(vector)[0])
    assert prediction == np.bincount(labels).argmax()
    neighbors = dict(points=[dict(label=int(label), distance=float(distance[i])) for i, label in zip(nearest, labels)],
                     prediction=prediction)

    forest_vector = forest_scaler.transform(sample[:, forest_columns]).astype(np.float32)
    forest_train = X_train[:, forest_columns]
    standardized = forest_scaler.transform(forest_train)
    train_labels = train.label.to_numpy()
    preview = [int(np.flatnonzero(train_labels == label)[0]) for label in forest.classes_]
    matrix = dict(
        shape=list(forest_train.shape), labels=train_labels[preview].tolist(),
        rows=forest_train[preview].tolist(), z=standardized[preview].tolist(),
        mean=forest_scaler.mean_.tolist(), scale=forest_scaler.scale_.tolist(),
        class_mean=[standardized[train_labels == label].mean(axis=0).tolist() for label in forest.classes_],
        columns=[feature_name(column) for column in forest_columns],
    )
    return dict(
        sample=dict(subject=str(reference.subject), start=start, label=truth,
                    waveform=waveform.T.tolist(), features=sample[0].reshape(8, 3).tolist()),
        windows=summary["windows"], users={name: len(group) for name, group in summary["subjects"].items()},
        scores=scores.to_dict(orient="records"), best=best.to_dict(orient="index"),
        test=summary["test"]["prediction"], curve=curve, neighbors=neighbors, matrix=matrix,
        forest_params=dict(trees=len(forest.estimators_), max_depth=forest.max_depth,
                           criterion=forest.criterion),
        forest=describe_tree(forest, forest_vector, forest_columns),
    )
