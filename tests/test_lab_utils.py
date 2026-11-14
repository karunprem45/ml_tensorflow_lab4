import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lab_utils as lu  # noqa: E402


def fake_data(n=300, n_features=5, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype("float32")
    y = np.repeat(np.arange(n_classes), n // n_classes)
    return X, y


# ---------- data ----------

def test_split_sizes_and_no_overlap():
    X, y = fake_data()
    X_tr, X_te, y_tr, y_te = lu.split_data(X, y, test_size=0.2)
    assert len(X_tr) == 240 and len(X_te) == 60
    rows_tr = {tuple(r) for r in X_tr}
    rows_te = {tuple(r) for r in X_te}
    assert rows_tr.isdisjoint(rows_te)


def test_split_is_stratified():
    X, y = fake_data()
    _, _, _, y_te = lu.split_data(X, y, test_size=0.2)
    # each of the 3 classes should keep its 1/3 share in the test set
    assert np.bincount(y_te).tolist() == [20, 20, 20]


def test_scale_fits_on_first_array_only():
    X, _ = fake_data()
    X_fit, X_other = X[:200], X[200:] + 10.0
    X_fit_s, X_other_s = lu.scale(X_fit, X_other)
    assert np.allclose(X_fit_s.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(X_fit_s.std(axis=0), 1, atol=1e-3)
    # the shifted array must NOT be re-centred, which would mean it leaked into the fit
    assert X_other_s.mean() > 5


def test_cv_folds_cover_every_row_once():
    _, y = fake_data()
    folds = lu.cv_folds(y, n_splits=3)
    assert len(folds) == 3
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val.tolist()) == list(range(len(y)))
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(val_idx)


# ---------- training helpers ----------

def test_lr_schedule():
    assert lu.lr_schedule(0, 0.01) == 0.01
    assert lu.lr_schedule(4, 0.01) == 0.01
    assert lu.lr_schedule(5, 0.01) == pytest.approx(0.0095)


def test_macro_f1_perfect_and_wrong():
    y = np.array([0, 1, 2])
    assert lu.macro_f1(y, np.eye(3)) == pytest.approx(1.0)
    wrong = np.eye(3)[[1, 2, 0]]
    assert lu.macro_f1(y, wrong) == pytest.approx(0.0)


def test_select_best_uses_cv_not_test():
    results = [
        {"run": "a", "cv_macro_f1": 0.80, "test_macro_f1": 0.95},
        {"run": "b", "cv_macro_f1": 0.85, "test_macro_f1": 0.60},
    ]
    assert lu.select_best(results)["run"] == "b"


def test_select_best_empty_raises():
    with pytest.raises(ValueError):
        lu.select_best([])


def test_confusion_matrix_figure_makes_png():
    cm = np.array([[5, 1], [2, 7]])
    fig = lu.confusion_matrix_figure(cm, ["a", "b"])
    png = lu.figure_to_png_bytes(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ---------- model (needs TensorFlow) ----------

def test_build_model_output_shape_and_probabilities():
    pytest.importorskip("tensorflow")
    model = lu.build_model(n_features=5, n_classes=3, optimizer_name="sgd", lr=0.01)
    X, _ = fake_data(n=6)
    probs = model.predict(X, verbose=0)
    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert lu.current_lr(model) == pytest.approx(0.01)


def test_build_model_rejects_unknown_optimizer():
    pytest.importorskip("tensorflow")
    with pytest.raises(ValueError):
        lu.build_model(5, 3, optimizer_name="rmsprop_typo")
