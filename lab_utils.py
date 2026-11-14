"""Helper functions for Lab 4 (TensorBoard).

The notebook calls these functions. They live in their own file so they can be unit tested
(see tests/test_lab_utils.py). TensorFlow is imported inside the functions that need it,
so the data and selection functions can be tested without it.
"""
import io

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = [
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
    "Aspen", "Douglas-fir", "Krummholz",
]


# ---------- data ----------

def load_covtype(n_samples=20000, seed=42):
    """Load a stratified sample of the Forest Cover Type dataset.

    Labels in the original data are 1-7, so they are shifted to 0-6.
    """
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype()
    X = data.data.astype("float32")
    y = (data.target - 1).astype("int64")
    X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=seed)
    return X, y


def split_data(X, y, test_size=0.2, seed=42):
    """Stratified train/test split. The test part is only used once, at the very end."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def scale(X_fit, *others):
    """Fit a StandardScaler on X_fit only, then apply it to X_fit and every other array.

    This keeps validation and test data from leaking into the scaling.
    """
    scaler = StandardScaler().fit(X_fit)
    return (scaler.transform(X_fit),) + tuple(scaler.transform(o) for o in others)


def cv_folds(y, n_splits=3, seed=42):
    """Stratified K-fold indices: a list of (train_idx, val_idx) pairs."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------- training helpers ----------

def lr_schedule(epoch, lr):
    """Keep the starting learning rate for 5 epochs, then shrink it by 5% each epoch."""
    return float(lr) if epoch < 5 else float(lr) * 0.95


def macro_f1(y_true, probs):
    """Macro F1 from predicted class probabilities. Macro F1 treats every class equally,
    which matters here because some forest types are rare."""
    return float(f1_score(y_true, np.asarray(probs).argmax(axis=1), average="macro"))


def select_best(cv_results, metric="cv_macro_f1"):
    """Pick the config with the highest mean cross-validation score.

    Only the CV metric is used. Test scores are never looked at here, so the test set
    stays held out until the chosen model is evaluated once.
    """
    if not cv_results:
        raise ValueError("cv_results is empty")
    return max(cv_results, key=lambda r: r[metric])


def build_model(n_features, n_classes, optimizer_name="adam", lr=1e-3):
    """Small dense network for multi-class classification."""
    from tensorflow import keras

    optimizers = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD}
    if optimizer_name not in optimizers:
        raise ValueError(f"unknown optimizer: {optimizer_name}")
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers[optimizer_name](learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def current_lr(model):
    """Read the optimizer's learning rate as a float (works on Keras 2 and Keras 3)."""
    lr = model.optimizer.learning_rate
    try:
        return float(np.array(lr))
    except (TypeError, ValueError):
        return float(lr.numpy())


def make_custom_scalars_callback(logdir, X_val, y_val):
    """Callback that writes two extra scalars to TensorBoard after every epoch:
    the current learning rate and validation macro F1."""
    import tensorflow as tf
    from tensorflow import keras

    class CustomScalars(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.writer = tf.summary.create_file_writer(logdir + "/custom")

        def on_epoch_end(self, epoch, logs=None):
            probs = self.model.predict(X_val, verbose=0)
            with self.writer.as_default():
                tf.summary.scalar("learning_rate", current_lr(self.model), step=epoch)
                tf.summary.scalar("val_macro_f1", macro_f1(y_val, probs), step=epoch)
            self.writer.flush()

    return CustomScalars()


# ---------- confusion matrix image for TensorBoard ----------

def confusion_matrix_figure(cm, class_names):
    """Draw a confusion matrix (rows = true class, columns = predicted class)."""
    from matplotlib.figure import Figure

    cm = np.asarray(cm)
    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def figure_to_image(fig):
    """Turn a matplotlib figure into a 4-D image tensor for tf.summary.image."""
    import tensorflow as tf

    image = tf.image.decode_png(figure_to_png_bytes(fig), channels=4)
    return tf.expand_dims(image, 0)
