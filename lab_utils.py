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


def balanced_class_weights(y, n_classes, power=1.0):
    """Class weights for Keras' class_weight argument: (n_samples / (n_classes * count)) ** power.

    With power=1 this is scikit-learn's "balanced" formula: every class contributes the same
    total weight to the loss. power=0.5 ("sqrt") is a milder version that still boosts rare
    classes but much less. A class that is missing from y gets weight 1.0.
    """
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes)
    return {
        c: float((len(y) / (n_classes * counts[c])) ** power) if counts[c] else 1.0
        for c in range(n_classes)
    }


CLASS_WEIGHT_POWERS = {"none": None, "sqrt": 0.5, "balanced": 1.0}


def class_weights_for(setting, y, n_classes):
    """Turn a config's class-weight setting ("none", "sqrt" or "balanced") into weights."""
    if setting not in CLASS_WEIGHT_POWERS:
        raise ValueError(f"unknown class weight setting: {setting}")
    power = CLASS_WEIGHT_POWERS[setting]
    return None if power is None else balanced_class_weights(y, n_classes, power)


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
    if hasattr(lr, "numpy"):
        return float(lr.numpy())
    return float(lr)


def make_custom_scalars_callback(logdir, X_val, y_val):
    """Callback that writes two extra scalars to TensorBoard after every epoch:
    the current learning rate and validation macro F1. Both are shown on the
    Custom Scalars dashboard (see custom_scalars_layout)."""
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


# ---------- Custom Scalars dashboard ----------

# Tags written during training, and the Custom Scalars chart each one goes in.
# epoch_loss / epoch_accuracy come from the Keras TensorBoard callback,
# learning_rate / val_macro_f1 come from make_custom_scalars_callback.
CUSTOM_SCALAR_CHARTS = {
    "Training health": {
        "Loss (train and validation)": r"^epoch_loss$",
        "Learning rate": r"^learning_rate$",
    },
    "Model quality": {
        "Validation macro F1": r"^val_macro_f1$",
        "Accuracy (train and validation)": r"^epoch_accuracy$",
    },
}


def custom_scalars_layout(charts=CUSTOM_SCALAR_CHARTS):
    """Build the layout for TensorBoard's Custom Scalars dashboard.

    Each chart is a multiline chart that collects one tag from every run, so all
    configs and folds can be compared on the same axes.
    """
    from tensorboard.plugins.custom_scalar import layout_pb2

    return layout_pb2.Layout(category=[
        layout_pb2.Category(title=category, chart=[
            layout_pb2.Chart(title=title, multiline=layout_pb2.MultilineChartContent(tag=[regex]))
            for title, regex in chart_map.items()
        ])
        for category, chart_map in charts.items()
    ])


def write_custom_scalars_layout(logdir):
    """Write the Custom Scalars layout once, at the top of the log folder."""
    import tensorflow as tf
    from tensorboard.plugins.custom_scalar import summary as cs_summary

    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        tf.summary.experimental.write_raw_pb(
            cs_summary.pb(custom_scalars_layout()).SerializeToString(), step=0
        )
    writer.flush()


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
