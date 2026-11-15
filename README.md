# Lab 4: TensorBoard (modified)

Karuniya Premnath, IE 7374 MLOps

![Unit tests](https://github.com/karunprem45/ml_tensorflow_lab4/actions/workflows/tests.yml/badge.svg)
![Notebook end-to-end run](https://github.com/karunprem45/ml_tensorflow_lab4/actions/workflows/notebook.yml/badge.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karunprem45/ml_tensorflow_lab4/blob/main/Lab4_TensorBoard.ipynb)

This lab is based on `Labs/Tensorflow_Labs/TensorBoard/Lab1.ipynb` from [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps).
The original trains a one-layer model on 1,000 made-up points and shows the loss in TensorBoard.
I moved it to a real classification problem, compared five configs with cross-validation (optimizer, learning rate and class weights), and used seven TensorBoard dashboards instead of one.

## Dataset

Forest Cover Type (UCI, loaded with `sklearn.datasets.fetch_covtype`). Each row is a 30m x 30m patch of forest in Colorado with 54 features (elevation, slope, distance to water and roads, sunlight, wilderness area, soil type). The task is to predict which of 7 tree types covers it.

I use a stratified sample of 20,000 rows so everything runs in about 10 minutes on Colab. The classes are very unbalanced (Cottonwood/Willow is under 1%), so the main metric is macro F1, not accuracy.

## My Modifications

| Area | Original lab | My version |
|---|---|---|
| Dataset | 1,000 made-up points (y = 0.5x + 2 + noise) | Forest Cover Type, 20,000 rows, 54 real features, 7 classes |
| Task | Regression | Multi-class classification |
| Model | Dense(16) then Dense(1), no activation | Dense(128, relu), Dropout(0.2), Dense(64, relu), Dense(7, softmax) |
| Training | One run, SGD with learning rate 0.2 | Five configs, each with 3-fold cross-validation. Up to 100 epochs; early stopping (patience 5, min change 0.001) decides when each run ends |
| Class imbalance | Not relevant | Three class-weight settings compared on the same model: none, `sqrt` and `balanced` |
| Model selection | None | Best config picked by mean CV macro F1. The test set is not used for picking |
| Evaluation | Printed average training loss labelled as "test loss" | Chosen model retrained on the full training set and tested once: macro F1, accuracy, per-class report |
| Callbacks | TensorBoard only | TensorBoard with weight histograms, learning rate schedule, early stopping, my own callback |
| Custom Scalars | Not used | My own Custom Scalars layout (2 groups, 4 charts): train/validation loss, learning rate, validation macro F1, train/validation accuracy. The learning rate and validation macro F1 are written by my callback with `tf.summary` every epoch |
| HParams | Not used | Optimizer, learning rate and class weights logged as hyperparameters, CV macro F1 and CV accuracy as metrics |
| Images | Not used | Test-set confusion matrix logged to the Images tab |
| Debugger | Turned on, but the model trains fine so there is nothing to see | Separate run on unscaled data with a high learning rate (0.5), so the loss goes to NaN and Debugger V2 shows alerts |
| Code structure | Everything in the notebook | Helper functions in `lab_utils.py` |
| Testing | None | 16 unit tests in `tests/`, run by GitHub Actions on Python 3.11 and 3.13 and inside the notebook, plus a CI job that runs the whole notebook |

## How the best model is picked

1. 20% of the data is split off as the test set at the start and not used again until the last step.
2. Each config is trained with 3-fold cross-validation on the remaining 80%. The scaler and the class weights are computed from each fold's training part only.
3. `select_best()` picks the config with the highest mean CV macro F1. It never sees test scores (there is a unit test that checks this).
4. That config is retrained on the full training set for its average best epoch from CV, and evaluated on the test set once.

## How to run

1. Click the **Open in Colab** badge at the top. It always opens the latest notebook from this repo.
2. Click Runtime > Run all. The first cell clones this repo to get `lab_utils.py` and the tests.
3. TensorBoard opens in the second-to-last cell. Debugger V2 opens in the last cell, in its own TensorBoard pointed at `logs/debugger`, because Debugger V2 only reads debugger data that sits directly in `--logdir`.

## Reproducibility

- `requirements.txt` pins the exact versions used for the saved run (TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.1.1). CI installs the same pinned versions on Python 3.11 and 3.13.
- All randomness is seeded (`keras.utils.set_random_seed(42)`, fixed seeds for the sample, the split and the folds), and the debugger run resets the seed before building its model.
- `tf.config.experimental.enable_op_determinism()` makes TensorFlow use deterministic kernels. Without it, multi-threaded CPU maths can add numbers in a different order on each run: two CI runs of the same code gave test macro F1 0.6864 and 0.7137. With it, I ran the notebook twice and every number (all 15 folds, the CV table, the test report) came out identical.
- Determinism holds on the same platform. A different CPU, OS or Python version can still change the later decimal places, so a Colab or CI run will not match the saved numbers digit for digit, but it picks the same config.
- Colab uses its own preinstalled packages instead of `requirements.txt`. At the time of writing it has TensorFlow 2.21.0, the same version pinned here.

To run the tests locally:

```bash
pip install -r requirements.txt
pytest -v tests/
```

## Results

Numbers from the run saved in the notebook (TensorFlow 2.21, seed 42), so the same numbers are in sections 3 to 5. A run on another platform can differ in the second or third decimal place (see Reproducibility above).

**Cross-validation (3 folds, training set only)**

| Config | Class weights | CV macro F1 (mean ± std) | CV accuracy | Epochs run (mean) |
|---|---|---|---|---|
| **Adam, lr 0.01** | **none** | **0.6708 ± 0.0098** | **0.7958** | **33.7** |
| Adam, lr 0.01 | sqrt | 0.6478 ± 0.0098 | 0.7614 | 28.0 |
| Adam, lr 0.01 | balanced | 0.5563 ± 0.0217 | 0.6683 | 24.3 |
| Adam, lr 0.001 | none | 0.5486 ± 0.0067 | 0.7584 | 56.0 |
| SGD, lr 0.01 | none | 0.3729 ± 0.0153 | 0.6996 | 70.0 |

Early stopping ended all 15 runs before the 100-epoch limit, so every run trained until validation loss stopped improving.
Adam with lr 0.01 and no class weights was picked by CV macro F1 and retrained on the full training set for 31 epochs (its mean best epoch in CV).

**Did class weights help?** No, not here. On the same model and folds, `balanced` weights dropped CV macro F1 from 0.6708 to 0.5563, and the milder `sqrt` weights gave 0.6478.
With `balanced`, Cottonwood/Willow gets weight 30.48 and Lodgepole Pine 0.29, so one Cottonwood/Willow row counts about 105 times as much (about 10 times with `sqrt`). The model then predicts the rare classes too often. That helps their recall but costs much more precision on the common classes, and macro F1 counts both.
The `balanced` drop (0.115, more than five fold standard deviations) is clear. The `sqrt` drop (0.023) is about two standard deviations, but with only 3 folds that is a weak signal, so the fair reading is that `sqrt` weights gave no benefit here.
What actually helped the rare classes was training long enough (see below).

**Final test set (4,000 rows, used once)**

| Metric | This version (early stopping) | Earlier version (fixed 25 epochs) |
|---|---|---|
| Macro F1 | **0.6916** | 0.6577 |
| Accuracy | **0.8093** | 0.8067 |
| Cottonwood/Willow F1 (19 rows) | **0.57** | 0.33 |
| Aspen F1 (65 rows) | 0.50 | 0.53 |
| Douglas-fir F1 (120 rows) | 0.55 | 0.53 |

The earlier version stopped every run at a fixed 25 epochs while validation loss was still falling, so the models were undertrained. Letting early stopping decide raised test macro F1 by 0.034, almost all of it from Cottonwood/Willow (0.33 to 0.57). Aspen got slightly worse (0.53 to 0.50), so it is still the weakest class: the confusion matrix shows most Aspen rows being predicted as Lodgepole Pine.
The test macro F1 (0.6916) is close to the CV estimate (0.6708), so the CV score was an honest guide.

**Scalars (Time Series):** loss and accuracy for fold 1 of every config. Runs end at different epochs because early stopping ends each one on its own

![Scalars](screenshots/scalars.png)

**Custom Scalars:** my layout with two groups and four charts, shown for the chosen config (fold 1). Validation loss flattens while training loss keeps falling, which is when early stopping ends the run

![Custom Scalars](screenshots/custom_scalars.png)

**Histograms:** weights of the final model's three dense layers over its 31 training epochs

![Histograms](screenshots/histograms.png)

**Graphs:** the conceptual graph of the final model (Input > Dense 128 > Dropout > Dense 64 > Dense 7). Keras 3's TensorBoard callback writes only this Keras-level graph, so pick **Conceptual graph** in the Graphs tab; the Op graph option stays empty

![Graphs](screenshots/graphs.png)

**HParams:** CV scores of the five configs, with optimizer, learning rate and class weights as hyperparameters

![HParams](screenshots/hparams.png)

**Images:** test-set confusion matrix of the chosen model

![Confusion matrix](screenshots/confusion_matrix.png)

**Debugger V2:** the broken run (unscaled features, SGD lr 0.5) turns the loss into NaN in the first epoch. Debugger V2 flags 1000 NaN/∞ alerts and jumps to the first one: the first NaN values appear inside the cross-entropy loss (`SparseSoftmaxCrossEntropyWithLogits`, graph execution 33), so the problem starts in the loss of the very first training step

![Debugger](screenshots/debugger.png)

## What I learned

- TensorBoard is more than a loss plot. Scalars, Custom Scalars, histograms, HParams, images and the debugger each answer a different question about a training run.
- A fixed epoch count can quietly undertrain a model. When the best epoch is always the last one, the limit is too low. Early stopping on validation loss fixed this and raised test macro F1 from 0.66 to 0.69.
- Class weights are not an automatic fix for unbalanced data. Full `balanced` weights hurt macro F1 here, and it took a controlled comparison (same model, same folds, only the weights changed) to see that.
- Picking a model by its test score makes the test score look better than it really is. Cross-validation on the training data gives a fair way to choose, and the test set then gives an honest final number.
- Unscaled inputs plus a high learning rate can break training quickly. Debugger V2 shows where the NaN values start, which is much faster than guessing. With learning rate 0.1 the loss only spiked and then recovered, so I had to go to 0.5 before there was anything for the debugger to catch.
- Seeds alone do not make a TensorFlow run repeatable. Multi-threaded CPU kernels can still change the last digits, and that is enough to change which epoch early stopping picks. Turning on op determinism made reruns identical.
- Moving helper code out of the notebook made it possible to unit test it and run those tests in CI.

## Files

- `Lab4_TensorBoard.ipynb` - the notebook (run in Colab)
- `lab_utils.py` - data, model, class weights, Custom Scalars layout, selection and plotting helpers
- `tests/test_lab_utils.py` - unit tests
- `requirements.txt` - pinned Python package versions
- `.gitignore` - keeps TensorBoard logs, debugger dumps, caches and local notebook outputs out of the repo
- `.github/workflows/tests.yml` - runs the unit tests on Python 3.11 and 3.13 (the version Colab uses) on every push
- `.github/workflows/notebook.yml` - runs the whole notebook top to bottom on Python 3.13 and checks the results, early stopping, the NaN debugger run and the in-notebook tests; the executed notebook and TensorBoard logs are saved as a download
- `screenshots/` - TensorBoard screenshots from my run (the numbers above come from the same run)
