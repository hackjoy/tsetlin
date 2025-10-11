from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tsetlin import PyModel, PyConfig

iris = load_iris()
X = iris["data"]  # shape: (150, 4)
y = iris["target"]  # shape: (150,) with values 0,1,2
feature_names = iris["feature_names"]
target_names = iris["target_names"]

N_BINS = 4


# One hot encode scalars
def one_hot_binarise(X, num_bins):
    # X: shape (n_samples, n_features), continuous values
    n, f = X.shape
    # compute thresholds for each feature
    thresholds = [
        np.percentile(X[:, j], [100 * i / num_bins for i in range(1, num_bins)])
        for j in range(f)
    ]
    out = []
    for i in range(n):
        sample = X[i]
        one_hot_bits = []
        for j in range(f):
            val = sample[j]
            # find bin index
            b = np.digitize(val, thresholds[j])
            # create one-hot for that feature
            vec = [0] * num_bins
            vec[b] = 1
            one_hot_bits.extend(vec)
        out.append(one_hot_bits)
    return np.array(out, dtype=int)


X_bin = one_hot_binarise(X, N_BINS)

# Build dataset: [x1, x2, ..., xn, y_class_index]
dataset = [list(x) + [label] for x, label in zip(X_bin, y)]

# Split into train/test
train_data, test_data = train_test_split(
    dataset, test_size=0.2, random_state=2, stratify=[row[-1] for row in dataset]
)

# print("Full dataset counts:", Counter([row[-1] for row in dataset]))
# print("Train counts:", Counter([row[-1] for row in train_data]))
# print("Test counts:", Counter([row[-1] for row in test_data]))

# Create feature/literal per bin, e.g. [¬petal length (cm)_bin0, petal length (cm)_bin0, ...]
x_features = []
for feat in feature_names:
    for b in range(N_BINS):
        x_features.append(f"{feat}_bin{b}")

y_classes = list(target_names)  # ["setosa", "versicolor", "virginica"]

config = PyConfig(
    vote_margin_threshold=6,
    activation_threshold=5,
    memory_max=10,
    memory_min=0,
    epochs=400,
    probability_to_forget=0.8,
    probability_to_memorise=0.2,
    clauses_per_class=12,
    seed=42,
)

model = PyModel(x_features=x_features, y_classes=y_classes, config=config)

# --- Train ---
# print("Rules BEFORE training:")
# print(model.get_rules_state_human())
model.train(train_data)
# print("\nRules AFTER training:")
# print(model.get_rules_state_human())

# --- Human-readable rules ---
print("\nHuman readable rules state:")
print(model.get_rules_state_human())

# Predict on a test sample ---
# sample = test_data[0][:-1]  # Features only
# true_label = test_data[0][-1]
# pred_class, votes, active_clauses = model.predict(sample)
# print("Test sample:", sample)
# print("Actual test label:", y_classes[true_label])
# print(f"Predicted label: {y_classes[pred_class]} (Votes: {votes})")

# --------------  Metrics ---------------------

# Train data actual vs predicted
y_true_train = []
y_pred_train = []
for row in train_data:
    x = row[:-1]
    true = row[-1]
    pred, _, _ = model.predict(x)
    y_true_train.append(true)
    y_pred_train.append(pred)

# Test data actual vs predicted
y_true_test = []
y_pred_test = []
for row in test_data:
    x = row[:-1]
    true = row[-1]
    pred, _, _ = model.predict(x)
    y_true_test.append(true)
    y_pred_test.append(pred)

print("Confusion matrix:")
print(confusion_matrix(y_true_test, y_pred_test))

print("Per-class metrics:")
print(classification_report(y_true_test, y_pred_test, target_names=target_names))

train_acc_sklearn = accuracy_score(y_true_train, y_pred_train)
test_acc_sklearn = accuracy_score(y_true_test, y_pred_test)

print(f"SKLearn Train accuracy: {train_acc_sklearn:.4f}")
print(f"SKLearn Test accuracy:  {test_acc_sklearn:.4f}")
