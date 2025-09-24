from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from tsetlin import PyModel, PyConfig

# Load Iris dataset
iris = load_iris()
X = iris["data"]  # shape: (150, 4)
y = iris["target"]  # shape: (150,) with values 0,1,2
feature_names = iris["feature_names"]
target_names = iris["target_names"]


# --- Preprocess: Binarize Features ---
def binarize_features(X, num_bins=3):
    result = []
    for col in X.T:  # iterate over features
        thresholds = np.percentile(
            col, [100 * i / num_bins for i in range(1, num_bins)]
        )
        bins = np.digitize(col, thresholds)
        result.append(bins)
    return np.array(result).T


X_bin = binarize_features(X, num_bins=3)

# Convert each feature into binary (above/below column mean)
# X_bin = (X > np.median(X, axis=0)).astype(int)
# print("Feature column sums:", np.sum(X_bin, axis=0))

# Build dataset: [x1, x2, ..., xn, y_class_index]
dataset = [list(x) + [label] for x, label in zip(X_bin, y)]

# Split into train/test
train_data, test_data = train_test_split(
    dataset, test_size=0.2, random_state=2, stratify=[row[-1] for row in dataset]
)

# print("Full dataset counts:", Counter([row[-1] for row in dataset]))
# print("Train counts:", Counter([row[-1] for row in train_data]))
# print("Test counts:", Counter([row[-1] for row in test_data]))

# Extract cleaned feature names
x_features = [f.replace(" (cm)", "") for f in feature_names]
y_classes = list(target_names)  # ["setosa", "versicolor", "virginica"]

# --- Configure the model ---
config = PyConfig(
    vote_margin_threshold=3,
    activation_threshold=2,
    memory_max=10,
    memory_min=0,
    epochs=400,
    probability_to_forget=0.8,
    probability_to_memorise=0.2,
    clauses_per_class=10,  # 80 per class = 240 total
    seed=42,
)

model = PyModel(x_features=x_features, y_classes=y_classes, config=config)

# --- Train ---
# print("Rules BEFORE training:")
# print(model.get_rules_state_human())
model.train(train_data)
# print("\nRules AFTER training:")
# print(model.get_rules_state_human())

# --- Evaluate ---
train_acc = model.evaluate_accuracy(train_data)
test_acc = model.evaluate_accuracy(test_data)
print(f"Training accuracy: {train_acc:.2f}")
print(f"Test accuracy:     {test_acc:.2f}")

# --- Human-readable rules ---
# print("\nHuman-readable rules:")
# print(model.get_rules_state_human())

# --- Predict on a test sample ---
sample = test_data[0][:-1]  # Features only
true_label = test_data[0][-1]

pred_class, votes, active_clauses = model.predict(sample)

print("\nTest sample:", sample)
print("Actual test label:", y_classes[true_label])
print(f"Predicted label: {y_classes[pred_class]} (Votes: {votes})")
