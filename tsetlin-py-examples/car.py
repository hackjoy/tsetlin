from tsetlin import PyModel, PyConfig

config = PyConfig(
    vote_margin_threshold=4,
    activation_threshold=5,
    memory_max=10,
    memory_min=0,
    epochs=10000,
    probability_to_forget=0.8,
    probability_to_memorise=0.2,
    clauses_per_class=8,
    seed=1,
)

y_classes = ["¬CAR", "CAR"]
x_features = ["four_wheels", "transports_people", "wings", "yellow", "blue"]
model = PyModel(
    x_features=x_features,
    y_classes=y_classes,
    config=config,
)

dataset = [
    (1, 1, 0, 0, 1, 1),  # CAR
    (1, 1, 0, 1, 0, 1),  # CAR
    (1, 1, 0, 0, 0, 1),  # CAR
    (1, 1, 0, 0, 0, 1),  # CAR
    (1, 1, 1, 0, 1, 0),  # ¬CAR
    (1, 0, 1, 1, 0, 0),  # ¬CAR
    (1, 0, 1, 1, 0, 0),  # ¬CAR
    (0, 1, 1, 1, 1, 0),  # ¬CAR
]

model.train(dataset)

print(f"Full State: {model.get_rules_state()}\n")
print(f"Learned Rules: {model.get_rules_state_human()}\n")
print(f"Accuracy: {model.evaluate_accuracy(dataset)}\n")

x_1 = [0, 0, 1, 0, 0]  # has wings only (a ¬CAR example)
y_1 = "¬CAR"

pred_class, votes_per_class, activated_clauses = model.predict(x_1)

print(f"Prediction: {y_classes[pred_class]}")
print(f"True Class: {y_1}")
