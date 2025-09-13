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

y_classes = ["CAR", "PLANE", "BOAT"]
x_features = ["four_wheels", "transports_people", "wings", "yellow", "blue"]

model = PyModel(x_features=x_features, y_classes=y_classes, config=config)

dataset = [
    # four_wheels, transports_people, wings, yellow, blue, y_class_index
    (1, 1, 0, 0, 1, 0),  # CAR
    (1, 1, 0, 1, 0, 0),  # CAR
    (1, 0, 1, 0, 0, 1),  # PLANE
    (0, 0, 1, 0, 1, 1),  # PLANE
    (0, 0, 0, 1, 1, 2),  # BOAT
    (0, 0, 0, 1, 0, 2),  # BOAT
]

model.train(dataset)

print(f"Learned Rules: {model.get_rules_state_human()}\n")
print(f"Accuracy: {model.evaluate_accuracy(dataset)}\n")

x_1 = [0, 0, 1, 0, 0]  # has wings only (PLANE)
pred_class, votes_per_class, activated_clauses = model.predict(x_1)

print(f"Prediction: {y_classes[pred_class]}")
print(f"Classes: {y_classes}")
print(f"Votes per class: {votes_per_class}")
