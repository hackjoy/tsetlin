from tsetlin import PyModel, PyConfig

config = PyConfig(
    vote_margin_threshold=2,
    activation_threshold=5,
    memory_max=10,
    memory_min=0,
    epochs=10000,
    probability_to_forget=0.9,
    probability_to_memorise=0.1,
    clauses_per_class=4,
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
print(model.get_rules_state())
print(model.get_rules_state_human())
print(model.evaluate_accuracy(dataset))

x_1 = [0, 0, 1, 0, 0, 0]  # has wings and ¬CAR
pred_class, votes_per_class, activated_clauses = model.predict(x_1)
print(f"Prediction: {y_classes[pred_class]}")
