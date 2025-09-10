.PHONY: venv
venv:
	uv venv
	uv pip install -r requirements.dev.txt
	@echo "✅ .venv created. Please run: source .venv/bin/activate"

.PHONY: install-dev-bindings
install-dev-bindings:
	uv run maturin develop -m ../tsetlin-py-bindings/Cargo.toml
	
.PHONY: install-release-bindings
install-release-bindings:
	uv pip install ../tsetlin-py-bindings/target/wheels/tsetlin-*.whl

.PHONY: dev
dev: venv install-dev-bindings

.PHONY: test
test:
	uv run pytest tests/

.PHONY: example_car
example_car:
	uv run python car.py

.PHONY: clean
clean:
	rm -rf .venv
	rm -rf ../tsetlin-py-bindings/target/wheels/


.PHONY: build_and_test_car
build_and_test_car: install-dev-bindings example_car
