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

.PHONY: clean
clean:
	rm -rf .venv
	rm -rf ../tsetlin-py-bindings/target/wheels/

EXAMPLE ?= binary_car
# make build_and_test EXAMPLE=classification/binary_car.py
.PHONY: build_and_test
build_and_test:
	@echo "Building and testing example: $(EXAMPLE)"
	make install-dev-bindings
	make example_$(EXAMPLE)

# make example_binary_car
.PHONY: example_%
example_%:
	uv run python classification/$*.py