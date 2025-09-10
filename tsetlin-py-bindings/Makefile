.PHONY: venv
venv:
	uv venv
	uv pip install maturin
	@echo "⚠️ Now activate it with: source .venv/bin/activate"

.PHONY: build-dev-bindings
build-dev-bindings:
	uv run maturin develop -m Cargo.toml

.PHONY: build-release-bindings
build-release-bindings:
	uv run maturin build --release -m Cargo.toml

.PHONY: python-package-publish-test
python-package-publish:-test
	maturin publish --repository testpypi -m Cargo.toml

.PHONY: python-package-publish-prod
python-package-publish-prod:
	maturin publish -m Cargo.toml

.PHONY: clean
clean:
	rm -rf .venv
	rm -rf ./target/wheels/