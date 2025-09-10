.PHONY: build
build:
	cargo build

.PHONY: test
test:
	cargo test

.PHONY: test-tsetlin-py-bindings
test-tsetlin-py-bindings:
	cargo test -p tsetlin-py-bindings

.PHONY: test-tsetlin-core
test-tsetlin-core:
	cargo test -p tsetlin-core

.PHONY: docs
docs:
	cargo doc --open
