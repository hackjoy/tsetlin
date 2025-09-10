# tsetlin

Rust implementation of the [Tsetlin Machine](https://arxiv.org/abs/1804.01508), a rule-based machine learning algorithm introduced by **Ole-Christoffer Granmo**.

This implementation was inspired by both the original [research paper](https://arxiv.org/abs/1804.01508) and the [official book](https://tsetlinmachine.org).

- 📦 `tsetlin-core`: Pure Rust implementation
- 🐍 `tsetlin-py-bindings`: Python bindings (via PyO3 + maturin)
- 🧪 `tsetlin-py-examples`: Python examples using the bindings

## License

Licensed under either of:

- MIT License (see [LICENSE-MIT](./LICENSE-MIT))
- Apache License, Version 2.0 (see [LICENSE-APACHE](./LICENSE-APACHE))

at your option.

## Attribution

This project is based on the Tsetlin Machine described in:

- **Granmo, Ole-Christoffer.**
_The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic._ arXiv preprint arXiv:1804.01508 (2018).  
  [https://arxiv.org/abs/1804.01508](https://arxiv.org/abs/1804.01508)

- **Granmo, Ole-Christoffer.**
_An Introduction to Tsetlin Machines._ Available at [https://tsetlinmachine.org](https://tsetlinmachine.org)