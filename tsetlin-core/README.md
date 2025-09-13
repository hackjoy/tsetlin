# tsetlin-core

`tsetlin-core` is a Rust implementation of the [Tsetlin machine](https://arxiv.org/abs/1804.01508), a rule-based machine learning algorithm originally introduced by Ole-Christoffer Granmo.

### Features
- [x] Training + inference API
- [x] Binary classification
- [x] Multiclass classification

### Roadmap

- Parallel clause updates using rayon
- Efficient clause representation (bitvec, bitmaps)
- Reduce memory allocations
- Real world dataset examples.
- Review API/codebase as a whole
- More tests/modular code