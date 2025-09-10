use ::tsetlin::{Config, Model};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyConfig {
    #[pyo3(get, set)]
    pub vote_margin_threshold: i32,
    #[pyo3(get, set)]
    pub activation_threshold: i32,
    #[pyo3(get, set)]
    pub memory_max: i32,
    #[pyo3(get, set)]
    pub memory_min: i32,
    #[pyo3(get, set)]
    pub epochs: usize,
    #[pyo3(get, set)]
    pub probability_to_forget: f64,
    #[pyo3(get, set)]
    pub probability_to_memorise: f64,
    #[pyo3(get, set)]
    pub clauses_per_class: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (
        vote_margin_threshold = 2,
        activation_threshold = 5,
        memory_max = 10,
        memory_min = 0,
        epochs = 10,
        probability_to_forget = 0.9,
        probability_to_memorise = 0.1,
        clauses_per_class = 4,
        seed = None
    ))]
    fn new(
        vote_margin_threshold: i32,
        activation_threshold: i32,
        memory_max: i32,
        memory_min: i32,
        epochs: usize,
        probability_to_forget: f64,
        probability_to_memorise: f64,
        clauses_per_class: usize,
        seed: Option<u64>,
    ) -> Self {
        PyConfig {
            vote_margin_threshold,
            activation_threshold,
            memory_max,
            memory_min,
            epochs,
            probability_to_forget,
            probability_to_memorise,
            clauses_per_class,
            seed,
        }
    }
}

#[pyclass]
pub struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(x_features: Vec<String>, y_classes: Vec<String>, config: PyConfig) -> Self {
        let internal_config = Config {
            vote_margin_threshold: config.vote_margin_threshold,
            activation_threshold: config.activation_threshold,
            memory_max: config.memory_max,
            memory_min: config.memory_min,
            epochs: config.epochs,
            probability_to_forget: config.probability_to_forget,
            probability_to_memorise: config.probability_to_memorise,
            clauses_per_class: config.clauses_per_class,
            seed: config.seed,
        };

        Self {
            inner: Model::new(x_features, y_classes, internal_config),
        }
    }

    fn train(&mut self, dataset: Vec<Vec<i32>>) {
        self.inner.train(&dataset);
    }

    fn predict(&self, example: Vec<i32>) -> (usize, Vec<i32>, Vec<Vec<usize>>) {
        self.inner.predict(&example)
    }

    fn get_rules_state(&self) -> Vec<Vec<Vec<(i32, i32)>>> {
        self.inner.get_rules_state()
    }

    fn get_rules_state_human(&self) -> String {
        match serde_json::to_string(&self.inner.get_rules_state_human()) {
            Ok(s) => s,
            Err(_) => "<failed to serialize rules>".to_string(),
        }
    }

    fn evaluate_accuracy(&self, dataset: Vec<Vec<i32>>) -> f64 {
        self.inner.evaluate_accuracy(&dataset)
    }
}

/// Python module definition
#[pymodule]
fn tsetlin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyConfig>()?;
    Ok(())
}
