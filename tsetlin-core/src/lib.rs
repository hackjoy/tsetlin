use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Literal {
    pub pos: i32,
    pub neg: i32,
}

impl Literal {
    pub fn new(initial: i32) -> Self {
        Self {
            pos: initial,
            neg: initial,
        }
    }
}

/// Configuration parameters for the Tsetlin Machine Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub seed: Option<u64>,
    /// A high vote_margin (above vote_margin_threshold) means many rules are
    /// predicting correctly for the class, so there is less need to recognise_feedback
    /// vote_margin_threshold then helps constrain this update mechanism so it does not
    /// reinforce what is already strong and encourages more diverse rules to be learned.
    pub vote_margin_threshold: i32,
    /// Literals are active in clause above this value.
    pub activation_threshold: i32,
    /// Upper limit on a Literal's memorisation state.
    pub memory_max: i32,
    /// Lower limit on a Literal's memorisation state.
    pub memory_min: i32,
    /// Number of iterations through the full training dataset.
    pub epochs: usize,
    pub probability_to_forget: f64,
    pub probability_to_memorise: f64,
    pub clauses_per_class: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub x_features: Vec<String>,
    pub y_classes: Vec<String>,
    pub label_index: usize,
    pub config: Config,
    /// rules_state contains vectors of Tsetlin automata for each y_class
    /// Each feature is represented by a Literal with a (positive, negated) state value that is
    /// adjusted during training. They have a memory_min (e.g. 0) and a memory_max (e.g. 10).
    /// Only values above the activation_threshold (e.g. 5) become active in
    /// the learned clause and then used when predicting (voting) on new input data.
    ///
    /// It can be indexed into as: [class][rule][feature]
    /// Lets see a binary classification example, predicting ¬CAR vs CAR, with two features ["four_wheels", "wings"]
    /// [
    ///   [                           ** The 1st class (i.e. ¬CAR) - these rules vote for this class if the condition passes
    ///     [ (5, 0), (0, 9) ]        **   This vector represents 1 set of Tsetlin automata.
    ///     [ (2, 0), (0, 4) ]        **   A 2nd Tsetlin automata predicting ¬CAR
    ///   ],
    ///   [                           ** The 2nd class (i.e. CAR)
    ///     [ (8, 2), (4, 0) ]        **   Tsetlin automata for the 2nd class
    ///     [ (4, 1), (0, 8) ].       **   The second feature here (0,8) relates to "wings", the state
    ///   ],                          **   value is 0 for the positive literal ("wings") and 8 for the negation ("¬wings")
    ///                               **   meaning NOT wings is highly predicitive of CAR.
    /// ]
    pub rules_state: Vec<Vec<Vec<Literal>>>,
}

impl Model {
    /// Create a new model given feature names, class names, and config
    pub fn new(x_features: Vec<String>, y_classes: Vec<String>, config: Config) -> Self {
        let label_index = x_features.len();
        let mut rules_state = vec![];
        for _ in 0..y_classes.len() {
            let mut class_rules = vec![];
            for _ in 0..config.clauses_per_class {
                class_rules.push(
                    x_features
                        .iter()
                        .map(|_| Literal::new(config.activation_threshold))
                        .collect(),
                );
            }
            rules_state.push(class_rules);
        }

        Self {
            x_features,
            y_classes,
            label_index,
            config,
            rules_state,
        }
    }

    pub fn get_rules_state(&self) -> Vec<Vec<Vec<(i32, i32)>>> {
        self.rules_state
            .iter()
            .map(|class_rules| {
                class_rules
                    .iter()
                    .map(|rule| rule.iter().map(|lit| (lit.pos, lit.neg)).collect())
                    .collect()
            })
            .collect()
    }

    /// Returns a human-readable view of the rules
    pub fn get_rules_state_human(&self) -> HashMap<String, Vec<HashMap<String, i32>>> {
        let threshold = self.config.activation_threshold;
        let mut result = HashMap::new();

        for (class_idx, class_rules) in self.rules_state.iter().enumerate() {
            let class_label = &self.y_classes[class_idx];
            let mut readable_rules = vec![];

            for rule in class_rules {
                let mut readable_rule = HashMap::new();

                for (feat_idx, literal) in rule.iter().enumerate() {
                    let feat_name = &self.x_features[feat_idx];

                    if literal.pos > threshold {
                        readable_rule.insert(feat_name.clone(), literal.pos);
                    }

                    if literal.neg > threshold {
                        readable_rule.insert(format!("¬{}", feat_name), literal.neg);
                    }
                }

                readable_rules.push(readable_rule);
            }

            result.insert(class_label.clone(), readable_rules);
        }

        result
    }

    pub fn evaluate_accuracy(&self, dataset: &[Vec<i32>]) -> f64 {
        let mut correct = 0;
        for example in dataset {
            let label = example[self.label_index] as usize;
            let features = &example[..self.label_index];
            let (predicted_class, _, _) = self.predict(features);
            if predicted_class == label {
                correct += 1;
            }
        }
        correct as f64 / dataset.len() as f64
    }

    /// Training function which ultimately learns a set of rules (rules_state) for prediction.
    /// For each example in the dataset, the current rules_state is evaluated to determine the vote_margin
    /// for the correct class. This then determines a threshold of how likely a rule is to be updated
    /// when iterating over the rules and applying feedback of the current example to each rule,
    /// according to how it evaluated. e.g. a rule that evaluated true for the correct class should recognise_feedback()
    pub fn train(&mut self, training_data: &[Vec<i32>]) {
        let pb = ProgressBar::new(self.config.epochs as u64);
        let mut rng = if let Some(seed) = self.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(rand::random::<u64>())
        };
        let probability_to_forget = self.config.probability_to_forget;
        let probability_to_memorise = self.config.probability_to_memorise;
        let memory_min = self.config.memory_min;
        let memory_max = self.config.memory_max;
        let activation_threshold = self.config.activation_threshold;
        let vote_margin_threshold = self.config.vote_margin_threshold;

        for epoch in 0..self.config.epochs {
            pb.set_message(format!("Epoch {}/{}", epoch + 1, self.config.epochs));
            pb.inc(1);

            for example in training_data.iter() {
                let true_class = example[self.label_index] as usize;
                let _num_classes = self.rules_state.len();

                // Calculate votes for each class given this example
                let mut votes_per_class = vec![0; self.rules_state.len()];
                for (class_idx, rules) in self.rules_state.iter().enumerate() {
                    for rule in rules {
                        if Self::evaluate_rule(activation_threshold, rule, example, true) {
                            votes_per_class[class_idx] += 1;
                        }
                    }
                }

                // Calculate which is the next most voted class compared to the true_class
                let mut top_competitor_votes = i32::MIN;
                for (idx, &votes) in votes_per_class.iter().enumerate() {
                    if idx != true_class && votes > top_competitor_votes {
                        top_competitor_votes = votes;
                    }
                }

                // Compute vote_margin and constrain by threshold, often written as "v"
                let abs_vote_margin = votes_per_class[true_class] - top_competitor_votes;
                let vote_margin =
                    abs_vote_margin.clamp(-vote_margin_threshold, vote_margin_threshold); // Constrain vote_margin within threshold
                // Higher vote_margin, will generate a lower feedback_threshold, lowering the chance of rule updates
                // Lower vote_margin, will generate a higher feedback_threshold, increasing the chance of rule updates
                let feedback_threshold = (vote_margin_threshold - vote_margin) as f64
                    / (2.0 * vote_margin_threshold as f64);

                // Update rules
                for (class_idx, rules) in self.rules_state.iter_mut().enumerate() {
                    for rule in rules.iter_mut() {
                        let should_update_rule = rng.random::<f64>() <= feedback_threshold;
                        if !should_update_rule {
                            continue;
                        }

                        let condition_evals_true =
                            Self::evaluate_rule(activation_threshold, rule, example, true);

                        if class_idx == true_class {
                            // Condition evaluated to true and correct
                            if condition_evals_true {
                                Self::recognise_feedback(
                                    rule,
                                    example,
                                    &mut rng,
                                    probability_to_forget,
                                    probability_to_memorise,
                                    memory_min,
                                    memory_max,
                                );
                            } else {
                                // Condition evaluated to false and incorrect
                                Self::erase_feedback(
                                    rule,
                                    &mut rng,
                                    probability_to_forget,
                                    memory_min,
                                    memory_max,
                                );
                            }
                        // Condition evaluated to true and incorrect
                        } else if condition_evals_true {
                            Self::reject_feedback(
                                rule,
                                example,
                                memory_min,
                                memory_max,
                                activation_threshold,
                            );
                        }
                        // Condition false and "correct" is a no-op.
                    }
                }
            }
        }
        pb.finish_with_message("Training complete ✅");
    }

    /// Predict the class of a new example
    pub fn predict(&self, example: &[i32]) -> (usize, Vec<i32>, Vec<Vec<usize>>) {
        let activation_threshold = self.config.activation_threshold;
        let mut votes_per_class = vec![0; self.rules_state.len()];
        let mut activated_clauses = vec![vec![]; self.rules_state.len()];

        for (class_idx, rules) in self.rules_state.iter().enumerate() {
            for (rule_idx, rule) in rules.iter().enumerate() {
                if Self::evaluate_rule(activation_threshold, rule, example, false) {
                    votes_per_class[class_idx] += 1;
                    activated_clauses[class_idx].push(rule_idx);
                }
            }
        }

        let predicted_class = votes_per_class
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        (predicted_class, votes_per_class, activated_clauses)
    }

    /// Evaluate whether a rule's active condition is valid
    /// for the current example
    fn evaluate_rule(
        activation_threshold: i32,
        rule: &[Literal],
        example: &[i32],
        // Rules without any literals memorised eval true at training time, false at prediction time.
        model_is_training: bool,
    ) -> bool {
        let mut any_literal_memorised = false;
        for (feat_idx, literal) in rule.iter().enumerate() {
            let observed_val = example[feat_idx];
            if literal.pos > activation_threshold {
                any_literal_memorised = true;
                if observed_val == 0 {
                    return false;
                }
            }
            if literal.neg > activation_threshold {
                any_literal_memorised = true;
                if observed_val == 1 {
                    return false;
                }
            }
        }
        if !any_literal_memorised && model_is_training {
            return true;
        }
        any_literal_memorised
    }

    /// Probabilistically reinforces Literal memory state (postively or negatively)
    /// using the observed training data example
    fn recognise_feedback<R: Rng>(
        rule: &mut [Literal],
        example: &[i32],
        rng: &mut R,
        probability_to_forget: f64,
        probability_to_memorise: f64,
        memory_min: i32,
        memory_max: i32,
    ) {
        for (feat_idx, literal) in rule.iter_mut().enumerate() {
            let observed_val = example[feat_idx];
            let memorise = rng.random::<f64>() < probability_to_memorise;
            let forget = rng.random::<f64>() < probability_to_forget;

            if observed_val == 1 {
                if memorise {
                    literal.pos = (literal.pos + 1).clamp(memory_min, memory_max);
                }
                if forget {
                    literal.neg = (literal.neg - 1).clamp(memory_min, memory_max);
                }
            } else {
                if forget {
                    literal.pos = (literal.pos - 1).clamp(memory_min, memory_max);
                }
                if memorise {
                    literal.neg = (literal.neg + 1).clamp(memory_min, memory_max);
                }
            }
        }
    }

    /// Rule did not evaluate true, therefore probabilistically weaken memory
    /// for positive and negative Literals.
    fn erase_feedback<R: Rng>(
        rule: &mut [Literal],
        rng: &mut R,
        probability_to_forget: f64,
        memory_min: i32,
        memory_max: i32,
    ) {
        for literal in rule.iter_mut() {
            if rng.random::<f64>() < probability_to_forget {
                literal.pos = (literal.pos - 1).clamp(memory_min, memory_max);
            }
            if rng.random::<f64>() < probability_to_forget {
                literal.neg = (literal.neg - 1).clamp(memory_min, memory_max);
            }
        }
    }

    /// Condition evaluated true but was not correct, a contradiction.
    /// Update the Literals in this rule so that we are less likely to
    /// evaluate to true next time.
    fn reject_feedback(
        rule: &mut [Literal],
        example: &[i32],
        memory_min: i32,
        memory_max: i32,
        activation_threshold: i32,
    ) {
        for (feat_idx, literal) in rule.iter_mut().enumerate() {
            let x = example[feat_idx];
            // Feature was false, boost positive literal if not memorised
            // so rule less likely to eval true next time.
            if x == 0 && literal.pos < activation_threshold {
                literal.pos = (literal.pos + 1).clamp(memory_min, memory_max);
            // Feature was true, boost negative literal if not memorised
            // so rule less likely to eval true next time.
            } else if x == 1 && literal.neg < activation_threshold {
                literal.neg = (literal.neg + 1).clamp(memory_min, memory_max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_training_and_prediction() {
        let features = vec![
            "four_wheels".into(),
            "transports_people".into(),
            "wings".into(),
            "yellow".into(),
            "blue".into(),
        ];
        let classes = vec!["¬CAR".into(), "CAR".into()];

        let config = Config {
            vote_margin_threshold: 2,
            activation_threshold: 5,
            memory_max: 10,
            memory_min: 0,
            epochs: 10,
            probability_to_forget: 0.8,
            probability_to_memorise: 0.2,
            clauses_per_class: 4,
            seed: Some(1),
        };

        let mut model = Model::new(features, classes, config);

        let dataset = vec![
            vec![1, 1, 0, 0, 0, 1],
            vec![1, 1, 0, 1, 0, 1],
            vec![0, 1, 1, 1, 1, 0],
            vec![1, 1, 1, 0, 1, 0],
        ];

        model.train(&dataset);

        let (pred, votes, active) = model.predict(&[0, 0, 1, 0, 1]); // Wings and Blue
        println!(
            "Prediction: {:?}, Votes: {:?}, Active: {:?}",
            pred, votes, active
        );
        assert_eq!(pred, 0);
    }
}
