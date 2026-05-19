// --- Postprocessing helpers ---

// Convert raw logits into a probability distribution using a numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    // Empty logits produce an empty probability vector instead of panicking.
    if logits.is_empty() {
        return Vec::new();
    }
    // Subtract the maximum logit before exponentiation to avoid overflow.
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    // Exponentiate the shifted logits.
    let exps = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect::<Vec<_>>();
    // Sum the exponentials so we can normalize them.
    let sum = exps.iter().copied().sum::<f32>();
    // Divide each exponentiated logit by the sum.
    exps.into_iter().map(|value| value / sum).collect()
}

// Return the index and value of the highest probability.
pub fn argmax(values: &[f32]) -> Option<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

// Translate a class index into a human-readable label, falling back to a synthetic label when needed.
pub fn decode_label(labels: &[String], index: usize) -> String {
    labels
        .get(index)
        .cloned()
        .unwrap_or_else(|| format!("LABEL_{index}"))
}

#[cfg(test)]
mod tests {
    // --- Local imports ---
    use super::{argmax, softmax};

    #[test]
    fn test_softmax_sums_to_one() {
        // WHAT: Softmax output forms a probability distribution that sums to one.
        // WHY: Confidence scores exposed to clients must be numerically meaningful.
        let probabilities = softmax(&[1.0, 2.0, 3.0]);
        let sum = probabilities.iter().copied().sum::<f32>();
        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_argmax_returns_highest_index() {
        // WHAT: `argmax` selects the index of the highest value.
        // WHY: Classification labels depend on choosing the top-scoring class correctly.
        let result = argmax(&[0.1, 0.7, 0.2]).expect("a non-empty list should have a maximum");
        assert_eq!(result.0, 1);
    }
}