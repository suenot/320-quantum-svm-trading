//! Quantum SVM Trading Library
//!
//! Implements a Quantum Support Vector Machine (QSVM) for market regime classification.
//! The quantum kernel is computed via classical simulation of ZZ feature map circuits.

use ndarray::Array2;
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ─── Quantum Feature Map & Kernel ─────────────────────────────────────────────

/// Compute the quantum state vector produced by the ZZ feature map for input `x`.
/// `x` values should be in [0, pi]. `depth` is the number of repetitions of the
/// feature map circuit.  Returns a complex state vector of length 2^n where n = x.len().
/// Stored as (re, im) pairs in a flat Vec of length 2 * 2^n.
pub fn zz_feature_map_state(x: &[f64], depth: usize) -> Vec<(f64, f64)> {
    let n = x.len();
    let dim = 1usize << n;

    // Start with |0...0>
    let mut state: Vec<(f64, f64)> = vec![(0.0, 0.0); dim];
    state[0] = (1.0, 0.0);

    for _d in 0..depth {
        // Layer 1: H gate on each qubit, then Rz(x_k) on qubit k
        // H|0> = (|0>+|1>)/sqrt(2), H|1> = (|0>-|1>)/sqrt(2)
        // Apply H to each qubit sequentially
        for q in 0..n {
            apply_hadamard(&mut state, n, q);
        }
        // Apply Rz(x_k) to each qubit k:  Rz(theta)|b> = exp(-i*theta*(-1)^b / 2)|b>
        // We use the convention Rz(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
        for q in 0..n {
            apply_rz(&mut state, n, q, x[q]);
        }

        // Layer 2: Entangling ZZ interactions for each pair (j, k)
        // RZZ(theta)|b_j, b_k> = exp(-i * theta * (-1)^(b_j XOR b_k) / 2) |b_j, b_k>
        // but we use the ZZ feature map convention: theta = (pi - x_j)*(pi - x_k)
        for j in 0..n {
            for k in (j + 1)..n {
                let theta = (PI - x[j]) * (PI - x[k]);
                apply_rzz(&mut state, n, j, k, theta);
            }
        }
    }

    state
}

fn apply_hadamard(state: &mut [(f64, f64)], n_qubits: usize, qubit: usize) {
    let dim = 1usize << n_qubits;
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let step = 1usize << qubit;

    for i in 0..dim {
        if i & step == 0 {
            let j = i | step;
            let a = state[i];
            let b = state[j];
            state[i] = (
                inv_sqrt2 * (a.0 + b.0),
                inv_sqrt2 * (a.1 + b.1),
            );
            state[j] = (
                inv_sqrt2 * (a.0 - b.0),
                inv_sqrt2 * (a.1 - b.1),
            );
        }
    }
}

fn apply_rz(state: &mut [(f64, f64)], n_qubits: usize, qubit: usize, theta: f64) {
    let dim = 1usize << n_qubits;
    let step = 1usize << qubit;
    // Rz(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();

    for i in 0..dim {
        if i & step == 0 {
            // qubit is 0: multiply by exp(-i*theta/2)
            let (re, im) = state[i];
            state[i] = (
                re * cos_half + im * sin_half,
                im * cos_half - re * sin_half,
            );
        } else {
            // qubit is 1: multiply by exp(i*theta/2)
            let (re, im) = state[i];
            state[i] = (
                re * cos_half - im * sin_half,
                im * cos_half + re * sin_half,
            );
        }
    }
}

fn apply_rzz(state: &mut [(f64, f64)], n_qubits: usize, q1: usize, q2: usize, theta: f64) {
    let dim = 1usize << n_qubits;
    let s1 = 1usize << q1;
    let s2 = 1usize << q2;
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();

    for i in 0..dim {
        let b1 = (i & s1 != 0) as u8;
        let b2 = (i & s2 != 0) as u8;
        let parity = b1 ^ b2;
        // ZZ eigenvalue: +1 if parity==0, -1 if parity==1
        // Phase: exp(-i * theta * eigenvalue / 2)
        let (c, s) = if parity == 0 {
            // exp(-i*theta/2)
            (cos_half, -sin_half)
        } else {
            // exp(+i*theta/2)
            (cos_half, sin_half)
        };
        let (re, im) = state[i];
        state[i] = (re * c - im * s, re * s + im * c);
    }
}

/// Compute the quantum kernel value K(x1, x2) = |<phi(x1)|phi(x2)>|^2
pub fn quantum_kernel(x1: &[f64], x2: &[f64], depth: usize) -> f64 {
    let s1 = zz_feature_map_state(x1, depth);
    let s2 = zz_feature_map_state(x2, depth);

    // inner product <s1|s2>
    let mut re = 0.0;
    let mut im = 0.0;
    for (a, b) in s1.iter().zip(s2.iter()) {
        // <a|b> = conj(a) * b
        re += a.0 * b.0 + a.1 * b.1;
        im += a.0 * b.1 - a.1 * b.0;
    }
    // |<s1|s2>|^2
    re * re + im * im
}

/// Compute the full quantum kernel matrix for a dataset.
pub fn quantum_kernel_matrix(data: &[Vec<f64>], depth: usize) -> Array2<f64> {
    let n = data.len();
    let mut kernel = Array2::zeros((n, n));
    for i in 0..n {
        kernel[[i, i]] = 1.0;
        for j in (i + 1)..n {
            let k = quantum_kernel(&data[i], &data[j], depth);
            kernel[[i, j]] = k;
            kernel[[j, i]] = k;
        }
    }
    kernel
}

/// Compute the cross-kernel matrix between train and test data.
pub fn quantum_kernel_matrix_cross(
    train: &[Vec<f64>],
    test: &[Vec<f64>],
    depth: usize,
) -> Array2<f64> {
    let n_train = train.len();
    let n_test = test.len();
    let mut kernel = Array2::zeros((n_test, n_train));
    for i in 0..n_test {
        for j in 0..n_train {
            kernel[[i, j]] = quantum_kernel(&test[i], &train[j], depth);
        }
    }
    kernel
}

// ─── SVM Training (Simplified SMO) ───────────────────────────────────────────

/// Trained QSVM model.
#[derive(Debug, Clone)]
pub struct QSVMModel {
    pub alphas: Vec<f64>,
    pub bias: f64,
    pub labels: Vec<f64>,
    pub support_indices: Vec<usize>,
    pub train_data: Vec<Vec<f64>>,
    pub depth: usize,
    pub c: f64,
}

/// Train a binary QSVM using a simplified SMO algorithm.
/// `labels` should be +1.0 or -1.0.
/// `c` is the regularization parameter.
/// `depth` is the ZZ feature map depth.
/// `max_iter` is the maximum number of SMO iterations.
pub fn train_qsvm(
    data: &[Vec<f64>],
    labels: &[f64],
    c: f64,
    depth: usize,
    max_iter: usize,
) -> QSVMModel {
    let n = data.len();
    assert_eq!(n, labels.len());

    // Compute kernel matrix
    let kernel = quantum_kernel_matrix(data, depth);

    // Initialize alphas to 0
    let mut alphas = vec![0.0f64; n];
    let mut bias = 0.0f64;
    let tol = 1e-5;

    let mut rng = rand::thread_rng();

    for _iter in 0..max_iter {
        let mut num_changed = 0;

        for i in 0..n {
            // Compute f(x_i)
            let mut fi = bias;
            for j in 0..n {
                fi += alphas[j] * labels[j] * kernel[[j, i]];
            }
            let ei = fi - labels[i];

            // Check KKT violation
            let yi_ei = labels[i] * ei;
            if (yi_ei < -tol && alphas[i] < c) || (yi_ei > tol && alphas[i] > 0.0) {
                // Pick random j != i
                let mut j = rng.gen_range(0..n);
                while j == i {
                    j = rng.gen_range(0..n);
                }

                let mut fj = bias;
                for k in 0..n {
                    fj += alphas[k] * labels[k] * kernel[[k, j]];
                }
                let ej = fj - labels[j];

                let alpha_i_old = alphas[i];
                let alpha_j_old = alphas[j];

                // Compute bounds
                let (lo, hi) = if (labels[i] - labels[j]).abs() > 1e-10 {
                    (
                        f64::max(0.0, alphas[j] - alphas[i]),
                        f64::min(c, c + alphas[j] - alphas[i]),
                    )
                } else {
                    (
                        f64::max(0.0, alphas[i] + alphas[j] - c),
                        f64::min(c, alphas[i] + alphas[j]),
                    )
                };

                if (hi - lo).abs() < 1e-10 {
                    continue;
                }

                let eta = 2.0 * kernel[[i, j]] - kernel[[i, i]] - kernel[[j, j]];
                if eta >= 0.0 {
                    continue;
                }

                // Update alpha_j
                alphas[j] -= labels[j] * (ei - ej) / eta;
                alphas[j] = alphas[j].clamp(lo, hi);

                if (alphas[j] - alpha_j_old).abs() < 1e-8 {
                    continue;
                }

                // Update alpha_i
                alphas[i] += labels[i] * labels[j] * (alpha_j_old - alphas[j]);

                // Update bias
                let b1 = bias - ei
                    - labels[i] * (alphas[i] - alpha_i_old) * kernel[[i, i]]
                    - labels[j] * (alphas[j] - alpha_j_old) * kernel[[i, j]];
                let b2 = bias - ej
                    - labels[i] * (alphas[i] - alpha_i_old) * kernel[[i, j]]
                    - labels[j] * (alphas[j] - alpha_j_old) * kernel[[j, j]];

                if alphas[i] > 0.0 && alphas[i] < c {
                    bias = b1;
                } else if alphas[j] > 0.0 && alphas[j] < c {
                    bias = b2;
                } else {
                    bias = (b1 + b2) / 2.0;
                }

                num_changed += 1;
            }
        }

        if num_changed == 0 {
            break;
        }
    }

    // Identify support vectors
    let support_indices: Vec<usize> = alphas
        .iter()
        .enumerate()
        .filter(|(_, &a)| a > 1e-8)
        .map(|(i, _)| i)
        .collect();

    QSVMModel {
        alphas,
        bias,
        labels: labels.to_vec(),
        support_indices,
        train_data: data.to_vec(),
        depth,
        c,
    }
}

/// Predict labels for test data using a trained QSVM model.
pub fn predict_qsvm(model: &QSVMModel, test_data: &[Vec<f64>]) -> Vec<f64> {
    let cross_kernel = quantum_kernel_matrix_cross(&model.train_data, test_data, model.depth);
    let n_test = test_data.len();
    let n_train = model.train_data.len();

    let mut predictions = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let mut f = model.bias;
        for j in 0..n_train {
            f += model.alphas[j] * model.labels[j] * cross_kernel[[i, j]];
        }
        predictions.push(if f >= 0.0 { 1.0 } else { -1.0 });
    }
    predictions
}

/// Predict raw decision values (before sign) for test data.
pub fn predict_qsvm_values(model: &QSVMModel, test_data: &[Vec<f64>]) -> Vec<f64> {
    let cross_kernel = quantum_kernel_matrix_cross(&model.train_data, test_data, model.depth);
    let n_test = test_data.len();
    let n_train = model.train_data.len();

    let mut values = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let mut f = model.bias;
        for j in 0..n_train {
            f += model.alphas[j] * model.labels[j] * cross_kernel[[i, j]];
        }
        values.push(f);
    }
    values
}

// ─── Multi-class QSVM (One-vs-Rest) ──────────────────────────────────────────

/// Multi-class QSVM using One-vs-Rest strategy.
#[derive(Debug, Clone)]
pub struct MultiClassQSVM {
    pub models: Vec<(f64, QSVMModel)>, // (class_label, binary_model)
}

/// Train a multi-class QSVM using One-vs-Rest.
pub fn train_multiclass_qsvm(
    data: &[Vec<f64>],
    labels: &[f64],
    c: f64,
    depth: usize,
    max_iter: usize,
) -> MultiClassQSVM {
    // Find unique classes
    let mut classes: Vec<f64> = labels.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();

    let mut models = Vec::new();

    for &cls in &classes {
        // Create binary labels: +1 for this class, -1 for others
        let binary_labels: Vec<f64> = labels
            .iter()
            .map(|&y| if (y - cls).abs() < 1e-10 { 1.0 } else { -1.0 })
            .collect();

        let model = train_qsvm(data, &binary_labels, c, depth, max_iter);
        models.push((cls, model));
    }

    MultiClassQSVM { models }
}

/// Predict using multi-class QSVM. Returns the class with highest decision value.
pub fn predict_multiclass(mc: &MultiClassQSVM, test_data: &[Vec<f64>]) -> Vec<f64> {
    let n_test = test_data.len();

    // Get decision values from each binary classifier
    let all_values: Vec<Vec<f64>> = mc
        .models
        .iter()
        .map(|(_, model)| predict_qsvm_values(model, test_data))
        .collect();

    let mut predictions = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let mut best_class = mc.models[0].0;
        let mut best_val = all_values[0][i];
        for (idx, (cls, _)) in mc.models.iter().enumerate() {
            if all_values[idx][i] > best_val {
                best_val = all_values[idx][i];
                best_class = *cls;
            }
        }
        predictions.push(best_class);
    }
    predictions
}

// ─── Feature Engineering ──────────────────────────────────────────────────────

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Compute log returns from close prices.
pub fn log_returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute rolling standard deviation (realized volatility) with the given window.
pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    if n < window {
        return vec![0.0; n];
    }
    let mut result = vec![0.0; n];
    for i in (window - 1)..n {
        let slice = &returns[(i + 1 - window)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / window as f64;
        result[i] = var.sqrt();
    }
    result
}

/// Compute RSI-like momentum indicator over a lookback window.
/// Returns values in [0, 1]: 1 = all gains, 0 = all losses.
pub fn rsi_indicator(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    if n < window {
        return vec![0.5; n];
    }
    let mut result = vec![0.5; n];
    for i in (window - 1)..n {
        let slice = &returns[(i + 1 - window)..=i];
        let avg_gain: f64 =
            slice.iter().filter(|&&r| r > 0.0).sum::<f64>() / window as f64;
        let avg_loss: f64 =
            slice.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum::<f64>() / window as f64;
        if avg_gain + avg_loss < 1e-12 {
            result[i] = 0.5;
        } else {
            result[i] = avg_gain / (avg_gain + avg_loss);
        }
    }
    result
}

/// Compute volume ratio: current volume / rolling mean volume.
pub fn volume_ratio(volumes: &[f64], window: usize) -> Vec<f64> {
    let n = volumes.len();
    if n < window {
        return vec![1.0; n];
    }
    let mut result = vec![1.0; n];
    for i in (window - 1)..n {
        let slice = &volumes[(i + 1 - window)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        if mean > 1e-12 {
            result[i] = volumes[i] / mean;
        }
    }
    result
}

/// Compute price position: where close sits in [low, high] range over a window.
/// Returns values in [0, 1].
pub fn price_position(candles: &[Candle], window: usize) -> Vec<f64> {
    let n = candles.len();
    if n < window {
        return vec![0.5; n];
    }
    let mut result = vec![0.5; n];
    for i in (window - 1)..n {
        let slice = &candles[(i + 1 - window)..=i];
        let lo = slice.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let hi = slice
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        if (hi - lo).abs() > 1e-12 {
            result[i] = (candles[i].close - lo) / (hi - lo);
        }
    }
    result
}

/// Engineer features from candle data.
/// Returns (features, valid_start_index) where features[i] corresponds to candles[valid_start_index + i].
/// Each feature vector has 5 elements: [return, volatility, rsi, volume_ratio, price_position].
pub fn engineer_features(candles: &[Candle], window: usize) -> (Vec<Vec<f64>>, usize) {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let rets = log_returns(&closes);
    let vol = rolling_volatility(&rets, window);
    let rsi = rsi_indicator(&rets, window);
    let vr = volume_ratio(&volumes, window);
    let pp = price_position(candles, window);

    // rets has length (n-1), others have length n or n-1
    // Align everything: rets[i] corresponds to candles[i+1]
    // vol, rsi are indexed on rets, vr and pp are indexed on candles
    let start = window; // need at least `window` returns, so start from index `window` in rets
    let n_rets = rets.len();

    let mut features = Vec::new();
    for i in start..n_rets {
        features.push(vec![rets[i], vol[i], rsi[i], vr[i + 1], pp[i + 1]]);
    }

    (features, start + 1) // +1 because rets is offset by 1 from candles
}

/// Normalize features to [0, pi] range.
pub fn normalize_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return vec![];
    }
    let n_features = features[0].len();
    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];

    for row in features {
        for (j, &val) in row.iter().enumerate() {
            if val < mins[j] {
                mins[j] = val;
            }
            if val > maxs[j] {
                maxs[j] = val;
            }
        }
    }

    features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &val)| {
                    let range = maxs[j] - mins[j];
                    if range.abs() < 1e-12 {
                        PI / 2.0
                    } else {
                        ((val - mins[j]) / range) * PI
                    }
                })
                .collect()
        })
        .collect()
}

// ─── Market Regime Labeling ───────────────────────────────────────────────────

/// Label market regimes based on forward returns.
/// +1.0 = bull, -1.0 = bear, 0.0 = sideways.
/// `forward_window` is the number of candles to look ahead.
/// `threshold` is the return threshold for bull/bear classification.
pub fn label_regimes(closes: &[f64], forward_window: usize, threshold: f64) -> Vec<f64> {
    let n = closes.len();
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        if i + forward_window >= n {
            labels.push(0.0); // insufficient forward data
        } else {
            let fwd_return = (closes[i + forward_window] / closes[i]).ln();
            if fwd_return > threshold {
                labels.push(1.0);
            } else if fwd_return < -threshold {
                labels.push(-1.0);
            } else {
                labels.push(0.0);
            }
        }
    }
    labels
}

/// Convert multi-class labels to binary: +1 if bull, -1 otherwise.
pub fn to_binary_labels(labels: &[f64]) -> Vec<f64> {
    labels
        .iter()
        .map(|&y| if y > 0.5 { 1.0 } else { -1.0 })
        .collect()
}

// ─── Evaluation Metrics ───────────────────────────────────────────────────────

/// Compute classification accuracy.
pub fn accuracy(predictions: &[f64], labels: &[f64]) -> f64 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&p, &l)| (p - l).abs() < 1e-10)
        .count();
    correct as f64 / n as f64
}

/// Compute a confusion matrix for given classes.
/// Returns (matrix, classes) where matrix[i][j] = count of true=classes[i], pred=classes[j].
pub fn confusion_matrix(predictions: &[f64], labels: &[f64]) -> (Vec<Vec<usize>>, Vec<f64>) {
    let mut classes: Vec<f64> = labels.to_vec();
    classes.append(&mut predictions.to_vec());
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();

    let nc = classes.len();
    let mut matrix = vec![vec![0usize; nc]; nc];

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        let i = classes.iter().position(|&c| (c - label).abs() < 1e-10).unwrap();
        let j = classes.iter().position(|&c| (c - pred).abs() < 1e-10).unwrap();
        matrix[i][j] += 1;
    }

    (matrix, classes)
}

/// Print a confusion matrix.
pub fn print_confusion_matrix(predictions: &[f64], labels: &[f64]) {
    let (matrix, classes) = confusion_matrix(predictions, labels);

    println!("\nConfusion Matrix (rows=true, cols=predicted):");
    print!("{:>10}", "");
    for c in &classes {
        print!("{:>10}", format!("{:.0}", c));
    }
    println!();

    for (i, row) in matrix.iter().enumerate() {
        print!("{:>10}", format!("{:.0}", classes[i]));
        for &val in row {
            print!("{:>10}", val);
        }
        println!();
    }
}

// ─── Bybit API Integration ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit API.
/// `symbol`: e.g., "BTCUSDT"
/// `interval`: e.g., "60" (1 hour), "D" (daily)
/// `limit`: number of candles (max 200)
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns most recent first, reverse to chronological order
    candles.reverse();

    Ok(candles)
}

/// Generate synthetic candle data for testing (when API is unavailable).
pub fn generate_synthetic_candles(n: usize, seed: u64) -> Vec<Candle> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64; // starting BTC-like price
    let mut timestamp = 1700000000u64;

    for _ in 0..n {
        let ret: f64 = rng.gen_range(-0.03..0.03);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.015));
        let low = close * (1.0 - rng.gen_range(0.0..0.015));
        let open = price * (1.0 + rng.gen_range(-0.01..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
        timestamp += 3600; // 1 hour intervals
    }

    candles
}

// ─── Unit Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_kernel_self() {
        // K(x, x) should be 1.0
        let x = vec![1.0, 0.5];
        let k = quantum_kernel(&x, &x, 1);
        assert!(
            (k - 1.0).abs() < 1e-10,
            "Self-kernel should be 1.0, got {}",
            k
        );
    }

    #[test]
    fn test_quantum_kernel_symmetry() {
        let x1 = vec![0.5, 1.2];
        let x2 = vec![1.0, 0.3];
        let k12 = quantum_kernel(&x1, &x2, 1);
        let k21 = quantum_kernel(&x2, &x1, 1);
        assert!(
            (k12 - k21).abs() < 1e-10,
            "Kernel should be symmetric: {} vs {}",
            k12,
            k21
        );
    }

    #[test]
    fn test_quantum_kernel_range() {
        // Kernel values should be in [0, 1]
        let x1 = vec![0.1, 2.5];
        let x2 = vec![2.0, 0.8];
        let k = quantum_kernel(&x1, &x2, 1);
        assert!(k >= -1e-10 && k <= 1.0 + 1e-10, "Kernel out of range: {}", k);
    }

    #[test]
    fn test_quantum_kernel_matrix_shape() {
        let data = vec![vec![0.5, 1.0], vec![1.5, 0.5], vec![0.2, 0.8]];
        let km = quantum_kernel_matrix(&data, 1);
        assert_eq!(km.shape(), &[3, 3]);
        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((km[[i, i]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_returns() {
        let closes = vec![100.0, 110.0, 105.0];
        let rets = log_returns(&closes);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((rets[1] - (105.0_f64 / 110.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_indicator() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.005];
        let rsi = rsi_indicator(&returns, 3);
        assert_eq!(rsi.len(), returns.len());
        // RSI values should be in [0, 1]
        for &r in &rsi {
            assert!(r >= 0.0 && r <= 1.0, "RSI out of range: {}", r);
        }
    }

    #[test]
    fn test_label_regimes() {
        let closes = vec![100.0, 102.0, 104.0, 103.0, 98.0, 97.0];
        let labels = label_regimes(&closes, 2, 0.02);
        assert_eq!(labels.len(), closes.len());
        // First label: forward 2 steps, 104/100 = 1.04, ln(1.04) ~ 0.039 > 0.02 => bull
        assert!((labels[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_features() {
        let features = vec![vec![0.0, -1.0], vec![1.0, 1.0], vec![0.5, 0.0]];
        let normed = normalize_features(&features);
        assert_eq!(normed.len(), 3);
        // min should map to 0, max to PI
        assert!((normed[0][0] - 0.0).abs() < 1e-10);
        assert!((normed[1][0] - PI).abs() < 1e-10);
        assert!((normed[0][1] - 0.0).abs() < 1e-10);
        assert!((normed[1][1] - PI).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_metric() {
        let preds = vec![1.0, -1.0, 1.0, -1.0];
        let labels = vec![1.0, -1.0, -1.0, -1.0];
        let acc = accuracy(&preds, &labels);
        assert!((acc - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_basic() {
        let preds = vec![1.0, -1.0, 1.0, -1.0];
        let labels = vec![1.0, -1.0, -1.0, -1.0];
        let (matrix, classes) = confusion_matrix(&preds, &labels);
        assert_eq!(classes.len(), 2);
        assert_eq!(matrix.len(), 2);
        // Total should equal number of samples
        let total: usize = matrix.iter().flat_map(|r| r.iter()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_svm_training_basic() {
        // Simple 2D linearly separable data
        let data = vec![
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![0.3, 0.1],
            vec![2.5, 2.5],
            vec![2.6, 2.8],
            vec![2.8, 2.5],
        ];
        let labels = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let model = train_qsvm(&data, &labels, 1.0, 1, 100);

        // Predict on training data
        let preds = predict_qsvm(&model, &data);
        let acc = accuracy(&preds, &labels);
        // Should get most right on linearly separable data
        assert!(acc >= 0.5, "Training accuracy too low: {}", acc);
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(100, 42);
        assert_eq!(candles.len(), 100);
        for c in &candles {
            assert!(c.close > 0.0);
            assert!(c.high >= c.low);
        }
    }

    #[test]
    fn test_feature_engineering() {
        let candles = generate_synthetic_candles(50, 42);
        let (features, _start_idx) = engineer_features(&candles, 5);
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 5);
    }

    #[test]
    fn test_zz_feature_map_normalization() {
        // State should be normalized (sum of |amplitude|^2 = 1)
        let x = vec![1.0, 0.5, 2.0];
        let state = zz_feature_map_state(&x, 2);
        let norm: f64 = state.iter().map(|(re, im)| re * re + im * im).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "State should be normalized, got norm = {}",
            norm
        );
    }
}
