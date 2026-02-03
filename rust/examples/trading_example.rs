//! Trading Example: Quantum SVM for Market Regime Classification
//!
//! This example fetches BTCUSDT data from Bybit, engineers features,
//! trains a QSVM classifier, and evaluates predictions on a test set.

use quantum_svm_trading::*;

fn main() {
    println!("=== Quantum SVM Trading Example ===\n");

    // --- Step 1: Fetch or generate data ---
    println!("Step 1: Loading market data...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} candles from Bybit API", c.len());
            c
        }
        Err(e) => {
            println!("  Bybit API unavailable ({}), using synthetic data", e);
            let c = generate_synthetic_candles(200, 42);
            println!("  Generated {} synthetic candles", c.len());
            c
        }
    };

    // --- Step 2: Feature engineering ---
    println!("\nStep 2: Engineering features...");
    let window = 10;
    let (features, start_idx) = engineer_features(&candles, window);
    println!(
        "  Computed {} feature vectors (5 features each), starting from candle index {}",
        features.len(),
        start_idx
    );

    if features.len() < 20 {
        eprintln!("Not enough data points for meaningful train/test split. Need at least 20.");
        return;
    }

    // --- Step 3: Label market regimes ---
    println!("\nStep 3: Labeling market regimes...");
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let all_labels = label_regimes(&closes, 5, 0.005);

    // Align labels with features
    let labels: Vec<f64> = (0..features.len())
        .map(|i| all_labels[start_idx + i])
        .collect();

    // Count regime distribution
    let n_bull = labels.iter().filter(|&&l| l > 0.5).count();
    let n_bear = labels.iter().filter(|&&l| l < -0.5).count();
    let n_side = labels.iter().filter(|&&l| l.abs() <= 0.5).count();
    println!(
        "  Regime distribution: Bull={}, Bear={}, Sideways={}",
        n_bull, n_bear, n_side
    );

    // Convert to binary for QSVM (bull vs not-bull)
    let binary_labels = to_binary_labels(&labels);
    let n_pos = binary_labels.iter().filter(|&&l| l > 0.0).count();
    let n_neg = binary_labels.iter().filter(|&&l| l < 0.0).count();
    println!(
        "  Binary labels (bull vs rest): Positive={}, Negative={}",
        n_pos, n_neg
    );

    // --- Step 4: Normalize features ---
    println!("\nStep 4: Normalizing features to [0, pi]...");
    let norm_features = normalize_features(&features);

    // --- Step 5: Train/test split (80/20 temporal) ---
    let split = (norm_features.len() as f64 * 0.8) as usize;
    let train_features = &norm_features[..split];
    let test_features = &norm_features[split..];
    let train_labels = &binary_labels[..split];
    let test_labels = &binary_labels[split..];

    println!(
        "\nStep 5: Train/test split: {} train, {} test",
        train_features.len(),
        test_features.len()
    );

    // --- Step 6: Train QSVM ---
    println!("\nStep 6: Training Quantum SVM...");
    println!("  Feature map: ZZ feature map (depth=1)");
    println!("  Regularization C=1.0");
    println!("  Max iterations: 200");

    // Use only first 2 features to keep quantum simulation tractable
    // (2 qubits = 4-dimensional Hilbert space, manageable for classical simulation)
    let n_qubits = 2;
    let train_reduced: Vec<Vec<f64>> = train_features
        .iter()
        .map(|f| f[..n_qubits].to_vec())
        .collect();
    let test_reduced: Vec<Vec<f64>> = test_features
        .iter()
        .map(|f| f[..n_qubits].to_vec())
        .collect();

    println!(
        "  Using {} qubits (first {} features) for quantum kernel",
        n_qubits, n_qubits
    );

    let model = train_qsvm(
        &train_reduced,
        train_labels,
        1.0,  // C
        1,    // depth
        200,  // max_iter
    );

    println!(
        "  Training complete. {} support vectors found.",
        model.support_indices.len()
    );
    println!("  Bias: {:.6}", model.bias);

    // --- Step 7: Evaluate on training set ---
    println!("\nStep 7: Evaluating model...");

    let train_preds = predict_qsvm(&model, &train_reduced);
    let train_acc = accuracy(&train_preds, train_labels);
    println!("  Training accuracy: {:.2}%", train_acc * 100.0);

    // --- Step 8: Evaluate on test set ---
    let test_preds = predict_qsvm(&model, &test_reduced);
    let test_acc = accuracy(&test_preds, test_labels);
    println!("  Test accuracy:     {:.2}%", test_acc * 100.0);

    // --- Step 9: Confusion matrix ---
    println!("\n  Test set confusion matrix:");
    print_confusion_matrix(&test_preds, test_labels);

    // --- Step 10: Show some predictions ---
    println!("\nStep 8: Sample predictions (last 10 test points):");
    let start = if test_preds.len() > 10 {
        test_preds.len() - 10
    } else {
        0
    };
    println!("  {:>6} {:>10} {:>10}", "Index", "Predicted", "Actual");
    for i in start..test_preds.len() {
        let pred_str = if test_preds[i] > 0.0 { "BULL" } else { "BEAR/SIDE" };
        let actual_str = if test_labels[i] > 0.0 {
            "BULL"
        } else {
            "BEAR/SIDE"
        };
        println!("  {:>6} {:>10} {:>10}", i, pred_str, actual_str);
    }

    // --- Step 11: Quantum kernel analysis ---
    println!("\nStep 9: Quantum kernel analysis...");
    if test_reduced.len() >= 2 {
        let k01 = quantum_kernel(&test_reduced[0], &test_reduced[1], 1);
        println!(
            "  Kernel(test[0], test[1]) = {:.6}  (1.0 = identical, 0.0 = orthogonal)",
            k01
        );
    }
    if !train_reduced.is_empty() && !test_reduced.is_empty() {
        let k_train_test = quantum_kernel(&train_reduced[0], &test_reduced[0], 1);
        println!(
            "  Kernel(train[0], test[0]) = {:.6}",
            k_train_test
        );
    }

    println!("\n=== Quantum SVM Trading Example Complete ===");
}
