# Chapter 190: Quantum SVM Trading

## 1. Introduction

Support Vector Machines (SVMs) have long been a workhorse of financial machine learning. Their ability to find optimal separating hyperplanes in high-dimensional feature spaces makes them naturally suited for classification problems in trading: distinguishing bull from bear regimes, predicting trade direction, and detecting anomalous patterns. However, as datasets grow in dimensionality and complexity, classical SVMs encounter computational bottlenecks, particularly when evaluating kernel functions in exponentially large feature spaces.

Quantum Support Vector Machines (QSVMs) offer a compelling alternative. By leveraging quantum computing principles, QSVMs can evaluate kernel functions in feature spaces that are intractable for classical computers. The core insight is that quantum circuits can efficiently map classical data into quantum Hilbert spaces of exponential dimension, compute inner products between these mapped states, and thereby construct kernel matrices that capture patterns invisible to classical kernels.

In this chapter, we build a complete Quantum SVM trading system in Rust. We simulate the quantum kernel computation classically (since fault-tolerant quantum hardware remains limited), but the mathematical framework is fully compatible with execution on real quantum processors. Our system fetches market data from the Bybit exchange, engineers financial features, labels market regimes, trains a QSVM classifier, and evaluates its predictive accuracy.

## 2. Mathematical Foundation

### 2.1 The QSVM Algorithm

A classical SVM solves the following optimization problem. Given training data {(x_i, y_i)} where x_i are feature vectors and y_i are class labels in {-1, +1}, we seek to maximize the margin between classes by finding the optimal separating hyperplane. The dual formulation involves computing a kernel matrix K where K_ij = K(x_i, x_j) = phi(x_i) . phi(x_j), with phi being a feature map into a higher-dimensional space.

In a QSVM, the feature map phi is implemented by a parameterized quantum circuit. The key steps are:

1. **Encoding**: Map classical data x into a quantum state |phi(x)> using a quantum feature map circuit U(x).
2. **Kernel Estimation**: Compute the quantum kernel as K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2, the squared overlap between quantum states.
3. **Optimization**: Use the quantum kernel matrix in a classical SVM optimization (e.g., SMO algorithm).
4. **Prediction**: Classify new data points using the trained model with quantum kernel evaluations.

### 2.2 Quantum Feature Map phi(x)

The ZZ feature map is one of the most widely studied quantum feature maps for classification tasks. For an n-dimensional input vector x = (x_1, x_2, ..., x_n), the ZZ feature map circuit acts on n qubits and consists of two layers:

**Layer 1 - Single-qubit rotations:**
Apply a Hadamard gate H to each qubit, followed by a Z-rotation:

```
U_phi(x) = exp(i * x_k * Z_k) * H_k   for each qubit k
```

**Layer 2 - Entangling interactions:**
Apply controlled-Z rotations between pairs of qubits:

```
U_ZZ(x) = exp(i * (pi - x_j)(pi - x_k) * Z_j Z_k)   for each pair (j, k)
```

The full feature map circuit is U(x) = U_ZZ(x) * U_phi(x), and the quantum state is |phi(x)> = U(x)|0...0>.

The feature map can be repeated r times (depth parameter) to increase expressibility:

```
U(x)^r = [U_ZZ(x) * U_phi(x)]^r
```

### 2.3 Kernel Estimation via Quantum Circuits

The quantum kernel K(x_i, x_j) is estimated by preparing the state:

```
|psi> = U(x_j)^dagger * U(x_i) |0...0>
```

and measuring the probability of obtaining the all-zeros outcome |0...0>. This probability equals |<phi(x_j)|phi(x_i)>|^2, which is precisely our kernel value.

On a real quantum computer, this would be estimated by repeated state preparation and measurement (sampling). In our classical simulation, we compute this exactly using matrix algebra.

### 2.4 Hyperplane in Quantum Feature Space

The decision function of the QSVM is:

```
f(x) = sign( sum_i alpha_i * y_i * K(x_i, x) + b )
```

where alpha_i are the dual variables (Lagrange multipliers) obtained from SVM optimization, y_i are training labels, K(x_i, x) is the quantum kernel, and b is the bias term. The support vectors are those training points with alpha_i > 0.

The separating hyperplane exists in the quantum feature Hilbert space, which has dimension 2^n for n qubits. This exponentially large space allows the QSVM to find separating boundaries that are impossible for classical kernels operating in polynomial-dimensional spaces.

## 3. Classical vs Quantum SVM

### 3.1 Computational Complexity Comparison

| Aspect | Classical SVM (RBF kernel) | Quantum SVM |
|--------|---------------------------|-------------|
| Feature space dimension | Infinite (for RBF) but structured | 2^n (Hilbert space) |
| Kernel evaluation (classical) | O(d) per pair | O(4^n) per pair (simulation) |
| Kernel evaluation (quantum HW) | N/A | O(poly(n)) per pair |
| Training (given kernel matrix) | O(N^2) to O(N^3) | O(N^2) to O(N^3) |
| Expressibility | Limited by kernel choice | Tunable via circuit design |

### 3.2 When Quantum Provides an Advantage

Quantum advantage in SVMs arises in specific scenarios:

1. **Hard classification problems**: When data is embedded in a feature space where classical kernels cannot efficiently separate the classes, but quantum kernels can. This occurs when the data-generating process has quantum-like correlations.

2. **High-dimensional feature spaces**: When the number of features is large, quantum feature maps can explore exponentially larger Hilbert spaces without exponential computational cost (on quantum hardware).

3. **Kernel alignment**: Quantum kernels are most advantageous when they have high alignment with the ideal kernel for the problem. Random quantum circuits tend to produce kernels that converge to trivial (identity-like) matrices as qubit count grows, so careful circuit design is essential.

4. **Financial applications**: Markets are complex adaptive systems with many interacting agents. The non-linear, high-dimensional nature of price dynamics may benefit from quantum feature spaces that capture multi-body correlations between features.

In practice, for the small feature dimensions typical in trading (5-20 features), the quantum advantage is primarily theoretical. Our implementation serves as a framework ready for deployment on quantum hardware as it matures.

## 4. Trading Application

### 4.1 Classifying Market Regimes

Market regime classification is a natural application of QSVM. We define three regimes based on rolling returns and volatility:

- **Bull regime (+1)**: Positive returns above a threshold with moderate volatility
- **Bear regime (-1)**: Negative returns below a threshold
- **Sideways regime (0)**: Returns within a narrow band around zero

The QSVM learns to map multi-dimensional feature vectors (returns, volatility, momentum indicators) to these regime labels. The quantum kernel can capture non-linear interactions between features that characterize regime transitions.

### 4.2 Predicting Trade Direction

Beyond regime classification, the QSVM can predict short-term trade direction. Given features computed at time t, the model predicts whether the price at time t+1 will be higher or lower. This is formulated as a binary classification with labels +1 (up) and -1 (down).

### 4.3 Detecting Anomalous Trading Patterns

One-class SVM variants using quantum kernels can detect anomalous market behavior. By training on "normal" market data, the model learns the boundary of typical market dynamics. Data points falling outside this boundary signal unusual activity, which may indicate regime changes, flash crashes, or manipulation.

### 4.4 Feature Engineering

Our feature set includes:

1. **Log returns**: r_t = ln(P_t / P_{t-1}) over multiple lookback windows
2. **Realized volatility**: Standard deviation of returns over a rolling window
3. **RSI-like momentum**: Ratio of average gains to average losses over a lookback period
4. **Volume ratio**: Current volume relative to moving average volume
5. **Price position**: Where current price sits relative to recent high/low range

Features are normalized to [0, pi] range for the quantum feature map encoding, since the ZZ feature map uses these values as rotation angles.

## 5. Implementation Walkthrough

### 5.1 Project Structure

```
190_quantum_svm_trading/
  rust/
    Cargo.toml
    src/
      lib.rs          # Core QSVM implementation
    examples/
      trading_example.rs  # Full trading pipeline
```

### 5.2 Quantum Feature Map Encoding

The ZZ feature map is implemented as a function that computes the unitary matrix U(x) for a given input vector x. For n features, we use n qubits. The implementation builds the circuit layer by layer:

```rust
// Pseudocode for ZZ feature map
fn zz_feature_map(x: &[f64], depth: usize) -> Array2<Complex64> {
    let n_qubits = x.len();
    let dim = 1 << n_qubits;
    let mut state = Array2::eye(dim); // identity

    for _ in 0..depth {
        // Apply H gates to all qubits
        // Apply Rz(x_k) to each qubit k
        // Apply RZZ((pi - x_j)(pi - x_k)) to each pair (j,k)
    }
    state
}
```

### 5.3 Quantum Kernel Matrix

The kernel matrix is computed by evaluating K(x_i, x_j) for all pairs of training samples:

```rust
fn quantum_kernel_matrix(data: &[Vec<f64>], depth: usize) -> Array2<f64> {
    let n = data.len();
    let mut kernel = Array2::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let k = quantum_kernel(&data[i], &data[j], depth);
            kernel[[i, j]] = k;
            kernel[[j, i]] = k;
        }
    }
    kernel
}
```

### 5.4 SVM Training

We use a simplified Sequential Minimal Optimization (SMO) approach to solve the SVM dual problem with the quantum kernel matrix. The key optimization loop selects pairs of Lagrange multipliers and updates them to improve the objective.

### 5.5 Bybit Integration

Data is fetched from the Bybit public API (no authentication required for market data):

```rust
let url = format!(
    "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
    symbol, interval, limit
);
```

The response contains OHLCV candles which are parsed into our internal data structures for feature engineering.

## 6. Bybit Data Integration

The Bybit API provides historical kline (candlestick) data that serves as our input. We fetch BTCUSDT perpetual futures data at various timeframes. The API returns arrays of [timestamp, open, high, low, close, volume, turnover] which we parse into structured candle data.

Key considerations for data integration:

- **Rate limiting**: Bybit allows generous rate limits for public endpoints, but we implement reasonable delays between requests.
- **Data quality**: We handle missing candles and verify timestamp continuity.
- **Normalization**: Raw prices are converted to returns and normalized indicators before feeding into the quantum feature map.
- **Train/test split**: We use temporal splitting (earlier data for training, later for testing) to avoid look-ahead bias, which is critical in financial ML.

The pipeline:
1. Fetch N candles of BTCUSDT from Bybit
2. Compute features (returns, volatility, RSI, volume ratio, price position)
3. Label each candle with a market regime
4. Normalize features to [0, pi]
5. Split into train/test sets
6. Compute quantum kernel matrices
7. Train QSVM on training set
8. Predict on test set and evaluate

## 7. Key Takeaways

1. **Quantum SVMs extend classical SVMs** by replacing the classical kernel with a quantum kernel computed from parameterized quantum circuits. The quantum feature map encodes data into an exponentially large Hilbert space.

2. **The ZZ feature map** is a practical choice for quantum kernel computation. It encodes both individual feature values (via single-qubit rotations) and feature interactions (via two-qubit entangling gates).

3. **Kernel estimation** on a quantum computer requires only polynomial resources in the number of qubits, but classical simulation scales exponentially. This is where quantum hardware will eventually provide a speedup.

4. **Market regime classification** is a natural application. The multi-class nature of regime identification (bull/bear/sideways) and the non-linear interactions between financial features make it a good candidate for quantum kernel methods.

5. **Feature engineering remains critical**. Even with a powerful quantum kernel, the choice and quality of input features determines model performance. Standard financial features (returns, volatility, momentum) provide a solid foundation.

6. **Classical simulation is sufficient for development**. Our Rust implementation simulates the quantum circuits exactly, allowing us to develop and test the full pipeline without quantum hardware. The same code logic maps directly to quantum execution when hardware is available.

7. **Practical considerations**: For production trading, the current quantum advantage is limited by hardware constraints (noise, qubit count, coherence times). The framework we build here is designed to transition seamlessly to real quantum backends as they mature.

8. **Rust provides performance benefits** for the computationally intensive kernel matrix computation and SVM optimization. The strong type system also helps catch errors in complex linear algebra operations at compile time.
