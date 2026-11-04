# RFedAGS: PyTorch reproduction for Stiefel PCA

This directory adds a PyTorch implementation alongside the original MATLAB/Manopt code. It does not modify or delete the MATLAB implementation.

## 1. What the authors evaluated

The paper and the MATLAB repository contain experiments on five geometric settings:

| Geometry | Task | Data | Main purpose |
|---|---|---|---|
| Stiefel | Federated PCA | Synthetic Gaussian data, CIFAR-10 | Main comparison of RFedAGS with RFedAvg, RFedSVRG, RFedProj and ZO-RFedProj; scalability tests |
| Hyperbolic | Hyperbolic structured prediction | WordNet mammals subtree | Compare prediction distance against RFedAvg and RFedSVRG |
| SPD | Fréchet mean computation | PathMNIST-derived covariance matrices | Compare RFedAGS with RFedAvg and RFedSVRG |
| Sphere | Principal eigenvector computation | MNIST | Comprehensive tests: aggregation bias, participation schemes, stragglers, data heterogeneity, local-step count, and algorithm comparisons |
| Grassmann | Low-rank matrix completion | Synthetic data, MovieLens 1M | Compare RFedAGS with centralized RSD, RCG and LRBFGS; study the number of local updates |

The Stiefel part also has two supplementary experiment groups in the MATLAB repository:

- AGS versus tangent-mean aggregation: a two-client synthetic example with d=50, r=1, 60 samples per client and K=50.
- Scalability: increase the number of clients, local samples per client, and Stiefel ambient/subspace dimensions.

The paper's main Stiefel PCA settings are:

| Setting | Synthetic | CIFAR-10 |
|---|---:|---:|
| Ambient dimension d | 100 | 3072 |
| Subspace dimension r | 5 | 4 |
| Number of clients N | 40 | 50 |
| Samples per client S | 100 | 1000 |
| Local steps K | 5 | 5 |
| Batch size B | 50 | 500 |
| Local step size alpha | 6e-3 | 3e-5 |
| Participation | p_i sampled from U(0,1) | p_i sampled from U(0,1) |

The synthetic MATLAB entry point is PCAStief_Syn_40.m (1000 communication rounds, evaluation every 10 rounds, five seeds). The CIFAR-10 entry point is PCAStief_CIFAR10.m (600 rounds, evaluation every 10 rounds, two seeds).

## 2. What is implemented here

The current PyTorch port focuses on the Stiefel PCA experiments, consistent with the initial reproduction scope. It contains:

- RFedAGS with local Riemannian SGD, arbitrary Bernoulli participation and online frequency estimates.
- RFedAvg with inverse-QR tangent-mean aggregation.
- RFedSVRG with the local variance-reduced update from the MATLAB code.
- RFedProj and ZO-RFedProj with polar projection.
- Synthetic Gaussian data generation and a FedAvg-style label-sorted non-IID CIFAR-10 partition.
- Unit tests for the Stiefel geometry, inverse QR retraction and RFedAGS update.

The Sphere, Grassmann, SPD and hyperbolic experiments are analyzed above but are not yet ported in this first PyTorch module.

## 3. Geometry and algorithm correspondence

The implementation follows the Stiefel factory used in the authors' Manopt code:

- Manifold: St(r,d) = {X in R^(d x r): X.T X = I}.
- Metric: Euclidean/Frobenius metric.
- Riemannian gradient: project the Euclidean gradient using P_X(G) = G - X sym(X.T G).
- Retraction: thin QR retraction with the diagonal sign convention used by Manopt qr_unique.
- Vector transport: projection onto the tangent space at the destination point, P_Y(U).
- RFedAGS local step: eta = -alpha * grad f_i(X_local; batch), followed by QR retraction.
- Gradient stream: add the first local step directly; transport subsequent local steps back to the round's starting tangent space.
- Server aggregation: with frequency estimates q_i, use X_next = R_X((1/N) sum_{i in S_t} zeta_i/q_i).

This follows RFedAGSP.m and ConstructRFLPCAProb_St_.m. The paper's Stiefel experiments use local RSGD rather than the optional local RSVRG mode in RFedAGSP.m.

For PCA, the global optimum is available from the top-r eigenvectors of the average covariance matrix. This implementation can therefore use an exact spectral objective reference instead of the numerical Riemannian L-BFGS reference used by the MATLAB scripts.

## 4. Important reproduction notes

1. The repository does not contain the preprocessed file CIFAR10_train_50_noniid_slight.mat referenced by the MATLAB script. The PyTorch loader downloads CIFAR-10 through torchvision and creates a label-sorted shard partition with two shards per client, following the McMahan/FedAvg non-IID partitioning convention. This is a reproducible substitute, but it cannot guarantee byte-for-byte equality with the missing MATLAB partition.
2. CIFAR-10 images are normalized to [0,1] by default. Use --no-normalize if the MATLAB data file you obtain stores raw 0-255 pixels. The step size is scale-sensitive.
3. MATLAB reports the maximum active-client runtime as the local parallel time. This PyTorch simulator executes clients sequentially and reports actual process wall-clock time. Compare objective/gradient curves first; do not treat the runtime curves as hardware-equivalent.
4. The original RFedAGSP.m accumulates the gradient-sample counter using only the last active client's counter. This port counts all active clients' sampled examples.
5. The synthetic data generator follows the MATLAB code's client-i scale i/N. The paper describes the corresponding heterogeneous Gaussian covariance family.

## 5. Install and run

From this directory:

    python -m pip install -r requirements.txt

Run the main synthetic Stiefel PCA experiment:

    python scripts/run_synthetic.py

Run all five methods on the synthetic experiment:

    python scripts/run_synthetic.py --algorithms rfedags,rfedavg,rfedsvrg,rfedproj,zo-rfedproj

Run RFedAGS on CIFAR-10:

    python scripts/run_cifar10.py --data-root ../data --device cuda

For CIFAR-10, enable the exact spectral reference (this requires a 3072 by 3072 eigendecomposition):

    python scripts/run_cifar10.py --data-root ../data --device cuda --compute-reference

Run the tests:

    python -m unittest discover -s tests -v

Metrics are written to the selected output directory as metrics.csv and optimality_gap.png.
