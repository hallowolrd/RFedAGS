# RFedAGS: PyTorch reproduction for Stiefel PCA

This directory contains a PyTorch implementation of the Stiefel/PCA experiments. The original MATLAB/Manopt files are retained unchanged.

## 1. Six Stiefel experiment groups

| Group | MATLAB entry point | Python command | Main settings |
|---|---|---|---|
| 1. Main synthetic comparison | `PCAStief_Syn_40.m` | `python scripts/run_synthetic.py` | d=100, r=5, N=40, S=100, K=5, B=50, lr=0.006, 1000 rounds, 5 repeats |
| 2. AGS vs tangent mean | `PCAStief_Syn_AGSvsTM.m` | `python scripts/run_ags_vs_tm.py` | d=50, r=1, N=2, S=60, K=50, B=60, lr=1e-7, 150 rounds, 1 run |
| 3. Local sample-size sweep | `PCAStief_Syn_localdatasizeincreas.m` | `python scripts/run_scalability.py --mode samples` | N=100, S=400/800/1200/1600/2000, d=100, r=5, K=5, B=S/2, lr=0.02, 2500 rounds, 4 repeats |
| 4. Client-count sweep | `PCAStief_Syn_numagentincreas.m` | `python scripts/run_scalability.py --mode agents` | N=60/80/100/150/200, S=1000, d=100, r=5, K=5, B=500, lr=0.02, 2500 rounds, 4 repeats |
| 5. Stiefel dimension sweep | `PCAStief_Syn_mandimincreas.m` | `python scripts/run_scalability.py --mode dimension` | (d,r)=(1000,5)/(2000,5)/(2000,10)/(4000,5)/(4000,10), N=50, S=1000, K=5, B=500, lr=0.02, 2500 rounds, 4 repeats |
| 6. CIFAR-10 | `PCAStief_CIFAR10.m` | `python scripts/run_cifar10.py` | d=3072, r=4, N=50, S=1000, K=5, B=500, lr=3e-5, 600 rounds, 2 repeats |

All synthetic scripts use float64 by default to match MATLAB's default numeric precision. Each repeated run shares its experiment's data, initialization and client participation probabilities; only the stochastic local sampling stream changes. Both per-run curves and their arithmetic mean are saved.

## 2. MATLAB discrepancies handled explicitly

The Python reproduction follows the intended experiment design rather than silently copying two apparent MATLAB copy/paste/data-generation mistakes:

1. **Client-count sweep:** the MATLAB script allocates 200000 rows but fills only 200 x 500 = 100000 rows, while it later assigns 1000 rows to each client. The Python port creates a complete master population of 200 clients with 1000 valid samples each. It uses the fixed population scale i/200 and takes the first N clients for each case. This is an intentional repair, not a bitwise reproduction of the faulty MATLAB data matrix.
2. **Dimension sweep:** the second MATLAB block is labelled (d,r)=(2000,5) but assigns d=1000, r=5 again. The Python port uses the labelled/intended setting (2000,5). This is an intentional correction.
3. **Sample-size sweep:** one fixed master dataset with 100 clients x 2000 samples is generated once. Every S case takes the first S samples from each client's same master data.
4. **Dimension sweep:** one fixed master dataset with 50 clients x 1000 samples x 4000 features is generated once. Smaller d cases take the first d feature columns.
5. **CIFAR-10:** the MATLAB repository references `CIFAR10_train_50_noniid_slight.mat`, but that preprocessed file is not included in the public repository. The Python script accepts the original file through `--mat-file` (variable name `data`, shape 50000x3072 or 3072x50000). Without it, the script uses a deterministic torchvision label-shard fallback; this is not the original client partition and must be reported as such.

## 3. Objective reference and runtime interpretation

For PCA, the global optimum is available from the top-r eigenvalues of the average covariance matrix. The Python scripts use this exact spectral reference by default to compute optimality gaps. The MATLAB scripts instead call Riemannian L-BFGS to estimate the reference value, so the reference computation differs while the mathematical optimum is the same.

The PyTorch simulator executes clients sequentially. Its elapsed time is wall-clock time for this simulator, not MATLAB's reported maximum parallel-client runtime. Compare objective/optimality-gap curves first; do not treat runtime curves as hardware-equivalent.

For CIFAR-10, the exact 3072x3072 eigendecomposition can be costly. Use `--skip-reference` to omit it; optimality-gap entries will then be NaN.

## 4. Install and run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the six experiments:

```bash
python scripts/run_synthetic.py
python scripts/run_ags_vs_tm.py
python scripts/run_scalability.py --mode samples
python scripts/run_scalability.py --mode agents
python scripts/run_scalability.py --mode dimension
python scripts/run_cifar10.py --data-root ../data
```

To use the author's preprocessed CIFAR-10 data, when you have obtained it:

```bash
python scripts/run_cifar10.py --mat-file /path/to/CIFAR10_train_50_noniid_slight.mat
```

Each experiment writes `metrics.csv` and a plot. Repeated runs are labelled individually; the mean curve has a `-mean` suffix. Scale sweeps write each case to its own directory and an aggregate summary under `results/scalability/<mode>/summary`.

## 5. Scope and algorithm notes

Implemented baselines: RFedAGS, RFedAvg, RFedSVRG, RFedProj and ZO-RFedProj. The Stiefel geometry uses the embedded Euclidean/Frobenius metric, tangent projection, thin QR retraction, and projection vector transport.

The Sphere, Grassmann, SPD and hyperbolic experiments are not part of this Stiefel-only Python port.
