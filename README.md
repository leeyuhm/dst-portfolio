# Data Science Toolbox portfolio

Selected computational work from the University of Bristol's *Data Science Toolbox* module.

The module covered statistical modelling, machine learning, deep learning, computational methods and modern data-science workflows. The assessed portfolio required students to investigate selected long-form questions in greater depth, supported where appropriate by additional experiments and code.

This repository contains the computational work supporting three of those investigations, using R and Python/PyTorch.

## R02.3: Time-series forecasting and cross-validation

**Question:** How should a prediction of global temperature in 2040 be constructed and validated, including appropriate uncertainty quantification?

`r02.3.R`

I investigated forecasting strategies for the global land-and-ocean temperature time series, with particular attention to the problems created by temporal dependence and increasing forecast horizons.

The analysis includes:

- ARIMA time-series forecasting
- Fifth-degree polynomial regression as a comparison model
- Blocked cross-validation appropriate for temporally ordered data
- Evaluation across multiple forecast horizons
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Analysis of forecast-error variability
- Parallelised repeated model evaluation using `foreach` and `doParallel`

The work examines how predictive accuracy and uncertainty change as the forecasting horizon increases and the limitations of using historical temperature data to extrapolate substantially into the future.

---

## R04.1: Selecting k in k-nearest-neighbour (k-NN) regression

**Question:** How should `k` be chosen in nearest-neighbour methods, and why can using a single fixed `k` be problematic?

`r04.1.R`  
`knn.R`

This investigation applies a method for selecting the number of neighbours in k-nearest-neighbour (k-NN) regression, motivated by the following paper:

Azadkia, M. (2019), *Optimal Choice of k for k-Nearest Neighbor Regression* https://arxiv.org/abs/1909.05495

The R analysis includes:

- k-NN regression
- Data-driven selection of `k`
- Leave-one-out cross-validation (LOOCV)
- Bootstrap analysis of the stability of the selected `k`
- Mahalanobis-distance analysis for identifying multivariate outliers
- Comparison of neighbour-selection behaviour before and after outlier handling
- Parallelised resampling

The supporting `knn.R` file contains the k-NN implementation used by the analysis; attribution to the original implementation is retained within the source code.

The investigation illustrates why an appropriate neighbourhood size depends on the data rather than being a universal fixed choice and assesses how stable that choice is under resampling.

---

## R07.1: Neural-network architecture and the Universal Approximation Theorem

**Question:** How do the width and depth of a multilayer perceptron affect its empirical performance, in the context of the Universal Approximation Theorem?

`r07.1.ipynb`

Using Fashion-MNIST, I constructed and trained multiple fully connected neural-network architectures in PyTorch, varying the number and size of hidden layers while keeping the training duration fixed.

The notebook includes:

- Six multilayer-perceptron architectures with different depths and widths
- Dynamic construction of hidden layers using `torch.nn`
- ReLU activation functions
- Cross-entropy loss
- Adam optimisation
- Mini-batch training
- Backpropagation
- GPU acceleration with CUDA where available
- Tracking of training and test loss
- Tracking of classification accuracy
- Comparison of model performance across architectures
- Confusion-matrix evaluation of the selected model

The experiment investigates the distinction between the theoretical representational capacity guaranteed by the Universal Approximation Theorem and the practical effects of network architecture on training and generalisation.

---

## R10.3: CPU vs GPU matrix multiplication

**Question:** How does matrix multiplication differ between CPU and massively parallel GPU implementations? What does this reveal about efficient parallel computation?

This investigation examines general matrix-matrix multiplication (GEMM) as an example of how algorithmic complexity interacts with hardware architecture.

The discussion covers:

- The theoretical \(O(n^3)\) computational cost of dense matrix multiplication
- CPU optimisation through blocking and cache reuse
- Spatial and temporal locality in memory access
- The importance of arithmetic intensity and reducing memory traffic
- Multi-core CPU parallelism
- GPU parallelism using large numbers of lightweight processing cores
- The SIMT (single-instruction, multiple-thread) execution model
- Memory-bandwidth limitations and the historical memory bottleneck in GPU computation
- Modern GPU developments including improved cache hierarchies and Tensor Cores
- Why dense matrix multiplication maps particularly well to massively parallel hardware
- The greater difficulty of optimising sparse and irregular matrix multiplication

The main conclusion is that computational performance depends not only on the number of arithmetic operations but also on data locality, memory-access patterns and the architecture on which an algorithm is executed. Dense matrix multiplication benefits strongly from blocking and predictable data reuse, whereas irregular or sparse workloads can make efficient use of caches and massively parallel hardware substantially more difficult.

---

## Repository structure

| File | Portfolio question | Focus |
|---|---|---|
| `r02.3.R` | R02.3 | Time-series forecasting and temporal cross-validation |
| `r04.1.R` | R04.1 | k-NN regression and data-driven selection of `k` |
| `knn.R` | R04.1 | Supporting k-NN implementation |
| `r07.1.ipynb` | R07.1 | MLP architecture comparison using Fashion-MNIST |
| `Portfolio.pdf` | — | Submitted portfolio |
| `figures/` | — | Supporting figures |
| `requirements.txt` | — | Python dependencies |
| `requirementsR.R` | — | R dependencies |
