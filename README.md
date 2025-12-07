# Overview (Heart Disease BN)

This notebook builds a Bayesian Network for the UCI heart-disease data, from preprocessing to exact inference. Data and outputs are organized in `data/` and `results/`.

## Pipeline

- **Data load & cleaning**: Read `data/processed.cleveland.data` (no header, `?` as NA). Immediately drop unused cols `fbs`, `restecg`. Optionally drop rows with NA (default) or keep.
- **Discretization**: Continuous cols binned by clinical cutpoints (default) or by quantiles (`q` configurable). Discretized frame `df_disc` is used for structure/parameter learning.
- **Train/Test split**: `df_disc` is shuffled and split 80/20 (configurable) into `train_df_disc`/`test_df_disc` before CPD fitting and prediction.
- **Structure learning**:
  - **PC**: Chi-square CI tests; `alpha`/`max_cond_vars` adjustable.
  - **HillClimbSearch**: BDeu score (default ESS=5).
  - **Custom**: Handcrafted edges reflecting clinical priors; edit the list directly.
- **Parameter learning**: Fit CPDs with `BayesianEstimator` + BDeu(ESS=5) for each learned/custom structure; prints CPD counts and `num`’s CPD.
- **Inference (exact)**: Variable Elimination and BeliefPropagation (clique tree) on a freshly fitted model (default custom). Two example queries for `P(num | evidence)`; evidence values must match discretized bin indices.
- **Prediction task (heart disease onset)**: Hold-out evaluation for `num` (5-class MAP + binary `num>0` accuracy) plus a helper to score a new patient and return `P(num)` and `P(num>0)`.
- **ML baselines (sklearn)**: LogisticRegression, RandomForest, MultinomialNB on the same discretized train/test split (`train_df_disc`/`test_df_disc`) for quick accuracy comparison against BN.
- **Visualization**: `show_graphviz` renders models via pygraphviz with full node names; defaults write PNGs under `results/`. `plot_dag` (matplotlib) also targets `results/`.

## Key files & paths

- **Input data**: `data/processed.cleveland.data` (others in `data/` if needed).
- **Outputs**: All plots under `results/` (PC/HC/custom Graphviz PNGs; matplotlib DAG).
- **Notebook parameters**: `cfg` in the “Run preprocessing” cell controls data path, binning strategy, quantiles, and missing-value handling.

## How to run

1. Execute cells top-down: imports → preprocessing → train/test split (`train_df_disc`/`test_df_disc`) → `show_graphviz` → PC/HC/custom structure → parameter fitting (uses train split) → inference.
2. Adjust edges/params:
   - Switch binning or `q` in the preprocessing cell.
   - Tweak PC (`alpha`/`max_cond_vars`), HC ESS, or edit the custom edge list.
   - Swap the model used for inference by changing `bn_for_query`.
3. Run the “预测任务：判断心脏病是否发作” cells to:
   - Split the discretized data into train/test, fit a BN, and print MAP + binary (`num>0`) accuracy.
   - Provide a patient dictionary (raw values, same coding as the dataset) to get `P(num)` and probability of heart disease being present.
4. Inspect outputs: CPD summaries printed in the parameter cell; VE/BP results in the inference cell; PNGs saved to `results/`.
