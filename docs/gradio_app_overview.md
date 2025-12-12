**Gradio App Overview (`gradio_app.py`)**

- Loads the trained Bayesian Network (`results/bn_model.pkl`; trains if missing), builds a `BeliefPropagation` engine, and renders the BN graph (Graphviz, fallback to matplotlib).
- Dark-themed UI with display-name labels; evidence dropdowns use the same binning/encoding as training (continuous features are pre-binned; `num` can also be evidence).
- Supports multi-variable queries and optional target states for joint probability; returns posterior tables, joint/top-10 assignments, and a posterior bar plot (`results/distributions.png`).

**How to Use**

- Launch: `python3 gradio_app.py` (opens at `http://127.0.0.1:7860`).
- BN Structure: shown at the top for quick reference.
- Query variables: check one or more variables (e.g., `Heart Disease Severity (num)` plus any others).
- Target states (optional): enter like `num=2, cp=3` to get that joint probability; leave empty to see top-10 joint assignments when multiple variables are queried.
- Evidence (optional): pick known patient attributes from dropdowns; leave blank to omit. Partial evidence is allowed.
- Run: click “Run inference.” Outputs show posterior distributions (with `P(num>0)` if `num` is queried), the joint probability or top-10 joints, and a saved/inline posterior plot.
