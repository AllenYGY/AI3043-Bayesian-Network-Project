import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import BeliefPropagation
from pgmpy.models import DiscreteBayesianNetwork

MODEL_PATH = Path("results/bn_model.pkl")
DATA_PATH = "data/processed.cleveland.data"
CSS_STYLE = """
.gradio-container {background-color:#0f172a;}
.gradio-container .block, .gradio-container .form, .gradio-container .panel {background-color:#111827;}
.gradio-container label, .gradio-container h1, .gradio-container h2, .gradio-container h3 {color:#e5e7eb;}
.gradio-container .markdown-body {color:#e5e7eb;}
"""


def save_model(bn_model, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bn_model, f)


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


COLUMNS_ALL = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
DROP_COLS = ["fbs", "restecg"]
COLUMNS = [c for c in COLUMNS_ALL if c not in DROP_COLS]
FEATURE_COLS = [c for c in COLUMNS if c != "num"]

DISPLAY_NAMES = {  # mapping of column names to more descriptive display names
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Number of Stenosed Vessels",
    "thal": "Thalassemia",
    "num": "Heart Disease Severity",
}

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CUSTOM_BINS = {
    "age": [45, 60],
    "trestbps": [140],
    "chol": [200, 239],
    "thalach": [150],
    "oldpeak": [1.0],
}
RIGHT_OPEN_COLS = {"trestbps", "thalach"}


def _fmt_val(v):
    if v in (math.inf, -math.inf):
        return "inf" if v > 0 else "-inf"
    if float(v).is_integer():
        return str(int(v))
    return str(v)


def interval_labels(edges, right=True):
    labels = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo == -math.inf:
            labels[i] = f"< {_fmt_val(hi)}"
        elif hi == math.inf:
            op = ">=" if not right else ">"
            labels[i] = f"{op} {_fmt_val(lo)}"
        else:
            labels[i] = f"{_fmt_val(lo)} - {_fmt_val(hi)}"
    return labels


BIN_LABELS: Dict[str, Dict[int, str]] = {}
for col, cuts in CUSTOM_BINS.items():
    edges = [-math.inf, *cuts, math.inf]
    right = col not in RIGHT_OPEN_COLS
    BIN_LABELS[col] = interval_labels(edges, right=right)

BIN_LABELS.update({
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"},
    "num": {0: "Healthy", 1: "HD-1", 2: "HD-2", 3: "HD-3", 4: "HD-4"},
})

OPTIONS = {
    col: [(lbl, code) for code, lbl in labels.items()]
    for col, labels in BIN_LABELS.items()
}


def display_label(col: str) -> str:
    return DISPLAY_NAMES.get(col, col)


def get_label(col: str, val: int) -> str:
    """Helper function to get readable label for a column value"""
    return BIN_LABELS.get(col, {}).get(val, str(val))


CUSTOM_EDGES = [
    ("age", "trestbps"),
    ("age", "ca"),
    ("age", "chol"),
    ("age", "thalach"),
    ("sex", "chol"),
    ("sex", "cp"),
    ("sex", "thalach"),
    ("cp", "exang"),
    ("cp", "num"),
    ("trestbps", "ca"),
    ("chol", "trestbps"),
    ("chol", "ca"),
    ("thalach", "exang"),
    ("thalach", "oldpeak"),
    ("exang", "oldpeak"),
    ("oldpeak", "slope"),
    ("slope", "num"),
    ("ca", "num"),
    ("thal", "num"),
]


def load_df(path: Path, has_header: bool = False) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else COLUMNS_ALL,
        na_values="?",
    )


def bin_custom(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, cuts in CUSTOM_BINS.items():
        if col not in out.columns:
            continue
        edges = [-math.inf, *cuts, math.inf]
        right = col not in RIGHT_OPEN_COLS
        out[col] = pd.cut(out[col], bins=edges, labels=False, right=right)
    return out


def preprocess(data_path: str, binning: str = "custom", q: int = 4):
    df = load_df(Path(data_path), has_header=False)
    df = df.drop(columns=DROP_COLS).dropna()
    if binning == "custom":
        df_disc = bin_custom(df)
    else:
        df_disc = df.copy()
        for col in CONTINUOUS:
            df_disc[col] = pd.qcut(df[col], q=q, labels=False, duplicates="drop")
    return df, df_disc


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = max(1, int(len(df_shuffled) * (1 - test_size)))
    return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]


def fit_with_bdeu(model_like, data: pd.DataFrame, ess: int = 5):
    edges = list(model_like.edges())
    nodes = list(model_like.nodes())
    bn = DiscreteBayesianNetwork(edges)
    bn.add_nodes_from(nodes)
    bn.fit(
        data,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=ess,
    )
    return bn


def build_model(data_path: str, ess: int = 5, test_size: float = 0.2, seed: int = 42):
    _, df_disc = preprocess(data_path, binning="custom")
    train_df, test_df = split_train_test(df_disc, test_size=test_size, seed=seed)
    model_custom = DiscreteBayesianNetwork(CUSTOM_EDGES)
    model_custom.add_nodes_from(COLUMNS)
    bn = fit_with_bdeu(model_custom, train_df, ess=ess)
    return bn, train_df, test_df


def plot_bn_image(model, outfile: Path = Path("results/bn_graph.png")) -> Path:
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    pos = {}
    layer1 = ["age", "sex"]
    for i, node in enumerate(layer1):
        if node in G.nodes():
            pos[node] = (i * 3, 5)

    layer2 = ["chol", "thalach", "trestbps", "cp", "exang"]
    for i, node in enumerate(layer2):
        if node in G.nodes():
            pos[node] = (i * 2.5 - 1, 3.5)

    layer3 = ["ca", "thal"]
    for i, node in enumerate(layer3):
        if node in G.nodes():
            pos[node] = (i * 3, 2)

    layer4 = ["slope", "oldpeak"]
    for i, node in enumerate(layer4):
        if node in G.nodes():
            pos[node] = (i * 4 + 1, 0.5)

    if "num" in G.nodes():
        pos["num"] = (3, -1)

    _ = plt.figure(figsize=(14, 10), facecolor="#0f172a")
    ax = plt.gca()
    ax.set_facecolor("#0f172a")

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in ["age", "sex"]:
            node_colors.append("#8b5cf6")
            node_sizes.append(3500)
        elif node in ["chol", "thalach", "trestbps", "cp", "exang"]:
            node_colors.append("#3b82f6")
            node_sizes.append(3200)
        elif node in ["cp", "exang", "oldpeak", "slope"]:
            node_colors.append("#10b981")
            node_sizes.append(3200)
        elif node in ["ca", "thal"]:
            node_colors.append("#f59e0b")
            node_sizes.append(3200)
        elif node == "num":
            node_colors.append("#ef4444")
            node_sizes.append(4000)
        else:
            node_colors.append("#6b7280")
            node_sizes.append(3000)

    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        ax.annotate(
            "",
            xy=(x[1], y[1]),
            xytext=(x[0], y[0]),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=3,
                alpha=0.9,
                color="#EBEFF4",
                connectionstyle="arc3,rad=0.1",
                shrinkA=20,
                shrinkB=20,
            ),
        )

    for i, node in enumerate(G.nodes()):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=[node_colors[i]],
            node_size=node_sizes[i] + 400,
            alpha=0.2,
            edgecolors="none",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=[node_colors[i]],
            node_size=node_sizes[i],
            alpha=0.95,
            edgecolors="#1e293b",
            linewidths=3,
        )

    labels = {n: DISPLAY_NAMES.get(n, n) for n in G.nodes()}
    wrapped_labels = {}
    for node, label in labels.items():
        if len(label) > 12:
            words = label.split()
            if len(words) > 1:
                mid = len(words) // 2
                wrapped_labels[node] = "\n".join([
                    " ".join(words[:mid]),
                    " ".join(words[mid:]),
                ])
            else:
                wrapped_labels[node] = label
        else:
            wrapped_labels[node] = label

    nx.draw_networkx_labels(
        G,
        pos,
        wrapped_labels,
        font_size=11,
        font_weight="bold",
        font_color="#f8fafc",
        font_family="sans-serif",
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#8b5cf6", label="Demographics", edgecolor="#1e293b", linewidth=2
        ),
        Patch(
            facecolor="#3b82f6", label="Physiology", edgecolor="#1e293b", linewidth=2
        ),
        Patch(
            facecolor="#10b981", label="ECG Features", edgecolor="#1e293b", linewidth=2
        ),
        Patch(
            facecolor="#f59e0b", label="Diagnostics", edgecolor="#1e293b", linewidth=2
        ),
        Patch(facecolor="#ef4444", label="Target", edgecolor="#1e293b", linewidth=2),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
        facecolor="#1e293b",
        edgecolor="#475569",
        labelcolor="#f8fafc",
    )

    plt.title(
        "Heart Disease Bayesian Network Structure",
        fontsize=18,
        fontweight="bold",
        color="#f8fafc",
        pad=20,
    )

    plt.axis("off")
    plt.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    return outfile


def plot_sensitivity_analysis(
    bn_model,
    data_df,
    display_names=None,
    bin_labels=None,
):
    """Interactive Sensitivity Analysis with readable labels - Returns Plotly Figure"""
    print("Starting sensitivity analysis...")

    infer = BeliefPropagation(bn_model)
    evidence_cols = [c for c in data_df.columns if c != "num"]
    results = {}

    try:
        q_baseline = infer.query(variables=["num"], show_progress=False)
        if 0 in q_baseline.state_names["num"]:
            p_healthy_baseline = q_baseline.values[
                q_baseline.state_names["num"].index(0)
            ]
        else:
            p_healthy_baseline = 0.0
        baseline_p_sick = 1.0 - p_healthy_baseline
    except Exception:
        baseline_p_sick = 0.5

    print(f"Baseline P(Heart Disease) = {baseline_p_sick:.3f}")

    for col in evidence_cols:
        min_p = 1.0
        max_p = 0.0
        min_state_label = "N/A"
        max_state_label = "N/A"

        possible_values = sorted(data_df[col].unique())
        valid_calc = False

        for val in possible_values:
            try:
                q_res = infer.query(
                    variables=["num"], evidence={col: val}, show_progress=False
                )

                if 0 in q_res.state_names["num"]:
                    idx_0 = q_res.state_names["num"].index(0)
                    p_healthy = q_res.values[idx_0]
                else:
                    p_healthy = 0.0

                p_sick = 1.0 - p_healthy
                current_label = get_label(col, val) if bin_labels else str(val)

                if p_sick > max_p:
                    max_p = p_sick
                    max_state_label = current_label

                if p_sick < min_p:
                    min_p = p_sick
                    min_state_label = current_label

                valid_calc = True
            except Exception:
                continue

        if not valid_calc:
            results[col] = (baseline_p_sick, baseline_p_sick, "N/A", "N/A")
        else:
            results[col] = (max_p, min_p, max_state_label, min_state_label)

    df_results = pd.DataFrame.from_dict(
        results, orient="index", columns=["Max P", "Min P", "Max Label", "Min Label"]
    )
    df_results["Range"] = df_results["Max P"] - df_results["Min P"]
    df_results = df_results.sort_values(by="Range", ascending=True)

    if display_names:
        labels = [display_names.get(col, col) for col in df_results.index]
    else:
        labels = df_results.index.tolist()

    # Enhanced colors for dark theme
    color_max = "#ff6b6b"  # Brighter red
    color_min = "#4ecdc4"  # Brighter cyan
    color_bar = "#94a3b8"  # Lighter gray
    color_base = "#fbbf24"  # Yellow for baseline (more visible)

    fig = go.Figure()

    # Draw connecting bars with better visibility
    for i, (idx, row) in enumerate(df_results.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[row["Min P"], row["Max P"]],
                y=[labels[i], labels[i]],
                mode="lines",
                line=dict(color=color_bar, width=14),  # Slightly thicker
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Min Risk points with glow effect
    fig.add_trace(
        go.Scatter(
            x=df_results["Min P"],
            y=labels,
            mode="markers+text",
            name="Min Risk",
            marker=dict(
                color=color_min,
                size=18,  # Slightly larger
                line=dict(width=2, color="#f8fafc"),  # White border
                opacity=0.9,
            ),
            text=[f"{x:.3f}" for x in df_results["Min P"]],
            textposition="middle left",
            textfont=dict(
                size=11, color="#f8fafc", family="Arial Black"
            ),  # Bold white text
            customdata=df_results["Min Label"],
            hovertemplate=(
                "<b style='font-size:14px'>%{y}</b><br>"
                + "<span style='color:#4ecdc4'>Condition: <b>%{customdata}</b></span><br>"
                + "<span style='color:#4ecdc4'>Risk: <b>%{x:.3f}</b></span>"
                + "<extra></extra>"
            ),
        )
    )

    # Max Risk points with glow effect
    fig.add_trace(
        go.Scatter(
            x=df_results["Max P"],
            y=labels,
            mode="markers+text",
            name="Max Risk",
            marker=dict(
                color=color_max,
                size=18,  # Slightly larger
                line=dict(width=2, color="#f8fafc"),  # White border
                opacity=0.9,
            ),
            text=[f"{x:.3f}" for x in df_results["Max P"]],
            textposition="middle right",
            textfont=dict(
                size=11, color="#f8fafc", family="Arial Black"
            ),  # Bold white text
            customdata=df_results["Max Label"],
            hovertemplate=(
                "<b style='font-size:14px'>%{y}</b><br>"
                + "<span style='color:#ff6b6b'>Condition: <b>%{customdata}</b></span><br>"
                + "<span style='color:#ff6b6b'>Risk: <b>%{x:.3f}</b></span>"
                + "<extra></extra>"
            ),
        )
    )

    # Baseline with better visibility
    fig.add_vline(
        x=baseline_p_sick,
        line_width=3,  # Thicker line
        line_dash="dash",
        line_color=color_base,
        annotation=dict(
            text=f"Baseline: {baseline_p_sick:.3f}",
            font=dict(size=13, color="#fbbf24", family="Arial Black"),
            bgcolor="#1e293b",
            bordercolor="#fbbf24",
            borderwidth=2,
            borderpad=4,
        ),
        annotation_position="top right",
    )

    fig.update_layout(
        title={
            "text": "Sensitivity Analysis: Risk Factor Impact",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(
                size=18, color="#f8fafc", family="Arial Black"
            ),  # Brighter white
        },
        xaxis=dict(
            title="P(Heart Disease)",
            title_font=dict(color="#f8fafc", size=14, family="Arial"),
            tickfont=dict(color="#f8fafc", size=12),
            range=[
                max(0, df_results["Min P"].min() - 0.05),
                min(1.0, df_results["Max P"].max() + 0.05),
            ],
            showgrid=True,
            gridcolor="#475569",  # Lighter grid
            gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#f8fafc", family="Arial"),  # Brighter
        ),
        plot_bgcolor="#0f172a",  # Match main background
        paper_bgcolor="#0f172a",  # Match main background
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#f8fafc", size=12),
            bgcolor="#1e293b",
            bordercolor="#475569",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=60, b=60),
        height=650,
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=13,
            font_family="Arial",
            font_color="#f8fafc",
            bordercolor="#475569",
        ),
    )

    print("Sensitivity analysis completed")

    return fig


def show_graphviz(model, prog: str = "dot", outfile: str = "results/bn_graphviz.png"):
    label_map = {n: DISPLAY_NAMES.get(n, n) for n in model.nodes()}
    try:
        g = model.to_graphviz()
    except Exception as e:
        print(f"to_graphviz failed: {e}")
        return None
    try:
        for n in g.nodes():
            g.get_node(n).attr["label"] = label_map.get(n, n)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        g.draw(outfile, prog=prog)
    except Exception as e:
        print(f"Graphviz draw failed: {e}")
        return None
    return outfile


def prepare_evidence(raw_evidence: Dict[str, Any]):
    if not raw_evidence:
        return {}
    return {k: v for k, v in raw_evidence.items() if v not in (None, "")}


def plot_distributions(
    results: Dict[str, Dict[int, float]],
    target_map: Optional[Dict[str, int]],
    evidence: Dict[str, Any],
    outfile: Path = Path("results/distributions.png"),
) -> Path:
    n_vars = len(results)
    if n_vars == 0:
        return None

    plt.style.use("seaborn-v0_8-darkgrid")

    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        facecolor="#0f172a",
    )
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, (var, dist) in enumerate(results.items()):
        ax = axes[idx]
        ax.set_facecolor("#111827")

        states = sorted(dist.keys())
        probs = [dist[s] for s in states]

        labels = [BIN_LABELS.get(var, {}).get(s, str(s)) for s in states]

        target_state = target_map.get(var) if target_map else None
        colors = ["#ef4444" if s == target_state else "#3b82f6" for s in states]

        bars = ax.bar(
            range(len(states)),
            probs,
            color=colors,
            alpha=0.85,
            edgecolor="#1f2937",
            linewidth=1.5,
        )

        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(
            labels, rotation=35, ha="right", fontsize=10, weight="bold", color="#e5e7eb"
        )
        ax.set_ylabel("Probability", fontsize=12, weight="bold", color="#e5e7eb")
        ax.set_title(
            f"{DISPLAY_NAMES.get(var, var)}",
            fontsize=14,
            fontweight="bold",
            color="#e5e7eb",
            pad=15,
        )
        ax.set_ylim(0, min(1.05, max(probs) * 1.15))
        ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#374151")
        ax.spines["bottom"].set_color("#374151")

        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                color="#e5e7eb",
            )

        if target_state is not None and target_state in states:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor="#ef4444",
                    edgecolor="#1f2937",
                    label=f"Target: {labels[states.index(target_state)]}",
                ),
                Patch(facecolor="#3b82f6", edgecolor="#1f2937", label="Other states"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=10,
                framealpha=0.85,
                edgecolor="#374151",
            )

    for idx in range(n_vars, len(axes)):
        axes[idx].axis("off")

    if evidence:
        evidence_strs = []
        for k, v in evidence.items():
            label = BIN_LABELS.get(k, {}).get(v, str(v))
            evidence_strs.append(f"{DISPLAY_NAMES.get(k, k)} = {label}")
        evidence_text = "Evidence: " + " | ".join(evidence_strs)
        fig.suptitle(
            evidence_text, fontsize=15, fontweight="bold", y=0.99, color="#e5e7eb"
        )
    else:
        fig.suptitle(
            "Prior Distribution (No Evidence)",
            fontsize=15,
            fontweight="bold",
            y=0.99,
            color="#e5e7eb",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    return outfile


def query_bn(
    bp: BeliefPropagation,
    variables: List[str],
    raw_evidence: Dict[str, Any],
    target_states: Optional[Dict[str, Any]] = None,
):
    if not variables:
        raise ValueError("Please select at least one variable to query.")

    evidence = prepare_evidence(raw_evidence)
    clean_evidence = {k: v for k, v in evidence.items() if k not in variables}
    qres = bp.query(variables=variables, evidence=clean_evidence or None)

    results = {}
    for var in variables:
        if len(variables) == 1:
            probs = pd.Series(qres.values, index=qres.state_names[var])
        else:
            factor = qres if hasattr(qres, "state_names") else qres
            marginal = factor.marginalize(
                [v for v in variables if v != var], inplace=False
            )
            probs = pd.Series(marginal.values, index=marginal.state_names[var])
        results[var] = probs.to_dict()

    joint_entries = []
    if len(variables) > 1:
        factor = qres if hasattr(qres, "state_names") else qres
        vars_order = factor.variables
        state_map = factor.state_names
        for combo in __import__("itertools").product(*[
            state_map[v] for v in vars_order
        ]):
            assignment = {v: val for v, val in zip(vars_order, combo)}
            prob = factor.get_value(**assignment)
            joint_entries.append((assignment, prob))
        joint_entries.sort(key=lambda x: x[1], reverse=True)

    joint_prob = None
    if target_states:
        if len(variables) > 1:
            factor = qres if hasattr(qres, "state_names") else qres
            try:
                relevant_states = {
                    k: v for k, v in target_states.items() if k in variables
                }
                joint_prob = factor.get_value(**relevant_states)
            except (KeyError, ValueError, AttributeError):
                joint_prob = None
        elif len(variables) == 1:
            var = variables[0]
            val = target_states.get(var)
            if val is not None and var in results:
                joint_prob = results[var].get(val)

    return results, joint_prob, joint_entries


# 在模型加载后生成敏感性分析
try:
    BN_MODEL = load_model(MODEL_PATH)
    TRAIN_DF, TEST_DF = None, None
    _, df_disc = preprocess(DATA_PATH, binning="custom")
    TRAIN_DF, TEST_DF = split_train_test(df_disc, test_size=0.2, seed=42)
except FileNotFoundError:
    BN_MODEL, TRAIN_DF, TEST_DF = build_model(DATA_PATH, ess=5, test_size=0.2, seed=42)
    save_model(BN_MODEL, MODEL_PATH)

BP_ENGINE = BeliefPropagation(BN_MODEL)
BN_IMAGE_PATH = plot_bn_image(BN_MODEL) or show_graphviz(BN_MODEL)

# Generate sensitivity analysis figure
SENSITIVITY_FIG = plot_sensitivity_analysis(
    BN_MODEL,
    TRAIN_DF,
    display_names=DISPLAY_NAMES,
    bin_labels=BIN_LABELS,
)


def initialize_display():
    """Initialize the display with default query"""
    return ui_predict(
        variables=["num"],
        age=None,
        sex=None,
        cp=None,
        trestbps=None,
        chol=None,
        thalach=None,
        exang=None,
        oldpeak=None,
        slope=None,
        ca=None,
        thal=None,
        num=None,
    )


def ui_predict(
    variables,
    age,
    sex,
    cp,
    trestbps,
    chol,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
    num,
):
    raw = {}
    for name, val in [
        ("age", age),
        ("sex", sex),
        ("cp", cp),
        ("trestbps", trestbps),
        ("chol", chol),
        ("thalach", thalach),
        ("exang", exang),
        ("oldpeak", oldpeak),
        ("slope", slope),
        ("ca", ca),
        ("thal", thal),
        ("num", num),
    ]:
        if val is None or val == "":
            continue
        raw[name] = val

    vars_list = variables or ["num"]

    highlight_map = {}
    for var in vars_list:
        if var in raw:
            highlight_map[var] = raw[var]

    res, joint_prob, joint_entries = query_bn(
        BP_ENGINE, vars_list, raw, target_states=None
    )

    evidence = prepare_evidence(raw)
    display_evidence = {k: v for k, v in evidence.items() if k not in vars_list}
    plot_path = plot_distributions(res, highlight_map, display_evidence)

    lines = ["## Posterior Distributions"]
    for var, dist in res.items():
        lines.append(f"### {DISPLAY_NAMES.get(var, var)}")
        lines.append("")
        for k, v in dist.items():
            label = BIN_LABELS.get(var, {}).get(k, str(k))
            marker = " **[HIGHLIGHTED]**" if highlight_map.get(var) == k else ""
            lines.append(f"- `{label}`: **{v:.4f}**{marker}")
        if var == "num" and 0 in dist:
            hd_prob = 1 - dist.get(0, 0.0)
            lines.append(f"- **P(num>0)** = **{hd_prob:.4f}**")
        lines.append("")

    joint_text = ""
    if len(vars_list) > 1 and joint_entries:
        top = joint_entries[:10]
        joint_lines = ["## Joint Distribution (Top 10)", ""]
        for i, (assign, prob) in enumerate(top, 1):
            assign_parts = []
            for k, v in assign.items():
                label = BIN_LABELS.get(k, {}).get(v, str(v))
                assign_parts.append(f"**{DISPLAY_NAMES.get(k, k)}** = `{label}`")
            assign_str = ", ".join(assign_parts)
            joint_lines.append(f"{i}. {assign_str}: **{prob:.6f}**")
        joint_text = "\n".join(joint_lines)

    return "\n".join(lines), joint_text, str(plot_path)


with gr.Blocks(title="Heart Disease BN Query") as demo:
    gr.HTML(f"<style>{CSS_STYLE}</style>")
    gr.Markdown(
        "# Heart Disease Bayesian Network\n"
        "Query the trained BN with optional evidence. States use the same binning as training; leave evidence blank to omit it."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(
                value=str(BN_IMAGE_PATH),
                label="Bayesian Network Structure",
                type="filepath",
            )
        with gr.Column(scale=1):
            gr.Plot(value=SENSITIVITY_FIG, label="Sensitivity Analysis")

    gr.Markdown("## Query Variables Selection")
    variables = gr.CheckboxGroup(
        choices=[(display_label(c), c) for c in COLUMNS],
        value=["num"],
        label="Query variables",
    )

    gr.Markdown("## Evidence Variables (optional)")

    with gr.Row():
        with gr.Column(scale=5):
            age = gr.Dropdown(
                choices=OPTIONS["age"], label=display_label("age"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            age_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            sex = gr.Dropdown(
                choices=OPTIONS["sex"], label=display_label("sex"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            sex_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            cp = gr.Dropdown(
                choices=OPTIONS["cp"], label=display_label("cp"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            cp_clear = gr.Button("Clear", size="sm")

    with gr.Row():
        with gr.Column(scale=5):
            trestbps = gr.Dropdown(
                choices=OPTIONS["trestbps"],
                label=display_label("trestbps"),
                value=None,
            )
        with gr.Column(scale=1, min_width=60):
            trestbps_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            chol = gr.Dropdown(
                choices=OPTIONS["chol"], label=display_label("chol"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            chol_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            thalach = gr.Dropdown(
                choices=OPTIONS["thalach"],
                label=display_label("thalach"),
                value=None,
            )
        with gr.Column(scale=1, min_width=60):
            thalach_clear = gr.Button("Clear", size="sm")

    with gr.Row():
        with gr.Column(scale=5):
            exang = gr.Dropdown(
                choices=OPTIONS["exang"], label=display_label("exang"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            exang_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            oldpeak = gr.Dropdown(
                choices=OPTIONS["oldpeak"],
                label=display_label("oldpeak"),
                value=None,
            )
        with gr.Column(scale=1, min_width=60):
            oldpeak_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            slope = gr.Dropdown(
                choices=OPTIONS["slope"], label=display_label("slope"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            slope_clear = gr.Button("Clear", size="sm")

    with gr.Row():
        with gr.Column(scale=5):
            ca = gr.Dropdown(
                choices=OPTIONS["ca"], label=display_label("ca"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            ca_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            thal = gr.Dropdown(
                choices=OPTIONS["thal"], label=display_label("thal"), value=None
            )
        with gr.Column(scale=1, min_width=60):
            thal_clear = gr.Button("Clear", size="sm")

        with gr.Column(scale=5):
            num = gr.Dropdown(
                choices=OPTIONS["num"],
                label=display_label("num") + " (can be evidence)",
                value=None,
            )
        with gr.Column(scale=1, min_width=60):
            num_clear = gr.Button("Clear", size="sm")

    run_btn = gr.Button("Run inference", variant="primary")

    gr.Markdown("## Results")

    with gr.Row():
        with gr.Column(scale=2):
            # 使用嵌套的 Row 和 Column 来居中图片
            with gr.Row():
                dist_plot = gr.Image(
                    label="Posterior Distribution Visualization", type="filepath"
                )

    with gr.Row():
        with gr.Column(scale=1):
            output = gr.Markdown(label="Distributions")
        with gr.Column(scale=1):
            joint_output = gr.Markdown(label="Joint Probability")

    run_btn.click(
        fn=ui_predict,
        inputs=[
            variables,
            age,
            sex,
            cp,
            trestbps,
            chol,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
            num,
        ],
        outputs=[output, joint_output, dist_plot],
    )

    age_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[age])
    sex_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[sex])
    cp_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[cp])
    trestbps_clear.click(
        fn=lambda: gr.update(value=None), inputs=[], outputs=[trestbps]
    )
    chol_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[chol])
    thalach_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[thalach])
    exang_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[exang])
    oldpeak_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[oldpeak])
    slope_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[slope])
    ca_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[ca])
    thal_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[thal])
    num_clear.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[num])

    demo.load(
        fn=initialize_display, inputs=[], outputs=[output, joint_output, dist_plot]
    )


if __name__ == "__main__":
    demo.launch()
