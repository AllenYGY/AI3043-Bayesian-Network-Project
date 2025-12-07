# %% [markdown]
# # Heart Disease Bayesian Network Pipeline
# - 数据预处理 + 连续变量离散化
# - 结构学习：PC / HillClimbSearch / 自定义边
# - 参数估计：MLE / Bayesian (BDeu)
# - 推理：Variable Elimination & Clique Tree 示例
#

# %%
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from IPython.display import Image, display
from pgmpy.base import DAG
from pgmpy.estimators import PC, BayesianEstimator, BDeu, HillClimbSearch
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# %% [markdown]
# ## Data preprocessing and discretization
# - 读取 `processed.cleveland.data`，清理缺失值
# - 连续特征：临床阈值分箱（默认）或分位数分箱
# - 输出：原始 df 与离散化 df_disc
#

# %%
# Full column list in raw data (14 cols)
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

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CUSTOM_BINS = {
    # Clinical cutpoints from the provided risk table
    "age": [45, 60],
    "trestbps": [140],
    "chol": [200, 239],
    "thalach": [150],
    "oldpeak": [1.0],
}
RIGHT_OPEN_COLS = {"trestbps", "thalach"}  # threshold stays in upper bin (right-open)

# Full display names for plotting (avoid abbreviations)
DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "ChestPainType",
    "trestbps": "RestingBP",
    "chol": "Cholesterol",
    "thalach": "MaxHeartRate",
    "exang": "ExerciseAngina",
    "oldpeak": "ST_Depression",
    "slope": "ST_Slope",
    "ca": "MajorVessels",
    "thal": "Thalassemia",
    "num": "Diagnosis",
}


def load_df(path: Path, has_header: bool) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else COLUMNS_ALL,
        na_values="?",
    )


def bin_custom(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, cuts in CUSTOM_BINS.items():
        edges = [-math.inf, *cuts, math.inf]
        right = col not in RIGHT_OPEN_COLS
        out[col] = pd.cut(out[col], bins=edges, labels=False, right=right)
    return out


def bin_quantile(df: pd.DataFrame, q: int) -> pd.DataFrame:
    out = df.copy()
    for col in CONTINUOUS:
        out[col] = pd.qcut(out[col], q=q, labels=False, duplicates="drop")
    return out


def preprocess_data(cfg: SimpleNamespace):
    df = load_df(Path(cfg.data), cfg.has_header)
    df = df.drop(columns=DROP_COLS)  # drop unused columns early
    if not cfg.keep_missing:
        df = df.dropna()
    df_disc = bin_custom(df) if cfg.binning == "custom" else bin_quantile(df, cfg.q)
    return df, df_disc


def plot_dag(model, outfile: str = "results/bn_pc.png", layout: str = "spring"):
    """Plot a DAG to PNG, highlighting num; labels use full names."""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    if layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=200)

    colors = ["#e74c3c" if n == "num" else "#3498db" for n in G.nodes()]
    sizes = [1100 if n == "num" else 900 for n in G.nodes()]
    labels = {n: DISPLAY_NAMES.get(n, n) for n in G.nodes()}

    plt.figure(figsize=(9, 7), facecolor="white")
    nx.draw_networkx_nodes(
        G, pos, node_size=sizes, node_color=colors, edgecolors="#34495e", linewidths=1.5
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=10, font_weight="bold", font_color="#2c3e50"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.6,
        edge_color="#2c3e50",
        connectionstyle="arc3,rad=0.05",
        alpha=0.9,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=250)
    plt.close()


def split_train_test(df, test_size: float = 0.2, seed: int = 42):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = max(1, int(len(df_shuffled) * (1 - test_size)))
    return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]


# %% [markdown]
# ### Run preprocessing
# 可在这里修改分箱策略/分位数/缺失值策略。

# %%
cfg = SimpleNamespace(
    data="data/processed.cleveland.data",
    has_header=False,
    binning="custom",  # "custom" 或 "quantile"
    q=4,
    keep_missing=False,
)

df_raw, df_disc = preprocess_data(cfg)
print(f"Raw shape: {df_raw.shape}, discretized: {df_disc.shape}")
print("Unique values per column:")
print(df_disc.nunique())
df_disc.head()


# %%

# Split discretized data for training/testing
split_cfg = SimpleNamespace(test_size=0.2, seed=42)
train_df_disc, test_df_disc = split_train_test(
    df_disc, test_size=split_cfg.test_size, seed=split_cfg.seed
)
print(
    f"Train size: {len(train_df_disc)}, test size: {len(test_df_disc)} (seed={split_cfg.seed})"
)


# %% [markdown]
# ## Structure learning: PC / HillClimbSearch / Custom edges


# %%
def show_graphviz(model, prog: str = "dot", outfile: str = "results/bn_graphviz.png"):
    """用 pygraphviz 渲染并内联显示（标签用全名）。"""
    label_map = {n: DISPLAY_NAMES.get(n, n) for n in model.nodes()}
    try:
        g = model.to_graphviz()
    except Exception as e:
        print(f"to_graphviz failed: {e}")
        return
    try:
        for n in g.nodes():
            g.get_node(n).attr["label"] = label_map.get(n, n)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        g.draw(outfile, prog=prog)
    except Exception as e:
        print(f"Graphviz draw failed: {e}")
        return
    display(Image(filename=outfile))


# %% [markdown]
# ### PC


# %%
def learn_pc(data: pd.DataFrame, alpha: float = 0.05, max_cond_vars=None):
    pc = PC(data)
    dag: DAG = pc.estimate(
        independence_test="chi_square",
        return_type="dag",
        significance_level=alpha,
        max_cond_vars=max_cond_vars,
    )
    model = DiscreteBayesianNetwork(dag.edges())
    model.add_nodes_from(data.columns)
    return model, dag


model_pc, dag_pc = learn_pc(train_df_disc, alpha=0.05)
print("PC edges:")
for e in model_pc.edges():
    print(e)
show_graphviz(model_pc, outfile="results/bn_pc_graphviz.png")


# %% [markdown]
# ### HillClimbSearch


# %%
def learn_hc(data: pd.DataFrame, ess: int = 5):
    score = BDeu(data, equivalent_sample_size=ess)
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=score)
    model.add_nodes_from(data.columns)
    return model


model_hc = learn_hc(train_df_disc, ess=5)
print("HillClimb edges (BDeu score):")
for e in model_hc.edges():
    print(e)
show_graphviz(model_hc, outfile="results/bn_hc_graphviz.png")


# %%
custom_edges = [
    # age
    ("age", "trestbps"),
    ("age", "ca"),
    ("age", "chol"),
    ("age", "thalach"),
    # sex
    ("sex", "chol"),
    ("sex", "cp"),
    ("sex", "thalach"),
    # cp
    ("cp", "exang"),
    ("cp", "num"),
    # trestbps
    ("trestbps", "ca"),
    # chol
    ("chol", "trestbps"),
    ("chol", "ca"),
    # thalach
    ("thalach", "exang"),
    ("thalach", "oldpeak"),
    # exang
    ("exang", "oldpeak"),
    # oldpeak
    ("oldpeak", "slope"),
    # slope
    ("slope", "num"),
    # ca
    ("ca", "num"),
    # thal
    ("thal", "num"),
]

model_custom = DiscreteBayesianNetwork(custom_edges)
model_custom.add_nodes_from(COLUMNS)
print("Custom edges:")
for e in model_custom.edges():
    print(e)
show_graphviz(model_custom, outfile="results/bn_custom_graphviz.png")


# %% [markdown]
# ## Parameter estimation (CPDs)


# %%
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


print(f"Fitting CPDs on train split: {len(train_df_disc)} rows")
fitted_models = {
    "pc": fit_with_bdeu(model_pc, train_df_disc),
    "hc": fit_with_bdeu(model_hc, train_df_disc),
    "custom": fit_with_bdeu(model_custom, train_df_disc),
}

for name, m in fitted_models.items():
    print(f"{name} model CPD count: {len(m.get_cpds())}")

print("Custom model CPD for num:")
print(fitted_models["custom"].get_cpds("num"))


# %% [markdown]
# ## Inference: Variable Elimination / Clique Tree
# 证据的取值需要与离散化后的取值编号一致。

# %%
# 确保有完整 CPD（重新拟合一次自定义结构，可改为 model_pc / model_hc）
bn_for_query = fit_with_bdeu(model_custom, train_df_disc, ess=5)

ve = VariableElimination(bn_for_query)
# BeliefPropagation = clique-tree based exact inference
bp = BeliefPropagation(bn_for_query)

queries = [
    {"variables": ["num"], "evidence": {"age": 2, "chol": 2, "exang": 1}},
    {"variables": ["num"], "evidence": {"cp": 3, "thal": 3}},
]

for q in queries:
    ve_res = ve.query(**q)
    bp_res = bp.query(**q)
    print("\nQuery:", q)
    print("VariableElimination:")
    print(ve_res)
    print("BeliefPropagation (Clique Tree):")
    print(bp_res)


# %% [markdown]
# ## 预测任务：判断心脏病是否发作
# - 移除 num 列后对特征做 8:2 hold-out 拟合 BN，用 MAP 预测 num 以及 P(num>0) 作为发作风险
# - 可选 PC / HC 结构（默认仍使用自定义与专家先验边对关联）
# - 举例：输入一个新病人记录，给出 P(num) 向量和 P(心脏病存在)
#

# %%

FEATURE_COLS = [c for c in COLUMNS if c != "num"]


def evaluate_prediction(
    model_like,
    data: pd.DataFrame,
    ess: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    train_df=None,
    test_df=None,
):
    if train_df is None or test_df is None:
        train_df, test_df = split_train_test(data, test_size=test_size, seed=seed)
    bn = fit_with_bdeu(model_like, train_df, ess=ess)
    infer = VariableElimination(bn)

    true_num = test_df["num"].tolist()
    pred_num = []
    prob_attack = []

    for _, row in test_df.iterrows():
        evidence = row.drop(labels=["num"]).to_dict()
        q = infer.query(variables=["num"], evidence=evidence)
        probs = pd.Series(q.values, index=q.state_names["num"])
        pred_num.append(int(probs.idxmax()))
        prob_attack.append(float(1 - probs.get(0, 0.0)))

    true_attack = [1 if n > 0 else 0 for n in true_num]
    pred_attack = [1 if n > 0 else 0 for n in pred_num]

    acc_num = (pd.Series(pred_num).eq(true_num)).mean()
    acc_attack = (pd.Series(pred_attack).eq(true_attack)).mean()

    print(f"Train size: {len(train_df)}, test size: {len(test_df)} (seed={seed})")
    print(f"MAP accuracy (num 0-4): {acc_num:.3f}")
    print(f"Binary accuracy (heart disease vs no disease): {acc_attack:.3f}")
    print(f"Mean P(heart disease present): {pd.Series(prob_attack).mean():.3f}")

    return bn, pd.DataFrame({
        "true_num": true_num,
        "pred_num": pred_num,
        "prob_attack": prob_attack,
    })


def predict_patient_prob(
    model,
    raw_sample: dict,
    binning: str = "custom",
    q: int = 4,
):
    """对单个样本给出 P(num) 与心脏病概率 (num>0视为有疾)."""
    sample_df = pd.DataFrame([raw_sample])
    missing = set(FEATURE_COLS) - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    sample_df = sample_df[FEATURE_COLS]

    if binning == "custom":
        sample_disc = bin_custom(sample_df)
    else:
        if q < 2:
            raise ValueError("q should be >=2 when using quantile binning")
        sample_disc = bin_quantile(sample_df, q=q)

    evidence = sample_disc.iloc[0].to_dict()
    qres = VariableElimination(model).query(variables=["num"], evidence=evidence)
    probs = pd.Series(qres.values, index=qres.state_names["num"])
    prob_attack = float(1 - probs.get(0, 0.0))
    return probs, prob_attack


# %%
# 选择要用的结构：model_custom / model_pc / model_hc
bn_pred, eval_df = evaluate_prediction(
    model_custom,
    df_disc,
    ess=5,
    test_size=split_cfg.test_size,
    seed=split_cfg.seed,
    train_df=train_df_disc,
    test_df=test_df_disc,
)

eval_df.head()


# %% [markdown]
# ## ML Baseline

from sklearn.ensemble import RandomForestClassifier

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def evaluate_ml_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLS].astype(int)
    y_train = train_df["num"].astype(int)
    X_test = test_df[FEATURE_COLS].astype(int)
    y_test = test_df["num"].astype(int)

    models = {
        "logreg_multinomial": LogisticRegression(
            max_iter=500, multi_class="auto", n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "multinomial_nb": MultinomialNB(),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_num = model.predict(X_test)
        prob = model.predict_proba(X_test)
        classes = model.classes_
        attack_cols = [i for i, c in enumerate(classes) if c > 0]
        p_attack = prob[:, attack_cols].sum(axis=1)

        acc_num = (pd.Series(pred_num).eq(y_test)).mean()
        acc_attack = (pd.Series(pred_num > 0).eq(y_test > 0)).mean()
        rows.append({
            "model": name,
            "acc_num": acc_num,
            "acc_attack": acc_attack,
            "mean_p_attack": p_attack.mean(),
        })

    return pd.DataFrame(rows)


# 与 BN 相同的离散化特征与 train/test 划分
ml_results = evaluate_ml_baselines(train_df_disc, test_df_disc)
ml_results


# %%
example_patient = {
    "age": 54,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 243,
    "thalach": 160,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}

probs_demo, p_attack = predict_patient_prob(
    bn_pred,
    example_patient,
    binning=cfg.binning,
    q=cfg.q,
)
print("\nExample patient P(num):")
print(probs_demo)
print(f"P(heart disease present) = {p_attack:.3f}")
