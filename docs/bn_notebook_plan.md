# 心脏病贝叶斯网络 Notebook 实现计划

- **输入数据**：`data/processed.cleveland.data`（无表头，`?` 为缺失）；预处理阶段会丢弃未用到的 `fbs`、`restecg`。
- **离散化**：连续列按临床阈值分箱（默认），或分位数分箱（`q` 可调）。
- **输出目录**：所有 PNG 写入 `results/`（自动创建）。
- **依赖**：见 `requirements.txt`（pgmpy、pandas、numpy、networkx、matplotlib、ipython、pygraphviz；需系统安装 graphviz）。

## Notebook 结构（`script.ipynb`）

- 导入与工具函数：定义列名、离散化函数、`plot_dag`（matplotlib）、`show_graphviz`（pygraphviz，全名标签）。
- 数据预处理：
  - 读取 `data/processed.cleveland.data`，drop `fbs`/`restecg`。
  - 缺失处理：默认 dropna，可配置保留。
  - 离散化：临床阈值或分位数；输出 `df_raw`/`df_disc` 并打印取值数。
- 结构学习：
  - PC：卡方 CI 检验（可调 `alpha`/`max_cond_vars`）。
  - HillClimbSearch：BDeu 评分（默认 ESS=5）。
  - 自定义边：列表直接编辑。
  - 各模型均生成 Graphviz PNG（`results/bn_pc_graphviz.png` 等）。
- 参数学习：
  - `BayesianEstimator` + BDeu(ESS=5) 拟合 CPD，打印每个模型 CPD 数量与目标 `num` 的 CPD。
  - 如需极大似然，可改用 `MaximumLikelihoodEstimator`。
- 推理（精确）：
  - 使用 VE 与 BeliefPropagation（Clique Tree）对拟合模型查询。
  - 示例查询两组证据；证据值需匹配离散化后的编号。

## 运行步骤

1. 依次运行单元（Notebook 不读 argv）。
2. 可调参数：
   - 预处理：`cfg` 中数据路径、分箱方式(`custom`/`quantile`)、`q`、缺失策略。
   - 结构：PC 的 `alpha`/`max_cond_vars`，HC 的 ESS，自定义边列表。
   - 推理：修改 `queries`；如需换模型，调整 `bn_for_query`。
3. 查看输出：
   - `results/` 下的 PNG（PC/HC/自定义结构）。
   - 参数单元的 CPD 打印输出。
   - 推理单元的 VE/BP 结果对照。

## 目录约定

- 数据：`data/processed.cleveland.data` 及其他 *.data/*.csv。
- 结果：`results/`（自动创建）。
- 规划与说明：`docs/`（包含本计划与列说明、整体概览）。
