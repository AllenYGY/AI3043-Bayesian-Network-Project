# 心脏病贝叶斯网络 Notebook 实现计划

- **输入数据**：`data/processed.cleveland.data`（无表头，`?` 表示缺失）；预处理阶段会丢弃未用到的 `fbs`、`restecg`，其余列全部参与。
- **离散化**：连续列按临床阈值分箱（默认），可切换分位数分箱并设置 `q`；保留自定义阈值表供调整。

## Notebook 结构（`script.ipynb`）

- 导入与工具函数：载入依赖，定义列名、离散化函数、`plot_dag`。
- 数据预处理：固定参数后读取 `data/processed.cleveland.data`，先 drop `fbs`、`restecg`，再清洗缺失、离散化，得到 `df_raw`/`df_disc`。
- 结构学习：
  - PC（卡方 CI 检验，`alpha`/`max_cond_vars` 可调）。
  - HillClimbSearch（BDeu 评分，默认 ess=5）。
  - 自定义边集合（直接在列表改先验边）。
- 参数学习：`BayesianEstimator` + BDeu(ess=5) 拟合 CPD，示例打印 `num` 的 CPD；可切换 `MaximumLikelihoodEstimator`。
- 推理示例：基于拟合模型（默认自定义版）用 VariableElimination 与 BeliefPropagation（Clique Tree）查询 `P(num | evidence)`。
- 绘图：`show_graphviz`/`plot_dag` 生成的 PNG 均保存在 `results/`。

## 使用方式

- 依次运行单元即可（Notebook 不读取命令行参数）。
- 参数调节：在“运行预处理”单元改数据路径/分箱策略/`q`/缺失策略；在结构学习单元改 PC/HC 参数或自定义边；推理单元替换 `queries` 证据（值需与离散化编号一致）。
- 输出：`results/` 下的图（PC/HC/自定义结构）、CPD 打印输出、VE 与 BeliefPropagation 查询结果对照。
