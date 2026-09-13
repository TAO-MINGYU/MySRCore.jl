# MySRCore 后端仓库开发日志

记录 Julia 后端、搜索核心、量纲语义、表达式表示和后端测试的重大变更。

## 记录格式

后续重大变更至少记录：日期、变更类型、影响范围、原因、修改路径、结果/验证证据、遗留风险和后续行动。

## 2026-09-04 - 建立文件夹级说明与日志约定

- 变更类型：结构与工作流规范化。
- 影响范围：本开发单元的说明文件、局部约束和日志入口。
- 结果：建立本目录的持续记录入口；未来必要的大幅修改应追加到本文件。
- 验证：目录职责已与 `MySR_Dev/AGENTS.md` 和 `memory/DEVELOPMENT_MEMORY.md` 的总规则对齐。

## 2026-09-05 - 合并静态 mutation affinity 实现

- 变更类型：搜索 mutation 能力实现。
- 影响范围：`src/MutationAffinity.jl`、`src/MutationFunctions.jl`、`src/Mutate.jl`、`src/Options.jl`、`src/OptionsStruct.jl`、`src/Core.jl`、`test/runtests.jl`。
- Decision：point operator/feature mutation 先使用 `formula_type` 生成合法候选，再按静态家族亲和度与均匀探索抽样；默认保留 `:family`，可用 `:none` 或矩阵覆盖；不加入历史 transition 学习。
- Confirmed：实现已从隔离 worktree 应用到正式 feature checkout；隔离 worktree 的直接测试与 `Pkg.test()` 全部通过。
- 验证：待正式 checkout 重新运行直接测试与包测试后补充最终证据。
- 遗留风险：默认强度 `4.0` 与探索比例 `0.2` 尚未通过匹配预算消融验证，不作性能提升声明。

## 2026-09-05 - 优化 mutation affinity 热路径

- 变更类型：性能与代码质量优化。
- 影响范围：`src/MutationAffinity.jl`、`src/MutationFunctions.jl`。
- Decision：保持原有亲和度/探索概率语义，改为无临时向量的在线加权抽样；将 operator family 判断改为直接分支；feature affinity 尺寸在树遍历前校验。
- Confirmed：热路径 `sample_affinity_target` 稳态分配为 0 bytes；同进程等价旧实现为 192 bytes。优化提交为 `51eed31`。
- 验证：优化后 MySRCore 完整直接测试全部通过；Python 量纲/接口测试 `16 passed`。
- 遗留风险：尚未进行大规模搜索性能或最终 HOF 收益 benchmark；当前仅证明局部分配减少与回归行为保持。

## 2026-09-05 - 修复 RNN-GPSR unary token 空集合错误

- Confirmed：`_expression_tokens` 在一元根节点上对空低阶算子元组执行 `sum`，导致 Julia `ArgumentError`；该错误在 RS Job 29580 的 `extrap-sigmoid` 任务中复现。
- Decision：对 `degree == 1` 显式使用零偏移，保持二元及更高阶 token 编码不变。
- 修改路径：`src/PopulationSeeding.jl`、`test/runtests.jl`。
- 验证：新增一元根 token 回归断言；同步后的 RS direct smoke 已无该崩溃，完整 Julia 包测试待后续远程验证。

## 2026-09-05 - Unary token regression verification

- **Confirmed**：`_expression_tokens` 的一元根偏移修复已在正式 checkout 生效，二元及更高阶
  token 编码保持原逻辑。
- **Verification**：`JULIA_DEPOT_PATH=/tmp/mysr-julia-test-depot julia --project=. -e
  'using Pkg; Pkg.test()'` 通过，包含 RNN-GPSR population seeding、unary token、维度门控和
  mutation affinity 全部测试集；远程 direct smoke4 亦未再现 empty reduction。

## 2026-09-06 - Formal direct-Slurm regression evidence

- **Confirmed**：正式 direct Slurm run 的 220 个 solver task 未再现此前 RNN-GPSR unary-root
  empty reduction；`extrap-sigmoid` 等含 unary operator 的任务均能完成或按 search timeout
  正常终止。
- **Verification**：本地 `Pkg.test()` 仍通过，unary-root regression 断言保留；远程 v3 bundle
  保存每个 task 的 raw/checkpoint 与 stdout/stderr checksum，未把 candidate-level scoring
  failures 误记为 MySRCore task crash。

## 2026-09-06 - RNN-GPSR 反馈语料切换

- **Confirmed**：`_append_feedback_examples!` 支持在第一次有效后端反馈时替换 structural bootstrap，后续轮次继续累积真实反馈；非有限成员不会触发清理。
- **Verification**：新增反馈替换 Julia 测试；`Pkg.test()` 全部通过。
- **Unknown**：真实 benchmark 中反馈切换对 proposal quality 的提升尚未测量。

## 2026-09-06 - Data-aware structural bootstrap evaluation

- **Confirmed**：`_independent_training_corpus` evaluates each valid bootstrap tree with the active dataset loss and returns the count through `build_rnn_gpsr_seed_pool`; a small complexity term only breaks exact loss ties. The API keeps the old two-value return form unless `return_evaluations=true` is requested.
- **Verification**：full `Pkg.test()` passed after updating the proposal-budget assertion to include bootstrap evaluations; direct RS run 30052 uses the synchronized source hash.
- **Unknown**：whether target-aware bootstrap improves final HOF recovery versus PySR is deferred to the paired deep benchmark.

## 2026-09-07 - Preserve RNN bootstrap for undersized feedback

- 变更类型：RNN-GPSR 回归修复。
- **Confirmed**：RS run 30052 中 `population_size=27` 与 `feedback_fraction=0.2` 产生 6 条首轮反馈，
  清空 8 条 bootstrap 后触发 Python RNN-GPSR 的最小样本异常。
- **Decision**：新增最小训练语料常量 8；首次反馈少于 8 条时保留 bootstrap 并追加反馈，只有有效反馈
  至少 8 条时才允许替换 bootstrap。
- **影响路径**：`src/PopulationSeeding.jl`、`test/runtests.jl`。
- **Verification**：MySRCore `Pkg.test()` 通过；RNN-GPSR 反馈轮次回归断言确认实际配置下 27→33 条语料，完整
  RNN-GPSR Python 测试 44 passed。
- **Unknown**：修复后的远程 benchmark 恢复率与最终 HOF 收益尚未测量。

## 2026-09-07 - 1.1.1 synchronized backend release

- **Confirmed**：Project version and MySRCore source snapshot are released as `v1.1.1`.
- **Verification**：commit `c31efde77250f8acafc2d331a91cfdb0b9e969e4` and tag `v1.1.1`
  were pushed; the remote benchmark run root uses this snapshot.
- **Unknown**：paired recovery and HOF metrics remain pending Slurm array `30489`.

## 2026-09-08 - 1.1.2 benchmark release record

- **Confirmed**：MySRCore project version, changelog and source snapshot are released as
  `v1.1.2` (commit `d2f640a`).
- **Decision**：the four-group ablation reuses this backend identically for AFE, RNN-GPSR and
  empty MySR; only frontend capability toggles differ, preserving a matched backend/resource
  comparison.
- **Verification**：the release regression suite had passed before the benchmark snapshot;
  remote task outcomes remain Unknown until Slurm completion.

## 2026-09-09 - Harden empty RNN-GPSR proposal callback

- **变更类型**：RNN-GPSR 回调边界修复。
- **Confirmed**：外部 RNN callback 在某轮没有可用 proposal 时可能返回 `nothing`；原实现会在
  proposal 遍历阶段抛出错误，使后续合法随机回退无法执行。
- **Decision**：`_generate_proposal_trees` 将显式 `nothing` 规范化为空 proposal batch，继续使用
  已有 grammar/dimension-aware random fallback 补齐请求数量；其他非序列返回值仍按契约报错。
- **修改路径**：`src/PopulationSeeding.jl`、`test/runtests.jl`。
- **Verification**：`Pkg.test()` 全部通过，新增 empty-callback 回归为 2/2；临时可写 Julia
  depot 下完整 MySRCore 测试通过。
- **Residual/Unknown**：未改变 RNN 训练策略或预算；修复后的远程 HOF 影响尚未测量。

## 2026-09-11 - 1.1.3 synchronized backend release preparation

- **变更类型**：发布同步与版本对齐。
- **Confirmed**：`Project.toml` 版本号更新为 `1.1.3`，`CHANGELOG.md` 与 `FORK_CHANGES.md`
  记录该版本元数据同步点；等待与前端 MySR 1.1.3 配套发布。
- **Decision**：后端不再引入本次提交外的功能变更，默认行为沿用 `1.1.2` 已验证路径；
  后续能力差异主要通过前端参数预算与功能门控实验再度验证。

## 2026-09-11 - Reconcile MySRCore 1.1.3 into isolated mutation-affinity worktree

- **变更类型**：隔离 worktree 合并与冲突调和。
- **Confirmed**：原始 MySRCore `3756a51` 的 1.1.3 元数据、`Configure.jl` worker
  package-loading 修复、RNN-GPSR fallback/反馈语料修复已合入
  `feature/mutation-affinity-reconcile-v1.1.3`；静态 mutation-affinity 与
  `formula_type` 合法性门控仍保留。
- **Decision**：采用 1.1.3 的无临时向量在线 affinity 抽样实现；保留
  `feature_affinity` 尺寸在 mutation 遍历前校验；日志合并保留后端历史记录。
- **影响路径**：`src/MutationAffinity.jl`、`src/MutationFunctions.jl`、
  `src/Configure.jl`、`src/PopulationSeeding.jl`、`src/Options.jl`、
  `src/OptionsStruct.jl`、`src/Mutate.jl`、`src/Core.jl`、`test/runtests.jl`、
  `Project.toml`、`CHANGELOG.md`、`FORK_CHANGES.md`。
- **Verification**：隔离 `env_mysr` + Julia 1.10.3 下直接运行 `test/runtests.jl`
  全部通过；mutation-affinity 9/9、RNN-GPSR 与量纲回归均通过。包级 `Pkg.test()`
  与 Python bridge 测试待本次提交后继续执行。
- **Backup**：合并前 HEAD 已保存为
  `backup/mutation-affinity-pre-reconcile-20260911`；原始 MySR 与 MySRCore
  checkout 未修改。
- **Unknown**：未执行大规模 benchmark；affinity 默认强度/探索比例的效果仍不作
  性能声明。

## 2026-09-12 - Fix strict-dimensional RandomizeMutation expression wrapping

- **Confirmed**：在 1.1.3 与 mutation-affinity worktree 合并后的前端回归中，
  `formula_type=:theoretical` 的 `RandomizeMutation` 会从量纲生成器得到裸
  `Node`，但 `MutationResult{N,P}` 要求返回原始 `AbstractExpression` 类型，
  因此曾触发类型错误。
- **Decision**：变异入口现在先取得表达式的 mutation contents/context，并在量纲
  生成成功后按同一 context 重新包装；`TemplateExpression` 等嵌套表达式沿内容
  上下文递归包装，量纲生成失败时仍使用原有随机回退路径。量纲合法性仍由
  `formula_type` 的候选生成/检查负责，未把量纲混入 affinity 分数。
- **影响路径**：`src/Mutate.jl`、`test/runtests.jl`。
- **Verification**：直接 `test/runtests.jl` 全部通过；`Pkg.test()` 全部通过；
  MySR 前端 `test_dimensional_formula_type.py` 与 `test_rnn_gpsr_seeding.py`
  共 `62 passed`（1 个 sklearn 收敛警告）；强制 randomize 的理论量纲小型
  bridge smoke 通过且确认运行时源码来自本 worktree；配置/量纲轻量前端集成
  另有 `6 passed`。
- **Residual/Unknown**：完整的高预算 `test_dimensional_constraints` 在本次
  bridge 启动的 300 秒上限内未完成；小型同路径 smoke 已通过。前端旧测试
  `test_mutation_and_plugin_configuration` 仍假设顶层 `SymbolicRegression`
  包名，而当前 MySRCore 公开边界是 `MySRCore.SymbolicRegression`，未在本
  后端修复中改变该测试/兼容层；`test_dimension_propagation` 仍使用默认
  `formula_type="empirical"`，与当前“formula_type 是量纲模式唯一来源”的
  决策不一致，未将其失败解释为本次后端回归。
- **Backup**：本修复前的 worktree HEAD 保存在
  `backup/reconcile-before-randomize-fix-20260911`；工作分支为
  `feature/mutation-affinity-reconcile-v1.1.3-fix`，修复提交为 `6c2049c`。

## 2026-09-12 - Smoke-test fixes for dimensional and template mutation paths

- **变更类型**：隔离 worktree 中的冒烟测试驱动 bug 修复。
- **Confirmed**：半理论模式的直接 `RandomizeMutation` 现在会先暂存并解包外层
  `C_dim`，完成内部随机化后按原系数重新包装；经验模式的
  `dimensional_scale_coefficient(::AbstractExpression, ...)` 不再误触发
  `get_tree`。
- **Confirmed**：`TemplateExpression.get_tree` 不再因 `zip` 截断
  `f(x, y)` 的变量；多 inner expression 使用声明特征数的总和；固定的常见
  combiner 算子（如 `sin`）仅在临时 AST 视图中补齐，不改变存储的搜索算子集合。
- **影响路径**：`src/DimensionalAnalysis.jl`、`src/Mutate.jl`、
  `src/TemplateExpression.jl`、`test/runtests.jl`。
- **Verification**：后端 `test/runtests.jl` 全部通过（新增模板回归 11/11，
  `C_dim`/量纲/affinity 回归均通过）；`env_mysr` 前端量纲与 RNN 测试
  `62 passed`（1 个 sklearn 收敛警告）；模板主流程集成测试 `1 passed`
  （95.72 秒）；`git diff --check` 通过。
- **Residual/Unknown**：默认 200 iterations × 62 populations 的两个
  fresh-process 模板测试运行超过 6 分钟后按冒烟范围安全中止；任意用户自定义
  固定 combiner 算子尚未自动发现；一个 type-spec 测试因隔离 Julia project
  没有旧包名 `SymbolicRegression` 而失败，未归因于本次后端改动。
- **Backup/Branch**：当前工作分支为
  `feature/mutation-affinity-smoke-fix-20260912`，本轮前备份为
  `backup/smoke-before-fix-20260912`；原始 MySR/MySRCore checkout 未修改，
  未执行远程操作。

## 2026-09-05 - 建立 crossover 优化隔离 worktree

- 变更类型：算法研究后的隔离实现。
- 影响范围：`src/Crossovers.jl`、`src/Core.jl`、`src/SymbolicRegression.jl`、`src/Crossover.jl`、`src/MutationFunctions.jl`、`test/runtests.jl`、`CROSSOVER_OPTIMIZATION_PLAN.md`。
- Decision：新增显式 `SizeMatchedCrossover(; size_tolerance=0.25)`；第一棵树沿用均匀节点采样，第二棵树一次 bottom-up 收集子树节点数，优先匹配相对容差内的候选，无候选时选择最近尺寸。默认 `SubtreeCrossover() => 1.0` 保持不变。
- Confirmed：普通 `Expression` 使用新匹配逻辑；TemplateExpression、共享图节点及其他自定义包装回退既有 `crossover_trees`，保留原有上下文和共享语义；节点复制避免新增 aliasing；量纲包装和外层约束检查未绕过。
- 验证：env_mysr 的 Julia 1.10.3 环境核验通过；focused crossover test 52/52 通过；完整 `test/runtests.jl` 已通过当前测试集；`git diff --check` 通过。
- Unknown：相对节点数匹配是否改善 HOF、测试误差、树膨胀或 wall-clock 尚未 benchmark；本 worktree 不合并回原 checkout。

## 2026-09-05 - crossover 代码质量与资源使用优化

- 变更类型：内部实现重构与候选选择优化。
- Decision：`SizeMatchedCrossover` 在单次 donor 遍历中完成容差候选和最近尺寸的 reservoir selection，移除 `distances`/`findall` 中间数组；公共 helper 对非法 tolerance 统一快速失败。
- Decision：抽取 `_crossover_with_tree_operator`，统一普通 crossover 与 size-matched crossover 的 dimensional-scale 解包、恢复和 trace 路径。
- Confirmed：默认 `SubtreeCrossover` 行为和外层约束接口保持不变，未新增依赖或额外 loss evaluation。
- 验证：重新运行当前完整 Julia 测试文件，所有测试集通过；`git diff --check` 通过。
- Unknown：分配次数和 wall-clock 的实际下降尚未用 profiler 量化；需要后续匹配 benchmark 才能确认资源收益。

## 2026-09-05 - 验证计数修正

- **Confirmed**：新增公共 helper 非法 tolerance 回归断言后，Size-matched crossover focused test 当前为 **53/53**；完整当前 Julia 测试文件仍全部通过。

## 2026-09-12 - 同步本地 MySRCore/MySR 代码到 crossover worktree

- 变更类型：跨仓库本地代码同步与冲突调和。
- **Confirmed**：canonical MySRCore `b92776a` 合并到 `feature/crossover-local-sync-20260912`，合并提交为 `08773e9`；保留 SizeMatchedCrossover、mutation-affinity、1.1.3、量纲和模板修复。
- **Confirmed**：唯一合并冲突是 `DEVELOPMENT_LOG.md` 的 add/add，已保留两边日志；源码文件无未解决冲突标记。
- **Confirmed**：配套 MySR worktree `/home/taomingyu/MySR_Dev/worktrees/crossover-optimization-python` 基于 canonical `a7c787b`，其 `juliapkg.json` 保持 1.1.3 发布配置；桥接测试使用临时 dev 配置指向本地 backend worktree。
- **验证**：MySRCore `Pkg.test()` 全部通过；MySR 前端量纲/RNN 聚焦测试 `62 passed`，仅有既有线程配置和 sklearn 收敛警告；两个 worktree `git diff --check` 通过。
- **Backup**：canonical MySRCore 和 MySR 均建立 `backup/local-sync-before-crossover-merge-20260912`；crossover worktree 建立 `backup/crossover-before-local-sync-20260912`。
- **Unknown**：未运行大规模搜索或 benchmark；原始 checkout 的未跟踪 `AGENTS.md`/`outputs/` 保持不动。

## 2026-09-12 - 修正 size-matched custom expression fallback

- **Confirmed**：复核本地代码优先合并后的差异时发现，`size_matched_crossover_trees(::AbstractExpression, ...)` 的 tolerance 参数曾匿名声明却在函数体引用，TemplateExpression/custom wrapper 会触发 `UndefVarError`。
- **Decision**：恢复具名参数并新增 TemplateExpression fallback 的非法 tolerance 回归断言；修复提交为 `5eed25a`，修复前备份为 `backup/pre-size-matched-fallback-fix-20260912`。
- **验证**：MySRCore `Pkg.test()` 全部通过，TemplateExpression 回归为 `12/12`；未改变 canonical checkout。

## 2026-09-12 - 三次基础测试第 1/2 轮

- **Confirmed**：静态加载测试在使用可写临时 Julia depot 后通过；首次失败来自 `env_mysr` 只读 depot 的 precompile pidfile，而非源码。
- **Confirmed**：完整 MySRCore `Pkg.test()` 通过，Size-matched crossover test 当前 `66/66`；未发现代码 BUG。
- **Decision**：增加尺寸选择器的精确命中、最近尺寸 fallback 和并列候选覆盖；下一步补充公共构造器文档与 `Inf` 边界测试。
- **Unknown**：上游 DynamicExpressions 的 `OperatorEnum` 弃用警告仍存在，未归因于本次 crossover。

## 2026-09-12 - 三层基础测试完成

- **Confirmed**：静态加载层在可写临时 depot 下通过；完整 MySRCore `Pkg.test()` 通过，Size-matched test `66/66`；MySR 前端本地 backend 桥接测试 `62 passed`。
- **Confirmed**：三层测试均未发现源码 BUG；警告仅为 env depot 只读导致的首次假失败、上游 `@nospecialize`/`OperatorEnum` 弃用提示、线程配置提示和 sklearn 收敛提示。
- **Decision**：将三层测试命令和 depot 规则写入 crossover plan；当前不再修改已通过的核心运行逻辑，后续性能工作需进入 profiler/匹配 benchmark。

## 2026-09-12 - 三层测试复跑与 aliasing 质量覆盖

- **Confirmed**：按三层协议复跑，静态加载通过；MySRCore `Pkg.test()` 通过，Size-matched test `79/79`；MySR 前端量纲/RNN 聚焦测试 `62 passed`。
- **Decision**：未发现源码 BUG；为 crossover 增加父子节点 object identity 不重叠的 aliasing 回归测试，防止后续 mutation 通过共享节点修改父代。提交为当前后续提交。
- **Unknown**：上游弃用与 sklearn 收敛警告仍未解决，未归因于本项目改动。

## 2026-09-12 - 最终代码质量审查

- **Confirmed**：完成静态加载、完整 MySRCore 回归和 MySR 前端桥接检查；最终 backend `Pkg.test()` 全部通过，SizeMatchedCrossover 回归 `81/81`，前端量纲/RNN 测试 `62 passed`。
- **Decision**：为内部尺寸选择器增加空 donor 与非正 target 的显式参数校验，避免未来扩展时产生索引 0 或隐晦错误；当前提交为本次最终质量改动。
- **Unknown**：上游弃用提示、线程配置提示和 sklearn 收敛提示仍未解决；性能收益仍需 profiler/benchmark。

## 2026-09-12 - 最终提交后的前端桥接复核

- **Confirmed**：在最终 backend 提交 `16ac155` 之后重新运行 MySR 前端量纲/RNN 桥接测试，结果为 `62 passed`（88.14s）。
- **验证**：测试通过；仅保留线程配置和 sklearn 收敛警告，未发现由本次 crossover 改动引入的失败。

## 2026-09-12 - 最终提交后的静态加载复核

- **Confirmed**：在最终提交 `a786901`（包含 backend `16ac155`）上，用 `env_mysr` 和可写临时 depot 加环境 depot 的配置重新执行 `using MySRCore`，输出 `final-static-load-ok`。
- **分析**：首次只使用空临时 depot 时因缺少已安装的 `Reexport` 依赖而失败；补充环境 depot 后通过，确认是测试环境配置问题而非源码问题。

## 2026-09-12 - 深度 crossover 质量加固

- **Confirmed**：尺寸选择器现在在内部入口也验证 finite、nonnegative tolerance，避免绕过公共构造器时接受 NaN 或负值。
- **Confirmed**：新增负值/NaN selector 回归，以及 child1/child2 之间无节点共享的 aliasing 回归；提交为 `e354e77`，修改前备份为 `backup/pre-selector-contract-20260912`。
- **Confirmed**：补充 `SizeMatchedCrossover` 的公开使用文档、fallback 语义和 API 文档索引；提交为 `b696f0f`，文档修改前备份为 `backup/pre-crossover-docs-quality-20260912`。
- **验证**：MySRCore `Pkg.test()` 全部通过，Size-matched crossover `95/95`；公共 API 检查输出 `public-crossover-api-ok`；`git diff --check` 通过。
- **Unknown**：本轮仍未测量大规模搜索性能收益；性能结论需要独立 profiler/匹配 benchmark。

## 2026-09-12 - 深度质量改动后的前端桥接复核

- **Confirmed**：使用当前 backend worktree（包含 `e354e77`、`b696f0f`、`8e8f8f4`）生成临时 dev juliapkg 配置并运行 MySR 量纲/RNN 聚焦测试，结果 `62 passed`（98.03s）。
- **验证**：仅有既有 sklearn 收敛警告；未发现 selector 边界加固或文档变更造成的前端桥接回归。

## 2026-09-12 - 本地 MySRCore 集成后桥接复核

- **Confirmed**：本地 `MySRCore.jl` 集成提交为 `d210713`，其 Julia backend 完整测试通过，SizeMatchedCrossover `95/95`。
- **Confirmed**：临时 Python bridge 配置直接指向 `/home/taomingyu/MySR_Dev/MySRCore.jl`，MySR 量纲/RNN 聚焦测试 `62 passed`（97.89s）。
- **验证**：仅有既有 sklearn 收敛警告；本地集成目录可被 Python 前端正常加载。

## 2026-09-12 - Nonnumeric TypeSpec dimensional-scale guard

- **Confirmed**：`wrap_dimensional_scale`、mutation 和 crossover 在兼容量纲策略下对非数值 TypeSpec 使用 `one(T)`，字符串/向量等类型会在初始化或演化阶段抛出 `MethodError`。
- **Decision**：新增 `dimensional_scale_identity` 能力检查；无乘法单位元时跳过外部尺度包装，并让量纲过渡、mutation、crossover 避免 eager `one(T)` 求值。提交 `e22054f`，备份分支 `backup/pre-typespec-nonnumeric-scale-20260912`。
- **Verification**：在 `env_mysr`、可写临时 Julia depot 加环境 depot 下运行 `Pkg.test()`，所有 MySRCore 测试集通过。
- **Residual/Unknown**：TemplateExpression 自定义 combiner 的多特征映射仍有独立越界失败，需后续聚焦修复；本次不改变数值型半理论 C_dim 行为。

## 2026-09-12 - Merge canonical local code into crossover worktree

- **Decision**：以 canonical MySRCore `main` 为代码主线合入本工作分支，保留已验证的
  `SizeMatchedCrossover` 及其边界/aliasing 回归；采用 canonical 的量纲尺度 identity
  处理，避免旧实现 eager `one(T)`。
- **Confirmed**：合并提交为 `b1fdf0c`；源码无未解决冲突，除 worktree 自带 `AGENTS.md`
  外与 canonical `main` 的差异仅为 crossover 扩展。
- **Verification**：env_mysr + Julia 1.10.3 下 `Pkg.test()` 通过；SizeMatchedCrossover
  回归 `95/95`，其余量纲、mutation-affinity、RNN-GPSR、TemplateExpression 测试全部通过。
- **Unknown**：未运行大规模搜索或性能 benchmark；尺寸匹配对最终 HOF 的收益仍待独立实验。

## 2026-09-12 - Three basic test rounds and public-dispatch quality coverage

- **Confirmed**：第一轮静态加载与 Python compileall 通过；第二轮后端完整
  `test/runtests.jl` 通过，新增公共 `crossover(...)` dispatch 回归后
  SizeMatchedCrossover 测试为 `98/98`。
- **Decision**：无源码 BUG 时采用低风险质量计划，补充公共入口的类型、树大小和父代隔离断言，避免只验证内部 helper。
- **Verification**：第三轮前端 bridge 使用临时 Julia project 指向本 worktree，量纲/RNN 测试 `62 passed`；最终 `git diff --check` 通过。
- **Unknown**：仍未测量大规模搜索性能或 HOF 收益；既有上游弃用警告未处理。

## 2026-09-12 - Full quality audit conclusion

- **Confirmed**：完成分支/状态、静态加载、完整 Julia 回归、Python bridge、compileall、Ruff 和 diff-check 审查；未发现本次合并引入的 BUG。
- **Decision**：将可验证的公共 dispatch 回归作为本轮质量提升；更大范围性能和重构列为后续独立计划，不在无 benchmark 证据时修改核心搜索逻辑。
- **Unknown**：上游弃用提示和大规模搜索性能仍需单独处理。

## 2026-09-13 - Low-risk mutation hot-path optimization

- **Decision**：保持 mutation 的合法候选筛选、静态 affinity 和等概率节点选择语义不变，
  将 `mutate_operator` 与 `mutate_feature` 的候选节点列表复制/随机打乱改为单次遍历
  reservoir sampling，减少热路径临时分配。
- **Confirmed**：优化提交 `6084ef8`，已合入 canonical 集成分支提交 `d12d9d6`；合并前备份
  为 `backup/pre-merge-performance-quality-20260913`。
- **Verification**：canonical `Pkg.test()` 全部通过，SizeMatchedCrossover `98/98`；Python
  bridge 量纲/RNN `62 passed`。
- **Unknown**：未用 profiler 量化总体吞吐或分配下降；其他模块仍需按相同基线逐项优化。

## 2026-09-13 - Linear-time Hall of Fame frontier scan

- **Decision**：保持 HallOfFame 的按复杂度最低 loss 和 `copy(member)` 防护语义，
  将 `calculate_pareto_frontier` 的逐复杂度嵌套比较改为 running minimum 单次扫描。
- **Confirmed**：对有限值、`Inf`、`-Inf` 和 `NaN`，新 predicate 与原
  `member.loss >= simpler.loss` 逐项比较等价；新增非有限 loss 回归测试。实现提交
  `a1bc0ff`（随后由 `e1f2333` 移除对 `zero(L)` 的额外类型要求），独立 worktree 分支为
  `feature/hof-frontier-linear-20260913`，备份分支为 `backup/pre-hof-frontier-linear-20260913`。
- **Verification**：完整 `test/runtests.jl` 全部通过（SizeMatchedCrossover `98/98`）；
  合成 `maxsize=5000` 微基准中 frontier 扫描耗时约降低 8.8 倍。该微基准不代表总体搜索
  吞吐，真实收益仍受成员拷贝和日志频率影响。
