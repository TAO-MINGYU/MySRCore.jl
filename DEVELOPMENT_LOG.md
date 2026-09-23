# MySRCore 后端仓库开发日志

记录 Julia 后端、搜索核心、量纲语义、表达式表示和后端测试的重大变更。

## 记录格式

后续重大变更至少记录：日期、变更类型、影响范围、原因、修改路径、结果/验证证据、遗留风险和后续行动。

## 2026-09-22 - 建立 MySRCore 第一阶段体检与改良思考文档

- 变更类型：设计审查与后续开发入口。
- 影响范围：`MYSRCORE_HEALTH_REVIEW.md`；未修改 Julia 源码、测试、Project/Manifest 或公共 API。
- 目的：围绕 parent selection、mutation/crossover、survival、loss/uncertainty、population profile 和 migration，记录当前实现、与目标默认行为的差距、跨领域问题和后续算法提案。
- 基线：`main@87b5d82`，版本线 `1.1.3`；上游参照为 `SymbolicRegression.jl 2.0.0-beta.8`。
- 证据规则：文档区分 `Confirmed`、`Decision`、`Unknown`、`Proposal`；未把局部测试或 smoke 解释为性能提升证据。
- 验证：已只读核对 `Options`、`ParentSelection`、mutation/crossover、loss、profile/migration、搜索主循环和 `test/runtests.jl`；文档创建后执行 Markdown 内容检查与 `git diff --check`。
- 遗留问题：epsilon 公共 API/default、semantic back-propagation、semantic crossover、默认策略切换、RNG 契约、loss calibration 和匹配 benchmark 均未在本次实现。

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

## 2026-09-13 - Mutation candidate allocation reduction

- **Decision**：在不改变候选合法性、affinity 或随机选择语义的前提下，使用单次遍历
  reservoir sampling 替代 `mutate_operator`/`mutate_feature` 的节点列表复制与 `shuffle!`。
- **Verification**：worktree 完整 `Pkg.test()` 通过，SizeMatchedCrossover `98/98`；提交 `6084ef8`。
- **Unknown**：尚未用 profiler 量化总吞吐收益。

## 2026-09-13 - Linear-time Hall of Fame frontier scan

- **Decision**：保持 HallOfFame 的按复杂度最低 loss 和 `copy(member)` 防护语义，将 `calculate_pareto_frontier` 的嵌套比较改为 running minimum 单次扫描。
- **Confirmed**：有限值、`Inf`、`-Inf` 和 `NaN` 比较语义保持一致；新增非有限 loss 回归。HOF 实现提交 `a1bc0ff`，稳健性补丁 `e1f2333`，合并自独立 worktree。
- **Verification**：合成 `maxsize=5000` 微基准约 8.8 倍加速；完整测试待当前合并提交后复跑。
- **Unknown**：真实搜索总体吞吐收益仍需 profiler/benchmark 量化。

## 2026-09-13 - Focused optimization integration verification

- **Confirmed**：HallOfFame 与 DimensionalAnalysis 改动已合入 `feature/integrate-performance-quality-20260913`（`fa5e764`，随后移除误跟踪的本地 `AGENTS.md` 为 `a8caffa`）。
- **Verification**：canonical `env_mysr` + 可写临时 depot 下 `Pkg.test()` 全部通过；Dimension-only fast paths `6/6`、Hall of Fame nonfinite semantics `1/1`、SizeMatchedCrossover `98/98`。
- **Residual/Unknown**：PopulationSeeding、ConstantOptimization 仍等待 profiler/消融证据；未运行远程大规模 benchmark。

## 2026-09-13 - Local-primary population migration merge

- **Decision**：以 canonical MySRCore 本地分支 `feature/integrate-performance-quality-20260913` 为主线，在隔离 worktree `feature/local-primary-population-migration-merge-20260913` 合入 population profiles、ring/pooled topology 与 best-only/best-plus-novelty policy；保留本地 SizeMatchedCrossover、量纲 fast path、HOF 和 RNN-GPSR 改动。
- **Confirmed**：合并提交 `09b316d`；新增 `IslandProfile`/`ProfiledOptions`、迁移候选解析与后端 Options 字段，未留下冲突标记。
- **Verification**：`env_mysr` + 可写临时 Julia depot 下完整 `Pkg.test("MySRCore")` 通过；SizeMatchedCrossover `98/98`、profile `11/11`、topology/policy `7/7`、novelty `3/3`、search integration `3/3`。
- **Unknown**：profile 偏好和新迁移策略在匹配预算下对 HOF/测试误差/吞吐的收益尚无 benchmark 证据。

## 2026-09-13 - DynamicExpressions constructor warning cleanup

- **Decision**：将 `Options` 内部旧式 `OperatorEnum(; binary_operators=..., unary_operators=...)` 调用改为当前 pair-based 构造器，保持 operator 顺序、helper-function 和旧算子清理语义不变。
- **Verification**：完整 `Pkg.test("MySRCore")` 通过；原 DynamicExpressions 构造器弃用提示不再出现。
- **Residual**：编译阶段仍可能显示 DispatchDoctor/Julia 的 `@nospecialize` 参数数量提示，属于参数很多的 `Options` 包装实现，不是 DynamicExpressions 弃用 API。
## 2026-09-13 - Multi-agent Julia quality audit

- **Confirmed**：迁移 fraction 现在要求 finite 且在 [0,1]，支持显式 RNG；Dataset 拒绝空样本和非法权重；TemplateStructure 拒绝非法 feature/parameter 计数，模板评估对特征行数不足给出 DimensionMismatch。
- **Verification**：完整 `Pkg.test()` 通过，新增 Input validation guards `5/5`；既有 `@nospecialize` 编译提示仍存在但不影响测试。
- **Unknown/Proposal**：多输入 custom combiner 的 ComposableExpression 适配、Dataset 更细的 y 校验和 novelty hash 碰撞保护仍待后续设计。
## 2026-09-14 - Multi-input custom combiner AST support

- **Confirmed**：`@template_spec` now routes non-inner combiner calls through an AST-aware `_template_call`; runtime `ValidVector` semantics remain direct, while `get_tree` records custom operators in a temporary operator vocabulary. Invalid arity produces an explicit `ArgumentError` instead of tuple `BoundsError`.
- **Verification**：MySRCore full `Pkg.test()` passed after the dispatch guard (`b8e3d63`), including custom combiner operator test `3/3`; MySR original `test_template_custom_combiner_infers_num_features` passed.
- **Decision**：AST fallback only handles a `MethodError` raised by the operator dispatch itself (`MethodError.f === op`), so internal user-function errors are not hidden.

## 2026-09-18 - Opt-in epsilon-lexicase parent selection and AFP survival

- **Decision**：在隔离分支 `worktree/parent-selection-20260918` 中增加
  `parent_selection`（`:tournament` / `:epsilon_lexicase`）和
  `survival_strategy`（`:regularized_evolution` / `:age_fitness_pareto`）选项；默认
  保持旧路径。
- **Confirmed**：epsilon-lexicase 使用完整数据、随机 case 顺序和每 case MAD ε；批处理、
  自定义 aggregate loss 或无法安全拆分的 custom elementwise loss 回退 scalar tournament，
  `parent_selection_diagnostic` 返回回退原因。AFP 在 parent+offspring pool 上以 scalar cost
  和 `PopMember.birth` 的新旧顺序执行 Pareto 生存筛选。
- **修改路径**：`src/ParentSelection.jl`、`src/LossFunctions.jl`、`src/Population.jl`、
  `src/RegularizedEvolution.jl`、`src/Tracing.jl`、`src/Options.jl`、
  `src/OptionsStruct.jl`、`src/SymbolicRegression.jl`、`test/runtests.jl`。
- **Verification**：新增 parent-selection testset 13/13；完整 `test/runtests.jl` 通过；
  tiny serial epsilon-lexicase + AFP search smoke 通过；`git diff --check` 通过。
- **Research**：下载论文、哈希和调研结论见 `MySR_Dev/research/parent_selection/`。
- **Unknown/Residual**：四臂 ablation 尚未在 Carbon/RS 运行；搜索质量、完整 HOF、复杂度
  frontier、耗时和内存影响均未作结论。环境 `env_mysr` 通过授权的 `Pkg.instantiate()`
  补齐了该 worktree `Project.toml` 的 Julia 依赖，生成的 `Manifest.toml` 未纳入源码提交。

## 2026-09-18 - Submit parent-selection benchmark

- **Confirmed**：包含本 worktree MySRCore 的 source snapshot 已部署到新的 Carbon run root；
  排除 Python cache 后本地/远端 source hash 一致。
- **Decision**：四个 opt-in arm 使用完整数据的 epsilon-lexicase 或 scalar tournament，
  并与 regularized evolution 或 age-fitness Pareto survival 交叉；MySR 默认策略未改变。
- **Verification**：MySRCore `test/runtests.jl` 全部通过，parent selection testset 为
  `14/14`；Carbon doctor/matched-environment gate 通过；array/reducer 为
  `32620/32621`、`32624/32625`、`32628/32629`、`32632/32633`。
- **Unknown**：远程 reducer 尚未完成，完整 frontier 和搜索质量影响待回收结果后评估。

## 2026-09-18 - Move parent-selection benchmark to node2

- **Decision**：取消旧 Carbon-pinned benchmark jobs，仅保留旧结果目录；不修改其他用户任务。
- **Confirmed**：新的 source snapshot/run root 固定到 node2，node2 为 512 CPU idle 节点。
- **Verification**：新的四组 array/reducer 为 `32696/32697`、`32702/32703`、`32708/32709`、
  `32714/32715`；每组 concurrency=2，首批 8 个 64-CPU array elements 均在 node2 运行。
- **Unknown**：远程 reducer 尚未完成，完整 frontier 和搜索质量影响待回收结果后评估。
## 2026-09-19 - Isolated surrogate-assisted evaluation worktree

- **Decision**：按用户要求，从本地 MySR 与 MySRCore.jl 的 `main` 分别建立成对隔离
  worktree；本次实现只写入 `worktrees/surrogate/`，不触碰
  `worktrees/parent-selection/`。工作分支为 `worktree/surrogate-20260918`，两仓库
  均保留 `backup/pre-surrogate-worktree-20260918`。
- **Confirmed**：MySRCore 新增 opt-in `SurrogateState`/`SurrogateDecision`，使用固定
  probe phenotype（表达式 probe 输出 + complexity）的距离加权 KNN；只有真实
  `eval_cost` 通过的候选进入有界训练集，预测不写入 Hall of Fame。变异和 crossover
  在真实 loss 前执行保守门控，默认 `surrogate_enabled=false`，因此旧配置保持原路径。
- **影响路径**：`src/Surrogate.jl`、`src/SymbolicRegression.jl`、`src/Options.jl`、
  `src/OptionsStruct.jl`、`src/Mutate.jl`、`src/Crossover.jl`、
  `src/RegularizedEvolution.jl`、`src/SingleIteration.jl`、`test/runtests.jl`。
- **说明更新**：`README.md` 的能力表增加了默认关闭的 surrogate-assisted evaluation
  入口说明。
- **Verification**：env_mysr + Julia 1.10.3、临时可写 Julia depot 下，surrogate 单元
  测试 `10/10`（含默认关闭与参数校验），串行小搜索 smoke 成功；直接运行
  `include("test/runtests.jl")` 和包级 `Pkg.test()` 的 MySRCore 全部 testsets 通过，
  `git diff --check` 通过。`parent-selection` 两个 worktree 的 HEAD 保持
  `86da100`/`890cf77`。
- **Residual/Unknown**：surrogate state 当前在每个 `s_r_cycle` worker dispatch 内创建，
  不跨外层 worker state 持久化；KNN 配置、拒绝策略对吞吐和恢复率的收益尚未经过匹配
  benchmark，不能作性能提升结论。MySR Python 尚未新增 surrogate 公共参数，后续需单独
  决定前端桥接契约。
- **Residual**：编译阶段仍可能显示 DispatchDoctor/Julia 的 `@nospecialize` 参数数量提示，属于参数很多的 `Options` 包装实现，不是 DynamicExpressions 弃用 API。
## 2026-09-13 - Multi-agent Julia quality audit

- **Confirmed**：迁移 fraction 现在要求 finite 且在 [0,1]，支持显式 RNG；Dataset 拒绝空样本和非法权重；TemplateStructure 拒绝非法 feature/parameter 计数，模板评估对特征行数不足给出 DimensionMismatch。
- **Verification**：完整 `Pkg.test()` 通过，新增 Input validation guards `5/5`；既有 `@nospecialize` 编译提示仍存在但不影响测试。
- **Unknown/Proposal**：多输入 custom combiner 的 ComposableExpression 适配、Dataset 更细的 y 校验和 novelty hash 碰撞保护仍待后续设计。

## 2026-09-18 - Uncertainty-aware loss presets and RNN objective alignment

- **Decision**：feature branch `feature/uncertainty-loss-v1` 增加 `Options.loss_preset`、`uncertainty_mode`、`robust_delta`、`student_nu`；continuous split-normal 作为 asymmetric Gaussian likelihood，likelihood 仅允许 `loss_scale=:linear`。
- **Confirmed**：LossFunctions 覆盖三类 uncertainty 情境；不对称数组保存在 `Dataset.extra`，并在多输出与 `SubDataset` batch 中按索引切片。Julia wrapper 拒绝非正/非有限 uncertainty 和 weights 混用。
- **Confirmed**：RNN-GPSR bootstrap 读取真实 `PopMember.cost`，与后续 feedback 使用同一 objective/cost contract；不改变 RNN 序列训练 loss。
- **Historical verification (superseded 2026-09-18)**：当时 MySRCore `Pkg.test()` 和 uncertainty `9/9` 通过，但 TypeSpec worker 残余尚未修复；后续 hardening 已以 TypeSpec `51 passed`、`39 subtests passed` 取代该记录。

## 2026-09-18 - Loss and TypeSpec worker hardening

- **Confirmed**：`src/LossFunctions.jl` 现在在 preset uncertainty 路径统一拒绝
  `Dataset.weights`，对称模式缺少 `sigma` 明确抛出 `ArgumentError`；新增 `eval_cost`
  回归验证负 likelihood 的 cost 仍为有限值。数值语义保持原有 preset 定义。
- **Confirmed**：`src/Configure.jl` 接受 package-loaded worker 的 `filename=nothing`，并只在
  local-include 分支要求 source filename；`src/TemplateExpressionMacro.jl` 改用确定性 FNV-1a
  名称，避免 Julia 进程 salt 导致 worker 找不到 combiner；`src/TemplateExpression.jl`
  支持参数化模板的结构 AST 构造。
- **Verification**：`Pkg.test(;coverage=false)` 全部通过，uncertainty testset `15/15`；
  MySR TypeSpec `51 passed`、`39 subtests passed`；提交 `e966720`。
- **Residual/Unknown**：本轮没有远程正式 benchmark；远端 Carbon 仅执行独立 smoke。未跟踪
  `AGENTS.md` 保持未跟踪，未 push。
## 2026-09-14 - Multi-input custom combiner AST support

- **Confirmed**：`@template_spec` now routes non-inner combiner calls through an AST-aware `_template_call`; runtime `ValidVector` semantics remain direct, while `get_tree` records custom operators in a temporary operator vocabulary. Invalid arity produces an explicit `ArgumentError` instead of tuple `BoundsError`.
- **Verification**：MySRCore full `Pkg.test()` passed after the dispatch guard (`b8e3d63`), including custom combiner operator test `3/3`; MySR original `test_template_custom_combiner_infers_num_features` passed.
- **Decision**：AST fallback only handles a `MethodError` raised by the operator dispatch itself (`MethodError.f === op`), so internal user-function errors are not hidden.

## 2026-09-18 - Authorized Julia dependency recovery and bounded frontend run

- **Decision**：为前端完整回归建立独立 Julia project/depot，不改写 `env_mysr` 的冻结
  baseline；当前源码分支继续保留 `e966720`、`dbbec81`、`b42b77f` 的 loss/worker 修复。
- **Confirmed**：隔离项目位于 `$CONDA_PREFIX/test_support/loss-audit-20260918/project`，
  depot 位于同级 `depot`，通过官方 General registry、`JULIA_PKG_OFFLINE=false` 解析并
  预编译 Bumper 0.6.0、Zygote 0.7.12、LoopVectorization 0.12.174、TensorBoardLogger
  0.1.26、SlurmClusterManager 1.1.0 和 ClusterManagers 2.0.0；MySRCore path 指向
  当前 checkout，Julia 版本为 1.10.3。
- **Verification**：此前后端完整 `Pkg.test()`、uncertainty `15/15`、TypeSpec `51 passed`
  和 RNN/migration `57 passed` 仍有效；前端 optional/notebook 聚焦回归 `8 passed`，
  pytest collection 为 `405 tests`。WSL Docker Engine 29.8.1/buildx 0.37.1 与
  `hello-world` 运行验证通过。
- **Environment limitation**：用户因计算资源达到上限中止完整 `pytest -q mysr/test`；该
  进程以 143 退出，未得到完整最终报告，不能据此宣称前端 405 项全通过。未跟踪的
  `AGENTS.md` 保持原状。

## 2026-09-18 - Integrate loss-audit branch into canonical main

- **Decision**：将 `feature/loss-audit-quality-20260918` 快进合并到本仓库 `main`；该 feature 相对 `main` 领先 5 个提交且 `main` 是其祖先，因此不制造额外合并提交。
- **Confirmed**：本地 `main` 与 `origin/main` 均指向 `0da5bd9`；已删除本地及远程 `feature/loss-audit-quality-20260918`，并保留 `backup/pre-main-merge-loss-audit-20260918` 与 `backup/pre-feature-delete-loss-audit-20260918`。
- **Verification**：合并后所有本轮修改的 Julia 文件及 `test/runtests.jl` 均通过 `Meta.parseall`，`git diff --check` 通过；`main` 已成功推送。
- **Scope**：独立的 `worktrees/parent-selection/MySRCore.jl` 及其 `worktree/parent-selection-20260918` 分支未修改、未删除。

## 2026-09-19 - Cross-population surrogate snapshot synchronization

- **Decision**：surrogate state 采用跨 population 的只读 snapshot 同步；每轮所有
  population 报告真实新样本后，由主搜索循环合并成下一轮 snapshot。worker 不共享可变
  surrogate state，也不把预测值写入 HOF。
- **Confirmed**：`SurrogateSnapshot`、`SurrogateReport`、报告去重/有界 FIFO 合并，以及
  `SearchState` 的每输出 snapshot/pending-round bookkeeping 已接入 `s_r_cycle`、warmup
  和主 dispatch。旧的三元组 worker 输出接口保持默认兼容，报告作为第六字段可选返回。
- **修改路径**：`src/Surrogate.jl`、`src/SingleIteration.jl`、`src/SearchUtils.jl`、
  `src/SymbolicRegression.jl`、`test/runtests.jl`。
- **Verification**：env_mysr、Julia 1.10.3、临时可写 depot 下，MySRCore 包级
  `Pkg.test()` 全部通过；新增 snapshot synchronization 与两 population 串行共享
  snapshot 回归通过。没有执行实际性能 benchmark。
- **Unknown**：当前同步粒度为 population round；surrogate 的 evaluations 节省、吞吐和
  恢复率仍需 matched benchmark，不能从本轮测试推出性能提升。

## 2026-09-19 - Sync canonical surrogate main into parent-selection worktree

- **Decision**：将 canonical `main` 的 surrogate、uncertainty-loss、worker-path 和 TypeSpec
  改动合入 `worktree/parent-selection-20260918`，并保留 ParentSelection、epsilon-lexicase
  和 AFP 生存策略；不修改 canonical `main`、其他 worktree 或用户未跟踪文件。
- **Confirmed**：合并提交为 `092c4a0`；备份引用为
  `backup/pre-surrogate-parent-selection-sync-20260919`。`SymbolicRegression.jl` 同时
  include/export `ParentSelection.jl` 与 `Surrogate.jl`；Options 同时保留两套配置契约。
- **Confirmed**：为避免 uncertainty/loss preset 与普通 per-case L2 误配，epsilon-lexicase
  在非默认 loss preset 或 uncertainty mode 下明确回退 scalar tournament，并返回
  `reason=:nonstandard_loss`。
- **Verification**：env_mysr + Julia 1.10.3 + 隔离可写 depot 下完整 `Pkg.test()` 通过；
  parent-selection testset `15/15`，surrogate/快照测试 `10/10`、`2/2`、`4/4`、`2/2`，
  其余既有测试全部通过。surrogate + epsilon-lexicase + AFP 串行组合搜索成功；
  `git diff --check` 通过。
- **Unknown**：完整搜索质量、HOF frontier、evaluations、耗时和资源收益仍需匹配 benchmark，
  本次集成测试不构成性能结论。

## 2026-09-19 - Enable uncertainty-aware epsilon-lexicase parent selection

- **Decision**：内置 `loss_preset` 与 `uncertainty_mode` 现在允许和
  `parent_selection=:epsilon_lexicase` 同时使用；自定义聚合损失、无法推导逐样本语义的
  自定义 elementwise loss，以及 batching 仍安全回退 scalar tournament。
- **Confirmed**：`LossFunctions.jl` 抽取统一的逐样本 preset loss；`eval_loss` 的 aggregate
  与 `eval_case_losses` 使用同一公式。对称/非对称 uncertainty、普通 built-in preset、
  observation weights 和非有限候选均保持既有语义；uncertainty 与 observation weights
  仍由现有校验互斥。
- **修改路径**：`src/LossFunctions.jl`、`src/ParentSelection.jl`、`src/Options.jl`、
  `test/runtests.jl`。
- **Verification**：`env_mysr` + Julia 1.10.3 下完整 `Pkg.test(; coverage=false)` 通过；
  uncertainty testset `16/16`，parent-selection testset `24/24`。实际
  `uncertainty_mode=:symmetry` + `parent_selection=:epsilon_lexicase` 串行搜索成功；
  临时独立 Julia Project 指向本 worktree 的 MySRCore 后，MySR Python bridge 的 uncertainty
  + epsilon-lexicase fit smoke 成功（2 条 equations）。`git diff --check` 通过。
- **Unknown**：本次只验证选择路径、数值一致性和 bridge 可用性；匹配预算下的 HOF、泛化、
  evaluations、耗时和资源收益仍未知，不能据此作性能结论。
- **Backup**：`backup/pre-uncertainty-lexicase-20260919` 保留修改前的 worktree HEAD。

## 2026-09-19 - Follow-up preset case-vector regression coverage

- **Confirmed**：补充 asymmetric Huber/Student-t 与 symmetric Gaussian NLL 的
  `eval_case_losses`/aggregate 一致性回归，并覆盖普通 weighted preset 的 case vector。
- **Verification**：更新后的完整 `Pkg.test(; coverage=false)` 通过；uncertainty testset
  `19/19`，parent-selection testset `24/24`，其余既有 testsets 全部通过。

## 2026-09-19 - Add opt-in competitive age-fitness survival

- **Decision**：新增 `survival_strategy=:competitive_age_fitness`。每个 mutation/crossover
  child 先按 `parent_ref` 与对应 parent 做局部 cost/complexity 比较；通过的 child 与旧
  population 合并后，按 AFP 的 cost/recency 压力筛选，并以 operator/feature structural
  fingerprint 去重；默认策略仍为 `:regularized_evolution`。
- **Confirmed**：失败 mutation、surrogate 拒绝和不优于 parent 的 child 不会替换旧 member；
  population 容量保持不变，trace 的 replacement slot 只记录真正发生的替换。新增
  `competitive_survivor_indices` 作为可测试的 survivor-pool helper。
- **影响路径**：`src/ParentSelection.jl`、`src/RegularizedEvolution.jl`、`src/Options.jl`、
  `src/SymbolicRegression.jl`、`test/runtests.jl`。
- **Verification**：新增 parent/survival 测试 30/30；competitive survival serial search、
  uncertainty + epsilon-lexicase + surrogate + competitive survival 两条 smoke 成功。
  复杂度 tie-break 调整后的完整回归将在本条实现结束前复跑。
- **Unknown**：新替换策略是否提高 HOF 恢复率、测试误差、结构多样性或 evaluations，仍需
  固定任务、seed、budget 的匹配 benchmark；当前测试不构成性能结论。

## 2026-09-19 - Final competitive survival regression verification

- **Confirmed**：复杂度-aware AFP tie-break 调整后，完整 `test/runtests.jl` 仍全部通过；其中
  parent-selection testset 为 `30/30`，uncertainty preset、surrogate、migration、RNN-GPSR、
  dimensional 和 template testsets 均通过。
- **Verification**：使用 `env_mysr`、Julia 1.10.3 和隔离可写 depot
  `/tmp/mysr-parent-survival-julia-depot`；`git diff --check` 通过。
- **Unknown**：未运行大规模搜索或远程 benchmark；新策略的质量与资源收益仍未知。

## 2026-09-22 - Population profile quotas and profile-local migration

- 变更类型：population API 与 migration 契约重构。
- Decision：新增 `PopulationProfileGroup` 与 `population_profile_groups`，share 总和必须为 1，Options 用最大余数法生成 population-local `IslandProfile` 映射；保留旧 `population_profiles` 逐 population 输入。
- Decision：删除 `migration_topology`；普通 migration 由 profile 组内随机 source population 驱动，`migration_policy` 继续负责组内候选筛选；HOF migration 保留全局 frontier 来源并按目标 profile compatibility 过滤。
- 影响路径：`src/OptionsStruct.jl`、`src/Options.jl`、`src/PopulationMigration.jl`、`src/Migration.jl`、`src/SymbolicRegression.jl`、`src/Core.jl`、`test/runtests.jl`。
- 验证：使用临时 Julia project/depot 指向本 worktree，完整 `test/runtests.jl` 通过；profile quota 9/9、migration novelty 4/4、search integration 3/3；`git diff --check` 通过。
- Unknown：没有运行 matched benchmark，不能据此声明搜索质量或资源收益改善。

## 2026-09-22 - Population migration quality audit

- **修复**：未配置 population profile 时，migration compatibility 现在传递 `nothing`，不再把全局 `Options.operator_affinity` 误当作目标 profile，从而保持默认/HOF 路径的兼容行为。
- **修复**：`PopulationProfileGroup` 在转换为 `Float64` 后重新检查有限性和正性；quota 取整前归一化通过容差检查的 share 总和，避免大 population 下出现无效剩余数。
- **Verification**：完整 `test/runtests.jl` 通过；quota `17/17`、profile-group search integration `9/9`、MySR population migration bridge `4 passed`，Julia parse、Python compileall 与 diff check 均通过。

## 2026-09-22 - Core quality v1: explicit epsilon, RNG streams, and safety hardening

- **范围**：在 `feature/core-quality-v1-20260922` 隔离 worktree 按
  `MYSRCORE_HEALTH_REVIEW.md` 实施第一阶段质量改良；canonical checkout 的源码和既有治理文档未被覆盖。
- **API/算法**：Options 新增 `epsilon` 与 `epsilon_mode`（`:mad` 自适应默认、`:absolute`、`:relative`），并导出 `DEFAULT_EPSILON`/`DEFAULT_EPSILON_MODE`；保留旧 tournament 与 regularized-evolution 默认。新增默认关闭的 `SemanticBackpropMutation`，用 inverse semantic target 的稳健常数替换作为安全基线；mutation/crossover/parent-selection/cycle/optimization/migration 的核心随机决策改用显式 RNG 流。
- **稳定性**：Options 关键约束改用 `ArgumentError`；AFP 在每轮缓存 dominance counts；survival/migration 结构哈希命中后追加递归结构确认，避免 hash collision 丢失候选；profile affinity 校验每个 source row 至少有正权重。
- **影响路径**：`src/Options.jl`、`src/OptionsStruct.jl`、`src/ParentSelection.jl`、`src/Mutations.jl`、`src/MutationFunctions.jl`、`src/Mutate.jl`、`src/Crossover.jl`、`src/Population.jl`、`src/RegularizedEvolution.jl`、`src/SingleIteration.jl`、`src/Migration.jl`、`src/PopulationMigration.jl`、`src/SymbolicRegression.jl`、`test/runtests.jl`。
- **Verification**：env_1_mysr 的 Julia 1.10.3，临时可写 depot/project 指向此 worktree；完整 `Pkg.test("MySRCore"; coverage=false)` 通过，Parent selection testset `39/39`，其余既有 testsets 全部通过；seed=42 的小型 serial search 重复运行表达式序列一致；`git diff --check` 通过。
- **Local microbenchmark**：64×128 epsilon-lexicase absolute threshold 中位数约 `3.09e-5 s`；population=1、population_size=10、ncycles=4 的小型 serial search 热身后 3 次约 `0.00355–0.00413 s`（中位数 `0.00358 s`）。这些是诊断性本机数字，不代表 matched benchmark 或默认策略收益。
- **边界/遗留**：semantic backprop 当前只对 `AbstractExpressionNode{T,2}` 做安全常数替换，默认关闭；跨线程/跨进程只保证独立 RNG 流和统计一致性，仍需正式 P/M matched benchmark 后决定是否切换默认 parent/survival 或提高新 mutation/crossover 权重；Python bridge 尚未新增 epsilon 参数映射。

## 2026-09-22 - Default policy v2 and search-path quality pass

- **默认策略**：当前 v2 与 Python 默认改为 `epsilon_lexicase` + `age_fitness_pareto`；显式
  `tournament`、`regularized_evolution` 和 `competitive_age_fitness` 保持可选。小于
  `2.0.0-` 的 versioned defaults 保留旧策略，避免历史配置静默改变。
- **契约修复**：策略未显式传入时由 versioned default profile 解析，消除 `Options` 构造器、
  `default_options()` 和 Python bridge 的默认漂移；导出 `DEFAULT_PARENT_SELECTION` 与
  `DEFAULT_SURVIVAL_STRATEGY`。
- **稳定性修复**：surrogate exploration 与旧 `sample_mutation` API 使用显式 RNG；现有
  plugin wrapper 调用形状保持兼容。lexicase unsupported fallback 继续记录 effective policy
  和 reason。
- **验证**：完整 MySRCore `Pkg.test("MySRCore"; coverage=false)` 通过；parent selection
  `50/50`、surrogate gate `11/11`；Python 策略/migration/uncertainty focused tests `10 passed`，
  backend default bridge smoke 成功，额外 Python default backend tests `2 passed`，compileall 与
  `git diff --check` 通过。
- **限制**：未运行完整 Python 405 项或 matched P/M benchmark；当前结论证明契约、兼容和稳定性，
  不代表新默认在搜索质量、吞吐或资源上已经优于旧策略。

## 2026-09-22 - Isolate RNN-GPSR lightweight GPSR budget

- **Decision**：`Options` now exposes independent `rnn_gpsr_populations`,
  `rnn_gpsr_population_size`, `rnn_gpsr_niterations`, and
  `rnn_gpsr_ncycles_per_iteration` fields with defaults `1`, `8`, `1`, and `4`.
  The formal population, population size, iterations, cycles, and migration settings
  remain separate from the RNN-GPSR bootstrap stage.
- **Compatibility**：`rnn_gpsr_cycles` remains a normalized compatibility alias;
  conflicting explicit values are rejected. The seed builder invokes the existing
  regularized-evolution cycle for each configured lightweight population/iteration,
  with no population migration during seeding.
- **Quality fixes**：Each lightweight population receives an independent deterministic
  RNG stream and forked plugin state. Seed pools are bounded and tree-deduplicated;
  bootstrap candidate evaluations are counted; and tournament sampling is clamped so
  the default formal tournament size remains valid for an eight-member lightweight
  population.
- **Verification**：Before canonical synchronization, the complete MySRCore test suite
  passed under Julia 1.10.3 with an isolated writable depot/project, including the new
  lightweight-budget regression. Final post-sync tests and bridge checks are recorded
  after re-running the affected suites.
- **Unknown**：Matched P/M benchmark quality, throughput, and evaluation-count effects
  remain to be measured; this change does not claim a general performance gain.

## 2026-09-22 - RNN-GPSR post-sync verification

- **Verification**：After merging canonical default-policy changes, Julia 1.10.3 with an isolated writable depot/project ran the complete `Pkg.test("MySRCore"; coverage=false)` suite successfully. The RNN-GPSR testsets included the independent-budget regression (`6/6`), and all existing uncertainty, parent-selection, surrogate, migration, dimensional, and template testsets passed. `git diff --check` also passed.

## 2026-09-23 - Child constant and structure refinement

- 变更类型：搜索质量与 child acceptance 前的局部优化。
- 隔离路径：`worktrees/constant-structure-optimization-20260922/MySRCore.jl`；分支
  `feature/constant-structure-optimization-20260922`。worktree 已吸收 canonical Core
  `9b31167` 的后续 RNN-GPSR lightweight budget 提交，未修改 canonical checkout。
- 实现：新增 `child_refinement` 选项（`:safe` 默认、`:thorough`、`:none`）；mutation 与
  crossover 在 raw evaluation 后执行有界常数精修，并对精修后的 cost/loss 更新 surrogate
  观察；加入 constant folding、operator combination、双重 negation 与 neutral-element
  的结构候选，只有完整重评估不劣且复杂度更低时才采用；半理论公式的外部量纲系数保留父代值；
  population end-of-iteration simplification 共用同一结构清理路径；新增 optimizer option
  override 与 bounded budget helper；crossover 对 NaN child 提前拒绝。
- 测试：env_1_mysr Julia 1.10.3，隔离可写 depot/project 指向本 worktree；完整
  `Pkg.test("MySRCore"; coverage=false)` 通过，包含 RNN-GPSR lightweight、量纲、模板和
  新增 `Child constant and structure refinement` `11/11`；`git diff --check` 通过；
  Float32 unary serial smoke 通过。
- 诊断：固定小型 hard sinusoid 搜索中，`:safe` 与历史 `:none` 在 3 个 seed 上均能完成；
  线性目标的 3-seed local comparison 中 `:safe` 找到近零损失解，`:none` 有两个 seed
  停在较高损失。该实验不是 matched benchmark，不据此宣称普遍性能收益。
- 提交：算法提交 `287fd62`；吸收 canonical 最新状态的 merge commit `295c6af`；未合并回
  canonical、未删除 worktree、未推送。
- 遗留：尚未运行完整 Python 405 项或正式 matched P/M benchmark；child refinement 的
  额外 evaluations 与默认策略长期收益需后续基准测试确认。

## 2026-09-23 - Child refinement determinism and C_dim quality repair

- 变更类型：合并前回归修复与代码质量改进。
- 修复：safe child refinement 取消随机重启，保留 child 创建时的 birth，避免常数精修被
  age-fitness Pareto 当作额外进化世代并破坏固定 seed 的重复性；semi-theoretical
  `C_dim` 内层重拟合正确传入 `options`，不再因参数缺失被 fallback catch 静默跳过。
- 回归：固定 seed 的 RNN-GPSR 前沿连续重复两次保持一致；新增 `C_dim` 外部系数保持和
  内层常数改善断言。
- 验证：env_1_mysr Julia 1.10.3、隔离可写 depot/project；完整
  `Pkg.test("MySRCore"; coverage=false)` 通过，新增 refinement testset `11/11`；
  Float32 unary serial smoke 通过；`git diff --check` 通过。
- 遗留：未运行完整 Python 405 项或 matched P/M benchmark；默认 child refinement 的
  长期性能收益仍需正式基准测试确认。

## 2026-09-23 - Child refinement canonical merge closure

- 收口：在重新读取 canonical `main@9b31167` 并保留其本地 RNN-GPSR 提交后，将
  `feature/constant-structure-optimization-20260922` 快进合并至 `main`，当前代码提交为
  `5c6c495`；保留备份引用 `backup/constant-structure-optimization-20260923-pre-merge`。
- 验证：canonical checkout 的 `pathof(MySRCore)` 指向当前 `/data5/taomingyu_5/MySR/MySRCore.jl`；
  完整 `Pkg.test("MySRCore"; coverage=false)` 通过，`git diff --check` 通过。
- 范围：只处理指定的 child-refinement worktree；其他 worktree 和外层 MySR 未相关本地改动未检查、未修改。
- 待执行：本记录提交后推送 canonical `main`，再删除准确的优化 worktree 与其 feature 分支。
