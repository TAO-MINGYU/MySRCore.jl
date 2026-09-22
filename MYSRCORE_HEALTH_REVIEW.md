# MySRCore 后端第一阶段体检与改良思考

日期：2026-09-22

基线：`MySRCore.jl/main@87b5d82`，版本线 `1.1.3`

上游参照：`SymbolicRegression.jl 2.0.0-beta.8`（见 `FORK_CHANGES.md`）

状态：第一阶段记录；本文不代表任何新代码已经实现或任何算法已经获得性能提升。

## 1. 目的、范围与证据规则

这份文档用于熟悉 MySRCore 后端，并为后续逐项改造提供一个共同事实入口。审查范围覆盖：

- parent selection：tournament、epsilon-lexicase、epsilon API 和默认策略；
- mutation/crossover：当前算子集合、概率层、mutation affinity、size-matched crossover，以及 semantic back-propagation 的设计空间；
- new-old competition：regularized evolution、age-fitness Pareto、competitive age-fitness；
- loss 与 uncertainty：内置 preset、非对称 uncertainty、case loss、cost 和数值稳定性；
- population specialization/migration：`IslandProfile`、profile group、组内随机 migration 和 HOF migration；
- 共同基础设施：Options 契约、随机数、量纲约束、表达式包装、HOF/Pareto、worker、测试和 benchmark 证据。

本文使用以下状态词：

- **Confirmed**：当前源码、测试或开发记录可以直接确认；
- **Decision**：当前已采用的接口或行为约定；
- **Unknown**：尚未有足够证据，不能当作结论；
- **Proposal**：后续可实施的设计或实验，不代表已经存在。

代码位置是导航和证据指针，不是稳定 API 承诺；实现改动后应重新核对行号和行为。

## 2. 后端结构导航

### 2.1 公共入口与 Options

- `src/MySRCore.jl`、`src/SymbolicRegression.jl`：包边界、公共导出、`equation_search` 和搜索状态；搜索主流程从 `src/SymbolicRegression.jl` 的 `_equation_search` 进入。
- `src/Core.jl`：模块装配和跨模块导出；它把 Options、Dataset、mutation/crossover、profile/migration 等子系统接到公共命名空间。
- `src/OptionsStruct.jl`：`Options` 字段、`AbstractOptions` 契约、`IslandProfile`、`PopulationProfileGroup` 和 profile view 的结构定义。
- `src/Options.jl`：Options 构造、默认值、参数合法性、插件合并、mutation/crossover 权重解析和 loss/uncertainty 校验。

`Options` 是一个大型、参数化且包含很多派生字段的不可变配置对象。公共构造器的 parent/survival 默认值在 `src/Options.jl` 的构造器参数中定义；`default_options()` 又提供一层搜索规模和默认 mutation 权重。后续改动必须同时检查这两层，不能只改其中一处。

### 2.2 搜索循环和数据流

一次搜索大致按以下顺序运行：

1. `Dataset` 保存输入、目标、权重、uncertainty 和量纲元数据；
2. `Population` 创建初始成员，必要时经过 dimensional generator、guesses 或 RNN-GPSR seeding；
3. `SingleIteration`、`RegularizedEvolution` 和相关 helper 选择 parent、执行 mutation/crossover、评价 child，并按 survival 策略更新 population；
4. `HallOfFame` 更新全局候选和 Pareto frontier；
5. 搜索循环在 `SymbolicRegression.jl` 中执行普通 population migration、HOF migration 和 seed/guess 注入；
6. worker/head/plugin 状态在 cycle/generation 边界同步。

这条数据流意味着一个新策略必须同时定义：配置入口、worker 可序列化形式、随机数来源、失败回退、评价预算、HOF 可见性和测试观测点。

### 2.3 Parent selection 与 survival

`src/ParentSelection.jl` 当前包含：

- `parent_selection_diagnostic`：判断 epsilon-lexicase 能否使用；
- `ParentSelectionContext`：每个 cycle 的 full dataset、RNG、case-loss cache 和 fallback reason；
- `_epsilon_lexicase_index`/`epsilon_lexicase_index`：按随机 case 顺序筛选候选；
- `age_fitness_pareto_survivor_indices`：以 cost 和 birth/age 进行 Pareto survival；
- `competitive_survivor_indices`：先执行 child-parent gate，再做结构去重和 AFP。

`Population.jl` 的 `best_of_sample` 会在支持时调用 epsilon-lexicase，否则退回 tournament。regularized evolution 的旧路径仍由搜索循环使用。三类 survival 名称和合法性检查位于 `Options`。

### 2.4 Mutation 与 crossover

- `src/Mutations.jl`：内置 mutation 类型及 `default_mutations()`；当前包括 constant/operator/feature、swap/rotate、add/insert/delete、simplify/randomize、optimize/backsolve、connection 和 do-nothing 等。
- `src/MutationWeights.jl`：旧 `MutationWeights` 兼容层、按名称解析和权重转 mutation list。
- `src/MutationFunctions.jl`：树级 mutation、合法候选生成、dimension-aware operator target、affinity sampling 和普通 subtree crossover helper。
- `src/Mutate.jl`：mutation event 的调度、plugin hook、接受/拒绝、失败回退和 member 更新。
- `src/Crossovers.jl`：`SubtreeCrossover` 与显式 opt-in 的 `SizeMatchedCrossover` 类型及默认列表。
- `src/Crossover.jl`：公共 crossover dispatch、外层表达式包装、量纲 scale 解包/恢复和 trace。
- `src/MutationAffinity.jl`：operator/feature affinity 矩阵及探索混合。

当前 mutation/crossover 权重是“候选类型到非负数”的列表，不是概率和为 1 的强类型策略图。抽样时由权重归一化逻辑解释；plugin 和 profile 还可以在 Options 构造后改变有效列表。

### 2.5 Loss 与 uncertainty

`src/LossFunctions.jl` 同时承担：表达式评价、普通 aggregate loss、逐 case loss、built-in loss preset、weights、uncertainty 读取和 `loss_to_cost`。`Options.loss_preset` 与 `Options.uncertainty_mode` 控制这些分支；当前 Options 文档列出 `l1`、`l2`、Huber、pseudo-Huber、log-cosh、Gaussian NLL、连续非对称 Gaussian NLL、非对称 robust preset 和非对称 Student-t NLL。

非对称模式从 `Dataset.extra.sigma_minus` 与 `sigma_plus` 读取 scale；对称模式读取 `sigma`。likelihood preset 要求 `loss_scale=:linear`，因为 NLL 可以为负，而 `loss_scale` 只是 score/cost 的尺度控制，不是 residual 的 log-space 变换。

### 2.6 Profile、普通 migration 与 HOF migration

`src/PopulationMigration.jl` 定义：

- `IslandProfile` 的有效 view：operator affinity、mutation weights、crossover weights、exploration floor；
- 内置 role bias：generalist、algebraic、rational、trigonometric、transcendental 等；
- `profiled_options`、`profile_for_population` 和 `migration_profile`；
- `population_profile_indices` 与 `random_migration_source`：同一 profile group 内随机选择 source。

`src/Migration.jl` 定义 candidate pool、profile compatibility、structural novelty 和 `migrate!`。普通 migration 在搜索循环中先随机选同组 source，再按 `migration_policy` 选择候选；HOF migration 从全局 dominating frontier 取候选，并按 destination profile 过滤。未配置 profile 时，当前实现保留全局兼容行为。

## 3. 当前功能状态与用户目标的差距

| 方向 | 当前 Confirmed 状态 | 与目标的差距或待决点 |
|---|---|---|
| Parent selection | 支持 tournament 与 epsilon-lexicase；不支持 batching、custom aggregate 或不可安全拆分的 custom loss 时回退 tournament，并提供 diagnostic。 | `Options` 当前默认仍为 `:tournament`；没有公共 epsilon 参数、系统默认 epsilon 常量或策略记录；回退可能使用户以为启用了 lexicase，实际运行的是 tournament。 |
| Survival | 支持 `:regularized_evolution`、`:age_fitness_pareto`、`:competitive_age_fitness`。 | 当前默认仍为 `:regularized_evolution`；AFP 和 competitive 的复杂度、重复形状、非有限 cost、birth 语义仍需匹配 benchmark。 |
| Mutation | 有多种内置 mutation、静态 operator/feature affinity、plugin 权重和 profile 权重。 | 没有 semantic back-propagation 内置算子；没有统一区分“mutation 类型概率”和“算子内部细化概率”的配置/观测层。 |
| Crossover | 默认 `SubtreeCrossover() => 1.0`；`SizeMatchedCrossover` 已实现为显式 opt-in，并保留 custom wrapper fallback。 | 新交换策略尚未成为默认；semantic crossover、语义候选筛选、额外评价预算和失败回退尚未定义。 |
| Loss | 内置普通、对称 uncertainty、非对称 uncertainty、robust 和 Student-t preset；case-loss 与 aggregate 共享主要路径。 | 需要独立核对每个 likelihood 的归一化、连续性、梯度/极端值行为、异方差 calibration 和与 parent/survival cost 的一致性。 |
| Profile specialization | `IslandProfile` 可覆盖 operator affinity、mutation/crossover multiplier、exploration floor；profile group 可按 share 展开。 | profile 只表达软搜索偏好，尚未覆盖 parent/survival/loss/预算等更高层策略；内置 role bias 对自定义 operator 的泛化有限。 |
| Ordinary migration | 同 profile 随机 source；支持 best-only 与 best-plus-novelty；profile compatibility 可过滤 operator。 | migration 的频率、方向性、替换数量和随机性尚未系统校准；兼容性矩阵只表达 operator 可达性，不等于完整 profile 语义。 |
| HOF migration | 全局 HOF/Pareto 候选按目标 profile 过滤后注入。 | 需要验证 HOF 过滤是否造成专化岛长期缺少跨域创新，以及全局 HOF 与 profile-local HOF 的职责边界。 |

## 4. 主要问题清单

### 4.1 API、默认值与可观测性

**Confirmed：默认策略与目标设计不一致。** `Options.parent_selection` 和 `Options.survival_strategy` 的构造器默认值，以及 `default_options()`，仍使用 tournament + regularized evolution（`src/Options.jl`）。这不是实现错误，但会让“系统默认使用 epsilon-lexicase + age-fitness Pareto”的产品目标无法从当前代码得到。

**Confirmed：epsilon 没有独立 API。** `_epsilon_lexicase_index` 内部用每个 case 的 MAD 作为 epsilon；用户不能传入 absolute epsilon、relative epsilon 或 epsilon mode，也没有从 Options 读取的系统默认值。`epsilon_lexicase_index(errors)` 只接受 errors 和 RNG。

**Confirmed：fallback 的有效策略可能不透明。** batching、custom aggregate loss、custom expression loss 和 custom elementwise loss 会让请求的 epsilon-lexicase 回退为 tournament。已有 diagnostic 是正确方向，但结果、trace 和 benchmark metrics 还没有统一记录 `requested`、`effective`、`fallback_reason`。

**Proposal：建立显式策略契约。** 为 parent selection 增加强类型或受控 Symbol 参数、epsilon mode/value、有效策略 diagnostic 和 per-cycle counters；Python 前端与 Julia Options 共用同一命名和默认值。默认切换前先完成 fallback 语义与 benchmark。

### 4.2 Parent selection 的算法与性能

**Confirmed：当前 epsilon-lexicase 每次筛选会创建 case values、finite values 和 kept 临时数组。** 对大 population、长数据集和高 cycle 数，这会使 parent selection 的分配量和 wall-clock 成为搜索瓶颈。

**Confirmed：case-loss cache 的生命周期是 cycle context。** 这是避免跨 mutation 使用旧 loss 的安全选择，但成员对象、plugin multiplier 和 profile view 的变化必须保持在同一 context 的不变量内。

**Unknown：MAD epsilon 是否适合所有数据分布。** 有离群点、重复目标、异方差、权重或极少样本时，MAD 可能过窄或过宽；当前没有 epsilon sensitivity、case order 或 candidate diversity 的系统指标。

**Proposal：分层 epsilon 设计。** 先保留 MAD 作为兼容模式，再增加 `:mad`、绝对值、按 case scale 的相对值和可选 quantile mode；对 zero-MAD case 定义明确 tie policy。实现时用复用 buffer、partial selection 或无分配扫描，保持与现有随机 case 顺序一致。

**Proposal：增加 parent-selection telemetry。** 每个 cycle 记录候选数、case 数、平均筛选轮数、最终剩余候选数、fallback 次数和 case-loss evaluation 数，供性能和行为 benchmark 使用。

### 4.3 Mutation 与 crossover 的概率层

**Confirmed：当前存在两套 mutation 默认来源。** `default_mutations()` 返回一套通用权重；`default_options()` 通过 `_mutation_weights(...)` 定义另一套搜索默认权重，随后还会合并 plugin mutation。两层权重的命名、用途和最终有效值容易混淆。

**Confirmed：权重不是可直接解释的概率。** mutation/crossover 抽样前会归一化或由 StatsBase 解释权重；profile 乘数和 adaptive mutation plugin 又会动态改变有效分布。用户无法从配置对象直接得到“每个 cycle 实际抽样概率”。

**Confirmed：semantic back-propagation 尚未实现。** 当前源码和内置 mutation 列表没有 semantic/back-propagation mutation；`EvaluateInverse`、`Backsolve` 等能力不能直接等同于该算子。

**Confirmed：`SizeMatchedCrossover` 已实现但默认不启用。** 当前默认 crossover 仍是 `SubtreeCrossover() => 1.0`；size tolerance 只匹配节点数，不匹配 weighted complexity、语义变化量或量纲难度。

**Unknown：新算子的额外评价预算是否可接受。** semantic mutation/crossover 通常需要逆函数求解、候选语义评价或多个 brood 子代，可能改变每个 cycle 的真实 evaluation count 和公平预算。

**Proposal：分开两级概率。** 一级决定 `mutation_kind`/`crossover_kind`；二级决定 operator target、节点位置、semantic step size、brood size、size tolerance 或 fallback。所有权重归一化后提供只读 effective probability report，并为每次 accepted/rejected/failed event 提供计数。

**Proposal：semantic back-propagation 的最小安全契约。** 算子应接受目标语义或局部 residual、声明最大额外评价数；使用显式 inverse registry；对不可逆、非有限、定义域失败、量纲不合法和超 maxsize 的候选回退到普通 mutation；不允许绕过 `check_constraints`、`validate_search_candidate`、complexity 和 loss contract。

**Proposal：crossover 采用渐进路线。** 先对 size-matched crossover 做复杂度/树高/语义变化统计，再增加 semantic brood/context-aware 候选；每个候选共享 eval context 和 cache，超过预算立即回退单候选 crossover。默认策略在 benchmark 证明收益前保持稳定。

### 4.4 Survival、new-old competition 与 HOF

**Confirmed：AFP 的主要实现是反复计算 active 集合中的 dominance count。** 在 parent+offspring pool 变大时，当前方法具有明显的高阶复杂度；competitive strategy 还包含结构 hash、代表选择和 duplicate fill。

**Confirmed：结构去重依赖 `UInt` hash。** hash 用于快速分组，但当前路径没有在 hash 相等时执行结构相等确认；极低概率的 hash collision 可能错误合并候选。更重要的是，“结构相同”只按 operator/feature/constant 形状判断，不等于数值表达式、参数化模板或语义等价。

**Unknown：birth 的方向与 AFP 压力是否符合预期。** 当前代码把较大的 `birth` 视为更年轻，并以更高 recency 作为 Pareto 目标；仍需检查 reset_birth、migration 注入、guess 注入和 deterministic 模式下的长期分布。

**Proposal：将 survival 拆成可测的 policy contract。** 明确 child-parent gate、duplicate policy、nonfinite policy、tie-break、complexity preference 和 migration birth reset；为每种策略输出保留/淘汰原因和 population age distribution。

**Proposal：增量化 AFP。** 在 population size 较大时维护近似 Pareto layers、cost/age 索引或候选窗口，避免每个 cycle 完整 O(n²) 重算；先以 reference implementation 做 differential test，再替换热路径。

### 4.5 Loss、非对称 uncertainty 与数值稳定性

**Confirmed：loss preset、uncertainty mode、weights 和 custom loss 有显式互斥/合法性校验。** 这是当前较完整的契约之一；同时 `eval_case_losses` 对不能安全拆分的 custom objective 返回 `nothing`，促使 parent selection 回退 scalar tournament。

**Confirmed：非对称模式按 residual 符号选择 sigma side，并包含 split-normal/robust/Student-t 分支。** 该路径会影响初始化、parent selection、constant optimization、survival cost、HOF 和 RNN-GPSR candidate quality。

**Unknown：所有 preset 的数学与统计性质尚未形成独立审计表。** 需要逐项确认归一化常数、residual=0 的左右连续性、梯度/次梯度、极小/极大 sigma、极大 residual、负 NLL、NaN/Inf 和数据类型提升行为。

**Unknown：uncertainty calibration 尚无 benchmark。** 当前实现使用给定 sigma 作为 likelihood scale，但还没有 coverage、standardized residual、NLL calibration、misspecified sigma 和 side-specific calibration 指标。

**Proposal：建立 loss reference suite。** 用手算值、有限差分/自动微分、极端值和多输出/切片数据验证 `eval_loss`、`eval_case_losses`、`eval_cost`、constant optimizer 和 HOF 排序的一致性；单独记录数学约定，避免后续把 `loss_scale` 与 log-space residual 混淆。

**Proposal：将 uncertainty 观测纳入搜索报告。** 保存 mode、preset、sigma 摘要、标准化 residual 分布和 calibration metrics；不要只报告最终 selected equation 的 raw loss。

### 4.6 Population specialization 与 migration

**Confirmed：profile 是 soft preference view，不改变全局 expression type、hard constraint、loss 或 population size。** 这可以防止 profile 绕过量纲/结构约束，但当前文档需要更明确说明哪些字段永远全局共享。

**Confirmed：内置 role bias 通过函数对象 identity 判断 operator family。** 对自定义 operator enum、包装函数、参数化 operator 或新 operator family，系统 profile 可能退化为 neutral，而用户需要手动 matrix。

**Confirmed：profile validation 检查矩阵形状、有限性、非负性和 mutation/crossover list 长度。** 仍需检查 zero row、zero destination column、空 mutation/crossover list、profile id 重复和 role/operator set 变更后的稳定行为。

**Confirmed：普通 migration 在同一 profile id 的 population group 内随机选 source；HOF migration 从全局 frontier 过滤到目标 profile。** 这实现了用户描述的两层 topology，但它依赖每个 destination 的有效 profile view 和 candidate compatibility。

**Confirmed：当前 migration 路径使用 `default_rng()`。** 搜索循环在 migration source 和 `migrate!` 调用处没有共享的显式 RNG 参数。**Unknown：deterministic/seed 的 migration 语义是否足够强。** 如果要求跨 worker、跨 cycle、跨 profile 可复现，应明确 seed stream 和 worker-local RNG 的来源，而不能只依赖全局 RNG 状态。

**Confirmed：`migrate!` 用 Poisson 抽样替换数量。** 给定 `fraction_replaced` 只决定期望替换数，不保证每轮实际替换比例；population 小或 fraction 小时大量 cycle 可能零替换。**Unknown：** 这是有意的随机语义，还是应提供 exact/binomial/Poisson 模式，尚未决定。

**Unknown：专化是否提高全局 HOF 质量。** role bias 可能增加结构多样性，也可能把岛限制在错误的 operator family；当前没有 profile-local diversity、cross-profile novelty、migration acceptance 和 HOF contribution 的长期证据。

**Proposal：将 profile 分成三层。**

1. `ProfileSpec`：只声明 operator/mutation/crossover 偏好；
2. `ProfileRuntime`：缓存有效权重、RNG stream、统计和失败原因；
3. `MigrationPolicy`：独立声明 source topology、candidate compatibility、novelty、fraction 和 replacement count。

这样可以避免把“专化方向”和“迁移机制”压在同一个 `IslandProfile` 结构中。

### 4.7 跨系统、约束与工程质量

**Confirmed：MySRCore 是独立 package identity，但源码大量沿用上游模块组织。** 上游同步、MySRCore public wrapper、MySR bridge 和前端参数规范化需要持续保持一致；一个 Julia-only 默认值变化也可能改变 Python 端的实际策略。

**Confirmed：量纲约束贯穿初始树、mutation、crossover、constant optimization、seeding、HOF 和 prediction。** 语义算子不能只在普通 `Expression` 上工作，必须覆盖 template、shared graph、`C_dim` wrapper 和自定义 expression fallback。

**Confirmed：现有后端测试集中在单个 `test/runtests.jl`。** 已有大量 focused testset，但 parent/survival、loss、profile/migration、template、RNN 和 dimension 的组合矩阵仍需要明确的分层入口。

**Unknown：完整组合路径的资源行为。** 例如 epsilon-lexicase + uncertainty + profile + migration + semantic candidate evaluation 可能同时增加 loss evaluation、内存和 worker 序列化成本；当前 focused smoke 不能代表长预算搜索。

**Proposal：先建立 contract test，再做算法 benchmark。** contract test 检查类型、默认值、fallback、seed、量纲和 HOF 不变量；benchmark 才比较 recovery、frontier、test error、evaluations、wall-clock、内存、失败率和 profile contribution。

## 5. 建议的算法改良路线

### 阶段 A：统一契约和可观测性

- 定义 parent selection、epsilon、survival、mutation/crossover probability 和 profile/migration 的公共命名；
- 明确 requested/effective/fallback 状态，并在 trace/report 中保存；
- 统一 RNG 注入，保证 serial、multithreading、multiprocessing 和 deterministic 模式的可解释性；
- 暴露最终 effective mutation/crossover weights 和每类 event counters；
- 给 loss preset 建立数学 reference table 和数值边界测试。

### 阶段 B：默认策略的可控切换

- 先保留旧默认作为 compatibility mode，增加显式 `default_policy` 或版本化策略集；
- 在 matched benchmark 中使用相同数据、seed、预算和资源，比较 tournament/epsilon-lexicase、regularized/AFP/competitive 的四臂或六臂组合；
- 只有当 fallback、超时、evaluation count 和 HOF/frontier 证据完整时，才决定是否把 epsilon-lexicase + AFP 设为系统默认。

### 阶段 C：mutation/crossover 的细化

- 先实现纯 contract 的 semantic back-propagation：逆函数注册、局部语义目标、约束检查、失败回退和额外 budget；
- 再比较普通 mutation、semantic mutation、size-matched crossover、semantic brood 和混合策略；
- 对每个新策略记录成功率、平均额外 evaluations、树大小变化、语义距离、loss improvement、constraint failure 和 wall-clock。

### 阶段 D：profile 与 migration 的适应化

- 先验证静态 role profile 是否增加跨岛结构多样性和 HOF contribution；
- 再考虑根据 case loss、operator usage、novelty 或 local stagnation 自适应调整 profile 强度；
- 让 migration frequency、candidate policy 和 replacement count 可独立消融；
- 研究 profile-local HOF 与 global HOF 的双层维护，避免全局 HOF 过滤造成长期偏置。

### 阶段 E：性能和长期稳定性

- 优化 epsilon-lexicase 的 buffer/cache 和 AFP 的重复计算；
- 为 structural signature 增加无碰撞确认或结构 equality fallback；
- 以 allocation、evaluation throughput、worker communication、GC 和 peak memory 为独立指标；
- 做长预算、跨 seed、跨 operator family、跨 uncertainty 和跨 population size 的稳定性实验。

## 6. 后续 benchmark 与验收矩阵

任何算法结论都必须同时保存完整 HOF/Pareto、逐复杂度 metrics、evaluations、耗时、资源和失败信息。建议最小矩阵如下：

| 轴 | 最小对照 |
|---|---|
| Parent | tournament / epsilon-lexicase；MAD / explicit epsilon（实现后） |
| Survival | regularized evolution / AFP / competitive AFP |
| Mutation | baseline / affinity / semantic back-propagation |
| Crossover | subtree / size-matched / semantic candidate（实现后） |
| Population | generalist / fixed profiles / profile groups |
| Migration | off / ordinary same-profile / HOF / both |
| Loss | ordinary / symmetric uncertainty / asymmetric uncertainty |
| Execution | serial / multithreading / multiprocessing；固定 threads、seed 和预算 |

必须分别报告：

- 搜索质量：validation/test error、exact/semantic recovery、完整 frontier recovery；
- 多样性：结构 unique count、operator usage、profile-local novelty、HOF contribution；
- 代价：actual evaluations、wall-clock、allocation、peak memory、worker communication；
- 稳定性：seed variance、timeout、nonfinite、constraint failure、fallback 和 crash 分类；
- loss：NLL、标准化 residual、coverage/calibration（uncertainty 模式）。

pipeline smoke、单次 serial fit、单个 selected equation 或 focused unit test 都不能单独证明普遍性能提升。

## 7. 下一阶段实施前需要锁定的决策

以下事项在真正改代码前必须形成明确 Decision：

1. epsilon API 是单一 `Float64`、按 case/数据 scale 的相对值，还是支持多种 mode 的配置对象；
2. 默认 epsilon 的数值、zero-MAD 行为、weighted case 行为和非有限 loss 行为；
3. parent/survival 默认切换是否通过版本化 policy，还是直接改变 `Options` 默认；
4. semantic back-propagation 的目标语义、逆函数集合、额外 evaluation budget 和失败回退；
5. mutation/crossover 的一级/二级权重是否都暴露给 Python 前端；
6. semantic crossover 是否以单候选、brood、context-aware 或组合策略作为第一版；
7. profile 是否允许影响 parent/survival/loss/预算，还是永久限定为 soft operator preference；
8. migration 的替换数量采用 Poisson、binomial、exact fraction 还是可配置模式；
9. deterministic 模式是否要求不同并行后端得到逐事件一致，还是只保证 seed 和统计可复现；
10. 新策略进入默认路径所需的 benchmark 门槛和失败/超时容忍度。

## 8. 第一阶段结论

**Confirmed**：MySRCore 已经具备 parent selection、三种 survival、loss/uncertainty、mutation affinity、size-matched crossover、population profile、普通 migration 和 HOF migration 的可运行基础；现有测试覆盖了不少局部契约。

**Confirmed**：当前系统默认仍是 tournament + regularized evolution；epsilon-lexicase 没有公共 epsilon 配置；semantic back-propagation mutation 尚未存在；size-matched crossover 不是默认策略。

**Unknown**：任何新策略是否提升 HOF、Pareto、泛化、速度或资源效率；profile 专化是否优于 generalist；非对称 loss 是否改善校准和恢复；这些都需要匹配 benchmark。

**Decision for this phase**：先把契约、默认值、fallback、随机性、概率可观测性、loss 数学和组合验收定义清楚，再分阶段实现算子和默认策略切换。本文不把 Proposal 写成已实现功能，也不以局部测试替代算法证据。

## 9. 主要证据入口

- Options、默认值与校验：`src/Options.jl`、`src/OptionsStruct.jl`；
- parent/survival：`src/ParentSelection.jl`、`src/Population.jl`、`src/RegularizedEvolution.jl`；
- mutation/crossover：`src/Mutations.jl`、`src/MutationWeights.jl`、`src/MutationFunctions.jl`、`src/Mutate.jl`、`src/Crossovers.jl`、`src/Crossover.jl`；
- loss/uncertainty：`src/LossFunctions.jl`、`src/Dataset.jl`；
- profile/migration：`src/PopulationMigration.jl`、`src/Migration.jl`、`src/SymbolicRegression.jl`；
- 回归与组合测试：`test/runtests.jl`；
- 上游和发布边界：`FORK_CHANGES.md`、`CHANGELOG.md`；
- 工作区证据边界：根 `AGENTS.md`、`reference/DEVELOPMENT_SUMMARY.md`、`review.md`。

## 10. 第一阶段实施回写（2026-09-22）

以下内容已经在隔离分支 `feature/core-quality-v1-20260922` 实现，尚未改变 canonical
默认策略：

- `Options` 增加 `epsilon` 与 `epsilon_mode`。系统默认是 `epsilon=nothing,
  epsilon_mode=:mad`，表示按当前 case 的 MAD 自适应；`:absolute` 和 `:relative`
  要求显式数值。`:mad` 下的显式数值作为 MAD 下限，避免 zero-MAD 时把候选集合错误收窄。
- 新增默认权重为 `0.0` 的 `SemanticBackpropMutation`。第一版只对普通二叉树执行逆语义路径、
  有限性检查和稳健常数替换；共享图、非二叉节点、不可逆 operator 或非有限目标安全回退。
  它是可验证的 contract baseline，不代表最终的 sparse semantic fitter。
- mutation/crossover/parent selection/cycle/optimization/migration 的关键抽样改用显式 RNG；
  固定 seed 的 serial search 已实测重复得到同一表达式序列。并行路径只承诺独立 stream 和
  统计可复现，尚未承诺逐事件相同。
- Options 关键输入改用 `ArgumentError`；AFP 复用每轮 dominance count；survival/migration
  的结构 hash 命中后增加递归 equality 确认；profile affinity 要求每个 source operator
  至少有正目的权重。
- 新增 API、semantic mutation、epsilon mode 与异常边界测试；完整 MySRCore `Pkg.test`
  已通过，MySR population-migration bridge focused tests 为 `4 passed`。

本阶段的本机诊断数字是 64×128 epsilon-lexicase absolute threshold 中位数约
`3.09e-5 s`，小型 serial search（1 population、10 members、4 cycles）热身后中位数约
`0.00358 s`。这些数字只用于回归和量级监视，不能替代第 6 节的 matched benchmark。
