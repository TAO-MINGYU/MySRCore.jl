# MySRCore crossover 优化计划

日期：2026-09-05
状态：Decision（本 worktree 的首个实现范围）/ Proposal（后续阶段）

## 本地代码同步状态（2026-09-12）

- **Confirmed**：canonical MySRCore `b92776a` 已合并到本 worktree，合并提交为 `08773e9`；原有 `SizeMatchedCrossover` 提交保留。
- **Confirmed**：配套 MySR worktree 位于 `/home/taomingyu/MySR_Dev/worktrees/crossover-optimization-python`，基于 canonical MySR `a7c787b`，版本线为 MySR/MySRCore 1.1.3。
- **Decision**：两个原始 checkout 仅用于提供本地最新代码和建立 backup branch；后续 crossover 开发继续只在这两个配套 worktree 中进行。

## 目标

降低随机 subtree crossover 产生极端大小交换的概率，并保留可与语义 crossover 公平比较的基线。当前实现只在本 worktree 中进行，不自动合并回原始 checkout。

## 方法整合

| 方法 | 可能收益 | 主要成本/风险 | 当前处理 |
|---|---|---|---|
| AGX / semantic backpropagation | 让子代靠近父代语义中点 | 逆函数、过程库、额外 evaluations、bloat | 后续 Proposal |
| semantic brood selection | 从多个候选中保留更好的子代 | 候选求值约为 k 倍，当前 API 还需缓存契约 | 后续 Proposal |
| context-aware crossover | 搜索更合适的接收上下文 | 需要枚举和评价大量插入位置 | 后续 Proposal |
| angle-aware mating | 让父代误差方向互补 | 需要 target 语义，改变 parent selection | 后续 Proposal |
| GP-GOMEA/linkage | 保护协同结构 building blocks | 需要固定模板或结构依赖模型 | 后续 Proposal |
| size-matched crossover | 按节点数匹配供体子树，减少大小突变 | 可能减少大步探索；节点数不是完整复杂度 | **本阶段实现** |

## 本阶段实现

新增显式 `SizeMatchedCrossover`（size-fair-inspired，而非声称满足严格无偏 size-fair 定义）：

1. 第一棵树沿用现有均匀节点采样（根节点也保留）；
2. 对第二棵树做一次 bottom-up 遍历，收集每个节点及其 subtree node count；
3. 优先在 `abs(size2-size1) <= size_tolerance * size1` 的节点中均匀抽样；没有候选时在全体最接近尺寸的节点中均匀抽样；
4. 交换同一对子树，复制节点避免 aliasing；
5. 对 TemplateExpression、GraphNode 等自带特殊包装或共享语义的表达式回退到既有 `crossover_trees`；
6. 默认 crossover 权重保持 `SubtreeCrossover() => 1.0`，新类型只能显式配置。

配置接口：`SizeMatchedCrossover(; size_tolerance=0.25)`。`size_tolerance` 是相对目标子树大小的容差，`0` 表示只接受相同节点数。
节点数只用于候选匹配；weighted complexity、嵌套深度和量纲合法性仍由外层 `compute_complexity` 与 `check_constraints` 判定。

## 验证

- 单元测试：构造器参数检查；精确/容差匹配和最近候选 fallback；父树不被修改且子树不共享可变节点；默认 crossovers 不变；新类型可通过 `Options(; crossovers=...)` 解析。
- 包测试：在 worktree 的 Conda/Julia 环境运行 `Pkg.test()`。
- 结构检查：确认默认 crossovers 不变、导出和 dispatch 完整、`git diff --check` 通过。
- 性能结论：本阶段只证明行为正确和接口可用；是否改善 HOF、树大小或运行时间必须用匹配 benchmark 单独验证。

## 代码质量提升阶段（本次执行）

目标分成三类，并只执行能由现有测试直接验证的低风险改动：

1. **能力上限**：让 `SizeMatchedCrossover` 的候选选择逻辑显式支持精确匹配、相对容差匹配和最近尺寸 fallback；对非法 tolerance 在公共 helper 层也立即报错，避免绕过构造器时产生隐式行为。
2. **使用效能**：将 donor 候选选择改为单次遍历中的 reservoir selection，减少 `distances`、`findall` 等中间数组；复用 member 解包、量纲系数恢复和 trace 的共用路径，减少每次 crossover 的重复代码。
3. **轻量性**：保持默认路径和外层约束接口不变，只抽取小型内部 helper；不引入语义缓存、额外 loss evaluation 或新的依赖，避免用代码压缩换取运行时风险。

验收证据：现有 crossover focused tests、完整 Julia 测试文件、`git diff --check`，以及原始 checkout 的 HEAD/status 对比。性能节省仍需后续 profiling/benchmark，不在本阶段宣称。

## 后续阶段（Proposal）

在 size-fair 行为测试和 profiling 通过后，再依次设计：

1. 语义互补 parent selection；
2. 固定预算 semantic brood selection，并扩展 prediction/loss 复用；
3. 小 procedure library 或 size-balanced context-aware crossover；
4. bounded、多候选、量纲感知 AGX。

任何后续阶段都必须保留完整 HOF、额外 evaluations、失败原因和复杂度记录，并不得直接提高默认新算子权重。
