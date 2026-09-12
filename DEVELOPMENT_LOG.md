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
