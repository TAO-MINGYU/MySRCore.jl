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
