# MySRCore 后端仓库局部约束

## 目录职责

本目录是 MySRCore 的 Julia 后端源码仓库，主要内容是 `src/`、测试、文档、示例和 Julia 包元数据。

## Agent 使用规则

- 只在本仓库处理搜索核心、表达式、量纲检查、常数优化、HOF/Pareto 和 Julia 包接口。
- Python 前端源码应在相邻的 `../MySR/` 中修改；不要把前端实现复制进本仓库。
- Dev 级研究、Benchmark 报告、运行输出和长期记忆放在父目录对应位置。
- 修改公共后端接口或跨仓库契约前，先读父目录 `../memory/DEVELOPMENT_MEMORY.md` 和相关 `../memory/MySR_Design/` 主题文件。
- 每次必要的大幅修改后，更新本目录 `DEVELOPMENT_LOG.md`；若改变跨仓库契约，还要更新父目录开发日志和开发记忆。

## 验证与边界

- 以本仓库实际 Julia 源码和测试为执行事实；不要把历史上游代码路径当作当前路径。
- 不在本目录运行大规模 Benchmark 或远程计算，除非用户明确授权。
