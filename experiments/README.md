# 实验记录

本目录包含项目研究过程中的实验脚本，按阶段组织。

这些脚本是历史研究记录。phase3 的最终评估器现在复用 `src/` 中的校验和
walk-forward 实现；实验参数仍需在独立验证区间冻结后，才能用于新的结论。
如果只想使用最终成果，请参考根目录的 `examples/` 目录。

## 目录结构

- `phase1/` — 第一阶段: 基础方法探索 (exp1-exp7)
- `phase2/` — 第二阶段: 参数优化 (exp8-exp10)
- `phase3/` — 第三阶段: DualPath + 加权TopK (round1, round2, eval_final)

## 详细实验日志

- [实验日志 (第一二阶段)](../docs/experiment_log.md)
- [迭代日志 (第三阶段)](../docs/iteration_log.md)
