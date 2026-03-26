# 文件：`baseline/__init__.py`

## 作用
- 作为 `baseline` 包的统一入口。
- 对外暴露两个常用对象：`BaselineConfig` 和 `run_baseline`。

## 如何使用
- 在其他脚本中直接导入：
  - `from baseline import BaselineConfig, run_baseline`
- 推荐配合 `scripts/run_baseline.py` 使用，无需手动拼装底层模块。
