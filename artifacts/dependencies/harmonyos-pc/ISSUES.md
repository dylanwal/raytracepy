# 未解决依赖与错误日志（4号）

## numba JIT 不可用

- 现象：import / 编译 OK，执行编译产物 → 段错误（`si_addr=0x3c091000500`），
  崩溃点 `src/raytracepy/raytrace.py:175`
- 已排除：numba 0.60/0.61 × llvmlite 0.43/0.44 × LLVM 14/15，均复现
- 规避（2026-09-28 更新）：**不再需要源码级恒等装饰器**。PR #4 已把
  `src/raytracepy/__init__.py` 改为尊重 `NUMBA_DISABLE_JIT`（默认行为不变），
  运行测试/示例加 `NUMBA_DISABLE_JIT=1` 即可
- 影响：约 3.0 倍慢（67.53 s vs 22.33 s）；数值与 Windows 参考哈希逐位一致
- 是否必须修复 JIT：**待老师裁定**

## 安装类问题（2026-09-28 补充，均已解决并写入安装脚本）

| 现象 | 原因 | 处理 |
|---|---|---|
| `No matching distribution found for numpy` | 设备 pip 全局索引指向 OpenHarmony 开发者镜像，其中没有 numpy 等包 | `export PIP_INDEX_URL=https://pypi.org/simple` |
| `ImportError: ... Permission denied` | pip 装好的扩展没有 `.codesign` 段，OpenHarmony 拒绝 `dlopen` | 运行 `fixall.py <site-packages>` |
| `symbol not found: _ZNSt20bad_array_new_lengthC1Ev` | numba 的 clang 编译扩展缺 libstdc++ 符号 | 运行 `add_stlshim.py <site-packages>` |
| `ModuleNotFoundError: No module named 'pyarrow'` | 未固定 `dask`，新版 `dask.dataframe` 需要 pyarrow | 固定 `dask==2023.3.0`（已写入依赖清单） |
| `pip install -r` 无法安装 numba / llvmlite | PyPI 无 musllinux wheel | 先用设备专用 aarch64 wheel，再装其余依赖 |

## 解释器验收未决

- `~/.local/alpine-py310` 为 Alpine 构建，非为鸿蒙编译（在设备上原生执行，
  无虚拟机 / 容器 / 模拟层）
- 若要求「为鸿蒙编译」→ 改用 `python@3.12` 并重测全部依赖；
  注意 numba/llvmlite 仍无 musllinux wheel，重编工作省不掉

## 其他

- 设备栈版本偏离 `requirements-reference.txt`：已数值对齐（整数计数与直方图
  sha256 逐位一致），是否接受由组长确认
- 范围外：环境脚本 → 3号（PR #3 已合并）；源码回退 → 5号（已由 PR #4 解决）；
  安装文档 → 6号
