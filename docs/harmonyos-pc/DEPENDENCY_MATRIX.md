# 鸿蒙 PC 依赖兼容矩阵（4号）

- 日期：2026-09-26（2026-09-28 由 1号 补充实测）｜分支：`feature/dependencies`
- 环境：CPython 3.10.15（`~/.local/alpine-py310`）+ 项目 `.venv`
- 证据：组长设备实测；未改 raytracepy 算法代码（仅 1 处环境适配，见 PR #4）。换解释器须重测。

## 兼容矩阵

| 包 | 参考 | 设备 | import | 状态 |
|---|---|---|---|---|
| numpy | 1.22.0 | 1.26.4 | OK | 可用（版本偏离；必须 <2，源码用 `np.NaN`） |
| scipy | 1.10.0 | 1.15.3 | OK | 可用（版本偏离） |
| pandas | 1.4.1 | 2.2.3 | OK | 可用（版本偏离） |
| numba | 0.56.4 | 0.61.2 | OK | 可用，但**必须关 JIT**（见下） |
| llvmlite | 0.39.1 | 0.44.0 | OK | 可用（随 numba，链接 LLVM 15.0.7） |
| plotly | 5.6.0 | 5.6.0 | OK | 可选（可视化） |
| datashader | 0.13.0 | 0.16.3 | OK | 可选（可视化） |
| pytest / pytest-cov | 9.1.1 / 7.1.0 | 同左 | OK | 可用 |
| raytracepy | 0.0.1 | 源码树 / wheel | OK | 可用，运行需 `NUMBA_DISABLE_JIT=1` |

钉版本：`requirements-harmonyos.txt`（含全部传递依赖）。
安装：`python scripts/install_dependencies_harmonyos.py`（已按下方顺序实现）。

## 安装顺序与后处理（必须，2026-09-28 补充）

```bash
export PIP_INDEX_URL=https://pypi.org/simple
1) numpy              (PyPI musllinux wheel)
2) 设备专用 llvmlite / numba wheel（PyPI 无 musllinux 版，先装，否则会源码构建失败）
3) requirements-harmonyos.txt 其余依赖
4) fixall.py <site-packages>        ELF 元数据 + musl 兼容 + 代码签名
5) add_stlshim.py <site-packages>   numba 的 4 个扩展链接 stl shim
6) raytracepy（`pip install -e .` 或 wheel）
```

第 4、5 步不可省略，实测失败现象：

| 现象 | 原因 | 处理 |
|---|---|---|
| `ImportError: ... Permission denied` 加载 `.so` | 扩展缺 `.codesign` 段，OpenHarmony 拒绝 `dlopen` | `fixall.py` |
| `symbol not found: _ZNSt20bad_array_new_lengthC1Ev` | numba 的 clang 编译扩展缺 libstdc++ 符号 | `add_stlshim.py` |

修复后扩展的 RUNPATH 指向 `~/.local/alpine-llvm15/usr/lib`（libstdc++）与
`~/.local/ohos-python-tools/lib`（libstlshim / libmusl_compat），这两个目录是**运行时依赖**。

## 传递依赖必须固定

`datashader` 依赖 `dask`；不固定版本时 pip 会装最新 `dask`，其 `dask.dataframe`
需要 `pyarrow`，目标环境导入 datashader 会失败。可用版本：`dask==2023.3.0`。

## numba JIT

import / 编译 OK，**执行编译产物段错误**（多版本已复现；崩溃点
`src/raytracepy/raytrace.py:175`）。

**规避已由 PR #4 合并**：`src/raytracepy/__init__.py` 原先无条件
`numba.config.DISABLE_JIT = False`，会覆盖 numba 读取的 `NUMBA_DISABLE_JIT`；
现已改为仅在未显式设置时强制开启（默认行为不变）。因此运行测试/示例加
`NUMBA_DISABLE_JIT=1` 即可，不再需要源码级恒等装饰器。

关 JIT 后数值与 Windows 参考**逐位一致**（直方图 sha256 相同），
300 万光线耗时 67.53 s vs 参考 22.33 s，约 3.0 倍。是否必须修复 JIT 待老师裁定。

## 端到端（关 JIT）

- `NUMBA_DISABLE_JIT=1 pytest -q tests/` → 7 passed
- `examples/single/single_light.py` 300 万光线跑通（退出码 0，生成 HTML）
- 干净 venv 按上述步骤从零安装 → 导入 OK + 冒烟仿真 OK + 7 passed
  （见 `PORTING_REPORT.md` 第六节与 `artifacts/environment/harmonyos-pc/fresh-env-install.log`）

## Windows 专用产物

`src/raytracepy/compile/math_custom.cp310-win_amd64.pyd` 为 Windows/AMD64 二进制，
目标环境不可用；该目录未被任何代码引用，**不影响导入与运行**。

## 未决

| 问题 | 谁 |
|---|---|
| JIT 是否必须修复 | 老师 |
| Alpine-py310 是否算原生验收 | 老师 |
| 接受设备栈偏离参考版本 | 组长 |

未解决项详见 `artifacts/dependencies/harmonyos-pc/ISSUES.md`。
