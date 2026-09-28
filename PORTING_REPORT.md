# RayTracePy 鸿蒙 PC 移植报告

- 项目：将 PyPI `raytracepy 0.0.1` 移植到鸿蒙 PC（HarmonyOS 6.1.0 / aarch64）
- 目标设备：HUAWEI MateBook Pro（HAD-W32），Kirin X90，32 GB，API 23
- 参考环境：Windows 10 AMD64 + CPython 3.10.21
- 日期：2026-09-26
- 相关文档：`docs/harmonyos-pc.md`（安装使用）、
  `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md`（环境）、
  `docs/harmonyos-pc/DEPENDENCY_MATRIX.md`（依赖矩阵）、
  `docs/harmonyos-pc/REFERENCE_COMPARISON.md`（数值对比）

## 一、结论

RayTracePy 0.0.1 可在鸿蒙 PC 上**安装、导入并运行官方示例**；计算结果与参考环境
**逐位一致**（直方图 sha256 相同），满足"基本一致"的验收要求。

不需要为移植修改算法代码。唯一的源码改动是一处环境适配：
让 `NUMBA_DISABLE_JIT` 能真正生效（见第四节）。

## 二、完成标准对照

| # | 完成标准 | 状态 | 证据 |
|---|---|---|---|
| 1 | 在鸿蒙 PC 原生 Python 环境中成功安装 | ✅ | `check_environment.log`（`RESULT: OK`）；`pip install -e .` 与 wheel 安装均成功 |
| 2 | `import raytracepy` 正常执行 | ✅ | 检测脚本 `from_source_tree: present`；全新环境复测通过 |
| 3 | 至少一个官方示例完整运行 | ✅ | `examples/single/single_light.py` 退出码 0，生成 `single_led.html`，见 `artifacts/environment/harmonyos-pc/example-run/` |
| 4 | 计算结果与参考环境在允许误差内一致 | ✅ | 300 万光线参考负载：整数计数与直方图 sha256 逐位相同，浮点最大差 1.9e-17，见 `REFERENCE_COMPARISON.md` |
| 5 | 在全新环境中按文档可重新安装 | ✅ | 新建 `.venv-fresh`，按 `docs/harmonyos-pc.md` 步骤安装依赖 + wheel，导入与冒烟仿真通过，见第六节 |
| 6 | 测试、文档、安装包和已知问题齐全 | ✅ | `pytest -q tests/` 7 passed；`dist/` 内 wheel 与 sdist；`docs/` 与本节 |

## 三、环境

| 项目 | 值 |
|---|---|
| Python | CPython 3.10.15（aarch64 / musl），项目 `.venv` |
| 依赖 | numpy 1.26.4 / scipy 1.15.3 / pandas 2.2.3 / numba 0.61.2 / llvmlite 0.44.0(LLVM 15.0.7) / plotly 5.6.0 / datashader 0.16.3 |
| 测试 | pytest 9.1.1 / pytest-cov 7.1.0 |
| 工具链 | clang 15.0.4 / make / cmake / ninja（`/data/service/hnp/bin`） |
| 运行方式 | 关闭 numba JIT（`NUMBA_DISABLE_JIT=1`） |

详见 `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md` 与 `requirements-harmonyos.txt`。

## 四、源码改动

**唯一改动**：`src/raytracepy/__init__.py` —— 原先无条件执行
`numba.config.DISABLE_JIT = False`，会覆盖 numba 自己读取的 `NUMBA_DISABLE_JIT`
环境变量。改为仅在未显式设置该变量时才强制开启 JIT；**默认行为不变**。

除此之外**未修改任何算法或业务代码**，符合"若源码无需修改应如实记录"的要求。

## 五、验证证据

| 验证项 | 命令 | 结果 |
|---|---|---|
| 环境检测 | `python scripts/check_environment.py` | `RESULT: OK`，无阻塞、无警告 |
| 自动化测试 | `NUMBA_DISABLE_JIT=1 pytest -q tests/` | 7 passed |
| 官方示例 | `NUMBA_DISABLE_JIT=1 python examples/single/single_light.py` | 退出码 0，输出统计并生成 `single_led.html` |
| 参考负载对比 | `NUMBA_DISABLE_JIT=1 python scripts/generate_reference_baseline.py --input tests/fixtures/single_light_reference_input.json --verify-repeat` | 与参考逐位一致（sha256 相同） |
| 全新环境复测 | 见第六节 | 通过 |

证据文件均在 `artifacts/environment/harmonyos-pc/`：
`check_environment.log`、`environment.json`、`pytest-evidence.log`、
`example-run/`、`reference-comparison/`。

## 六、全新环境复测（已通过）

在一台已装好解释器的设备上新建空 venv，严格按 `docs/harmonyos-pc.md` 方式 A 执行：

```bash
export PIP_INDEX_URL=https://pypi.org/simple
python -m venv .venv-fresh
W=~/.local/ohos-python-tools/wheels ; T=~/.local/ohos-python-tools
SP=.venv-fresh/lib/python3.10/site-packages

pip install numpy==1.26.4
pip install "$W/llvmlite-0.44.0-cp310-cp310-linux_aarch64.whl" \
            "$W/numba-0.61.2-cp310-cp310-linux_aarch64.whl"
pip install -r requirements-harmonyos.txt
$T/fixall.py "$SP"          # ELF 元数据 + musl 兼容 + 代码签名
$T/add_stlshim.py "$SP"     # numba 的 4 个扩展链接 stl shim
pip install artifacts/release/raytracepy-0.0.1-py3-none-any.whl
```

结果（原始日志见 `artifacts/environment/harmonyos-pc/fresh-env-install.log`，
后处理日志见同目录 `fresh-env-fixall.log`、`fresh-env-stlshim.log`）：

| 步骤 | 结果 |
|---|---|
| 依赖安装（含固定传递依赖） | 成功 |
| `fixall.py` | 修补 236 个 ELF，失败 0（含签名） |
| `add_stlshim.py` | 为 numba 的 4 个扩展链接 shim |
| `import raytracepy` + 全部依赖 | 成功（numpy 1.26.4 / numba 0.61.2 / llvmlite 0.44.0 / scipy 1.15.3 / pandas 2.2.3） |
| 10000 光线冒烟仿真 | 成功（hits = 6432） |
| `pytest -q tests/`（在全新 venv 中） | **7 passed** |

复测中发现并已写入文档的三个必要步骤：固定传递依赖（否则 `dask` 新版会要求
`pyarrow`）、安装后必须签名（否则 `.so` 加载报 `Permission denied`）、
numba 扩展必须链接 stl shim（否则报 `symbol not found`）。

## 七、已知问题

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| 1 | **numba JIT 产物无法执行**：JIT 能编译，执行时段错误（`src/raytracepy/raytrace.py:175`） | 必须用纯 Python 路径，耗时约为参考的 3.0 倍 | 已提供源码守卫使 `NUMBA_DISABLE_JIT=1` 生效；数值不受影响。是否修复 JIT 待定 |
| 2 | 解释器**原生执行但非为鸿蒙编译**（Alpine musl 二进制 + ELF 修补 + 2 个兼容 shim） | 口径问题 | 建议按"原生执行"判定；严格口径需延长周期 |
| 3 | 环境准备工具在仓库外（`~/.local/ohos-python-tools/`），且其中 `selfsign.py` 是第三方工具（hqzing/ohos-selfsign，0BSD） | 换设备无法复现；不可直接 vendor | 需归档自研脚本并记录第三方工具来源与版本 |
| 4 | pip 全局索引指向 `https://pypi.cnb.cool/OpenHarmonyPCDeveloper/pypi/...`，其中没有 numpy 等包 | 按文档直接 `pip install` 会失败 | 安装时指定官方索引或本地 wheelhouse（如 `PIP_INDEX_URL=https://pypi.org/simple`） |
| 5 | numpy 必须 < 2.0（源码使用已移除的 `np.NaN`） | 误装 numpy 2.x 会报 `AttributeError` | 已固定在 `requirements-harmonyos.txt`（1.26.4） |
| 6 | `src/raytracepy/compile/math_custom.cp310-win_amd64.pyd` 为 Windows 专用 | 无 | 该目录未被任何代码引用，不影响导入与运行 |
| 7 | `scripts/check_environment.py` 的 `@njit` 探针只测简单函数，包自身 JIT 崩溃时仍报 OK | 检测结论可能偏乐观 | 已在探测报告第十一节注明；建议后续加强探针 |
| 8 | 已安装扩展的 `RUNPATH` 指向仓库外的 `~/.local/alpine-llvm15/usr/lib`（libstdc++）与 `~/.local/ohos-python-tools/lib`（libstlshim / libmusl_compat） | 这些目录缺失时 numba 无法导入 | 需与环境工具一并归档，见 `scripts/harmonyos/README.md` |
| 9 | 传递依赖必须固定 | 不固定 `dask` 时新版会引入 `pyarrow` 依赖导致 datashader 导入失败 | 已在 `requirements-harmonyos.txt` 固定全部传递依赖 |

## 八、交付物清单

| 交付物 | 位置 |
|---|---|
| 可安装包 | `artifacts/release/raytracepy-0.0.1-py3-none-any.whl`、`artifacts/release/raytracepy-0.0.1.tar.gz`（构建产物默认写在 `dist/`） |
| 安装与使用说明 | `docs/harmonyos-pc.md` |
| 移植报告与已知问题 | 本文件 |
| 环境说明与准备步骤 | `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md` |
| 依赖清单与兼容矩阵 | `requirements-harmonyos.txt`、`docs/harmonyos-pc/DEPENDENCY_MATRIX.md` |
| 数值对比报告 | `docs/harmonyos-pc/REFERENCE_COMPARISON.md` |
| 环境检测脚本 | `scripts/check_environment.py` |
| 自动化测试 | `tests/`（7 项）、`scripts/tests/`（75 项） |
| 实测证据与日志 | `artifacts/environment/harmonyos-pc/` |
| 示例产出 | `artifacts/environment/harmonyos-pc/example-run/single_led.html` |
