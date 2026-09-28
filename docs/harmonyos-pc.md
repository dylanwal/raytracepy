# 鸿蒙 PC 安装与使用说明

RayTracePy 0.0.1 在鸿蒙 PC（HarmonyOS 6.1.0 / aarch64，HUAWEI MateBook Pro HAD-W32）
上的安装、验证与使用方式。已实测通过：安装、导入、官方示例、自动化测试。

- 目标环境版本矩阵与可重复准备步骤：`docs/harmonyos-pc/PYTHON_ENVIRONMENT.md`
- 依赖清单与兼容矩阵：`requirements-harmonyos.txt`、`docs/harmonyos-pc/DEPENDENCY_MATRIX.md`
- 实测证据：`artifacts/environment/harmonyos-pc/`
- 参考环境：Windows AMD64 + CPython 3.10.21（见 `docs/testing/REFERENCE_BASELINE.md`）

## 一、环境要求

| 项目 | 要求 |
|---|---|
| 系统 | HarmonyOS 6.1.0（API 23），aarch64，HongMeng Kernel 1.12.0 |
| Python | CPython 3.10.x（目标环境用 3.10.15，aarch64/musl） |
| numpy | **必须 < 2.0**（源码使用了 numpy 2.0 已移除的 `np.NaN`），推荐 1.26.4 |
| 其它依赖 | 见 `requirements-harmonyos.txt` |
| 工具链 | clang 15.0.4 / make / cmake / ninja（`/data/service/hnp/bin`） |

## 二、安装

### 方式 A：源码安装（已实测，含后处理）

以下流程已在一台干净 venv 上端到端验证通过（见 `PORTING_REPORT.md` 第六节）：

```bash
cd <repo>
python -m venv .venv-fresh && source .venv-fresh/bin/activate
export PIP_INDEX_URL=https://pypi.org/simple     # 见"常见问题"
W=~/.local/ohos-python-tools/wheels
T=~/.local/ohos-python-tools
SP=.venv-fresh/lib/python3.10/site-packages

# 1) numpy
pip install numpy==1.26.4

# 2) 设备专用 wheel（PyPI 无 musllinux 版；必须先于其它依赖安装）
pip install "$W/llvmlite-0.44.0-cp310-cp310-linux_aarch64.whl" \
            "$W/numba-0.61.2-cp310-cp310-linux_aarch64.whl"

# 3) 其余依赖（含全部传递依赖，已固定版本）
pip install -r requirements-harmonyos.txt

# 4) 后处理：ELF 元数据 + musl 兼容 + 代码签名
$T/fixall.py "$SP"

# 5) 后处理：给 numba 的 4 个扩展链接 stl shim
$T/add_stlshim.py "$SP"

# 6) 本项目
pip install -e .
```

**第 4、5 步不可省略**：OpenHarmony 要求 ELF 带 `.codesign` 段才能 `dlopen`，
未签名的扩展会报 `Permission denied`；numba 的扩展还缺一个 libstdc++ 符号，
需要 shim，否则报 `symbol not found: _ZNSt20bad_array_new_lengthC1Ev`。

### 方式 B：wheel 安装

本次交付的安装包已放在 `artifacts/release/`（构建时默认写到 `dist/`）：

```bash
pip install artifacts/release/raytracepy-0.0.1-py3-none-any.whl
```

wheel 的 tag 为 `py3-none-any`（与上游 PyPI 发布一致）。
依赖仍需按方式 A 的第 1–5 步安装，之后才能导入。

## 三、验证安装

```bash
cd <repo>
source .venv/bin/activate
python scripts/check_environment.py     # 期望：RESULT: OK
NUMBA_DISABLE_JIT=1 pytest -q tests/    # 期望：7 passed
```

## 四、运行官方示例

```bash
cd <任意输出目录>
NUMBA_DISABLE_JIT=1 python <repo>/examples/single/single_light.py
```

示例会运行 300 万光线的仿真、打印统计信息并生成 `single_led.html`。
实测输出与耗时见 `artifacts/environment/harmonyos-pc/example-run/`。

## 五、常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 运行时报 `Segmentation fault` | 设备上 numba JIT 执行编译产物会崩溃 | 加 `NUMBA_DISABLE_JIT=1`（需 `fix/numba-disable-jit-override` 的源码守卫） |
| `AttributeError: np.NaN` | 装了 numpy 2.x | 降到 `numpy<2`（本环境用 1.26.4） |
| `pip` 装不上 numba / llvmlite | PyPI 无 musllinux wheel | 用设备专用 wheel，见 `PYTHON_ENVIRONMENT.md` 第三节 |
| `No matching distribution found for numpy` 等 | pip 全局索引指向 OpenHarmony 开发者镜像，其中没有这些包 | `export PIP_INDEX_URL=https://pypi.org/simple` 后重试 |
| `ImportError: ... Permission denied` 加载 `.so` | 扩展没有 `.codesign` 段，OpenHarmony 拒绝 `dlopen` | 运行 `fixall.py <site-packages>`（含签名） |
| `symbol not found: _ZNSt20bad_array_new_lengthC1Ev` | numba 的 clang 编译扩展缺 libstdc++ 符号 | 运行 `add_stlshim.py <site-packages>` |
| `ModuleNotFoundError: No module named 'pyarrow'` | 未固定 dask 版本，新版 `dask.dataframe` 需要 pyarrow | 按 `requirements-harmonyos.txt` 固定 `dask==2023.3.0` |
| 末尾出现 `start: not found` | 示例用系统命令自动打开 HTML，设备无桌面环境 | 可忽略；HTML 已生成，可手动打开 |
| 运行明显比参考环境慢 | 纯 Python 路径（JIT 不可用） | 正常，约 2.87 倍；数值结果与参考逐位一致 |

## 六、相关文档

| 文档 | 内容 |
|---|---|
| `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md` | 目标环境说明、版本矩阵、可重复准备步骤 |
| `docs/harmonyos-pc/DEPENDENCY_MATRIX.md` | 依赖兼容矩阵 |
| `docs/harmonyos-pc/PYTHON_ENVIRONMENT_PROBE.md` | 环境探测报告与实测结果 |
| `docs/harmonyos-pc/REFERENCE_COMPARISON.md` | 参考环境与目标环境的数值对比 |
| `PORTING_REPORT.md` | 移植报告与已知问题 |
| `docs/testing/REFERENCE_BASELINE.md` | 参考环境基线与测试说明 |
