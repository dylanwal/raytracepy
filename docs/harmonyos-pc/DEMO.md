# 演示脚本（约 5 分钟）

面向验收演示，在鸿蒙 PC 目标设备上执行。所有命令均已实测。

## 准备

```bash
cd <repo>
source .venv/bin/activate
```

## 1. 环境（30 秒）

```bash
python -V                                  # Python 3.10.15
python -c "import raytracepy, numpy, numba; print(raytracepy.__file__, numpy.__version__, numba.__version__)"
```

说明：这是鸿蒙 PC 上的原生 aarch64 CPython 环境，依赖版本见
`requirements-harmonyos.txt`。

## 2. 环境自检（30 秒）

```bash
python scripts/check_environment.py
```

预期结尾：`RESULT : OK`，`blockers : none`，`warnings : none`。
（完整输出见 `artifacts/environment/harmonyos-pc/check_environment.log`）

## 3. 自动化测试（1 分钟）

```bash
NUMBA_DISABLE_JIT=1 pytest -q tests/
```

预期：`7 passed`。

> 必须带 `NUMBA_DISABLE_JIT=1`：设备上 numba 的 JIT 产物无法执行（见"已知问题"）。

## 4. 运行官方示例（2 分钟）

```bash
mkdir -p /tmp/demo && cd /tmp/demo
NUMBA_DISABLE_JIT=1 python <repo>/examples/single/single_light.py
```

预期：打印光线数与命中统计，最后生成 `single_led.html`。
本仓库保存的实测产物：`artifacts/environment/harmonyos-pc/example-run/single_led.html`
（含热力图与统计图，可直接在浏览器打开演示）。

## 5. 数值一致性（30 秒）

展示 `docs/harmonyos-pc/REFERENCE_COMPARISON.md` 的结论：

- 300 万光线参考负载，`histogram_sha256` 与 Windows 参考环境**逐位相同**：
  `bacd25e6…d0cc677`
- 命中数、命中率、直方图形状与全部统计量一致
- 浮点统计量最大差 1.9e-17（阈值 1e-6 / 1e-8）

## 6. 已知问题（30 秒）

| 问题 | 现状 |
|---|---|
| numba JIT 产物无法执行 | 用纯 Python 路径，结果一致，耗时约 3 倍 |
| 解释器原生执行但非为鸿蒙编译 | 见 `PYTHON_ENVIRONMENT_PROBE.md` 第十节 |
| 环境准备工具在仓库外 | 见 `scripts/harmonyos/README.md` |

## 演示材料清单

| 材料 | 位置 |
|---|---|
| 环境自检输出 | `artifacts/environment/harmonyos-pc/check_environment.log` |
| 测试证据 | `artifacts/environment/harmonyos-pc/pytest-evidence.log` |
| 示例运行日志 | `artifacts/environment/harmonyos-pc/example-run/example-run.log` |
| 示例产出（图形） | `artifacts/environment/harmonyos-pc/example-run/single_led.html` |
| 数值对比 | `docs/harmonyos-pc/REFERENCE_COMPARISON.md` |
| 参考环境对照产物 | `artifacts/reference/windows-python310/single_light_report.html` |
