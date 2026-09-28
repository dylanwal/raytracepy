# 目标设备环境证据

本目录保存 3号 对鸿蒙 PC 目标设备的环境探测与验证证据。

## 文件

| 文件 | 说明 |
|---|---|
| `device_probe.log` | 2026-09-25 首次探测：在设备终端逐条执行命令的原始输出（当时设备尚无 Python 运行时） |
| `check_environment.log` | 2026-09-26 在设备 `.venv`（CPython 3.10.15）上运行 `scripts/check_environment.py` 的完整输出，`RESULT: OK` |
| `environment.json` | 同一次运行的机器可读结果；键集覆盖 `artifacts/reference/windows-python310/environment.json` 的全部键，可直接对比 |
| `pytest-evidence.log` | 同一环境下的测试执行证据（JIT 开启时段错误；`NUMBA_DISABLE_JIT=1` 时 `7 passed`） |

## 采集信息

| 项目 | 值 |
|---|---|
| 采集人 | 3号（2026-09-25 探测）、1号（2026-09-26 环境验证） |
| 采集设备 | HUAWEI MateBook Pro（HAD-W32），HarmonyOS 6.1.0，aarch64 |
| 探测方式 | 设备终端（第三方应用 Alacritty）手动执行 |
| 验证方式 | 项目 `.venv` 内运行仓库脚本与测试 |

设备序列号未记录在案。

## 结论

- 2026-09-25 探测当日：设备上没有可供本项目使用的 CPython 运行时，
  见 `device_probe.log`。
- 2026-09-26 现状：设备已具备可用的 CPython 3.10.15 环境，检测脚本判定 `RESULT: OK`。
  环境说明见 `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md`；探测当日的分析见
  `docs/harmonyos-pc/PYTHON_ENVIRONMENT_PROBE.md`（其第一、五节仅对当日成立）。

## `scripts/check_environment.py` 的验证情况

该脚本由 3号 编写，纯标准库实现。

| 环境 | 结果 |
|---|---|
| Windows AMD64 + CPython 3.13.13（无依赖） | `BLOCKED`，退出码 1，正确报告必需依赖缺失并识别 Windows 专用 `.pyd` |
| Windows AMD64 + CPython 3.10.21（参考依赖） | `OK_WITH_WARNINGS`，无阻塞项，可从 `src/` 导入 |
| 鸿蒙 PC + CPython 3.10.15（2026-09-26） | `RESULT: OK`，无阻塞、无警告，见 `check_environment.log` |

输出的 JSON 键集覆盖参考基线文件的全部键，可直接与参考环境对比。

## 已知限制

- 设备的 numba JIT 能编译但执行产物时段错误，`pytest -q tests/` 需要
  `NUMBA_DISABLE_JIT=1`，且该变量需 `fix/numba-disable-jit-override` 的源码守卫，
  见 `pytest-evidence.log` 与 `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md` 第四节。
- `device_probe.log` 第 [13] 节"脚本未能在目标设备上执行"保留为 2026-09-25 当日事实；
  2026-09-26 起脚本已可在目标设备运行。
