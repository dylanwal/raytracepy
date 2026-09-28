# 参考环境与目标环境数值对比

- 负责人：2号（由 1号 在目标设备实测执行）
- 日期：2026-09-26
- 负载：`tests/fixtures/single_light_reference_input.json`
  （`examples/single/single_light.py` 的完整 300 万光线工作量，`seed=20260920`）
- 参考结果：`artifacts/reference/windows-python310/`
- 目标结果：`artifacts/environment/harmonyos-pc/reference-comparison/`

## 一、两个环境

| | 参考环境 | 目标环境 |
|---|---|---|
| 系统 | Windows 10（AMD64） | HarmonyOS 6.1.0（aarch64，HAD-W32） |
| Python | CPython 3.10.21 | CPython 3.10.15（musl） |
| numpy / numba | 1.22.0 / 0.56.4 | 1.26.4 / 0.61.2 |
| numba JIT | 开启 | **关闭**（设备上 JIT 产物无法执行，见移植报告已知问题 1） |
| 执行命令 | 见 `docs/testing/REFERENCE_BASELINE.md` | `NUMBA_DISABLE_JIT=1 python scripts/generate_reference_baseline.py --input tests/fixtures/single_light_reference_input.json --output-dir <dev> --verify-repeat` |

## 二、结果对比

| 指标 | 参考（Windows，JIT 开） | 目标（鸿蒙，纯 Python） | 判定 |
|---|---|---|---|
| total_rays | 3,000,000 | 3,000,000 | 一致 |
| hit_count | 1,923,054 | 1,923,054 | 一致 |
| miss_count | 1,076,946 | 1,076,946 | 一致 |
| hit_rate | 0.641018 | 0.641018 | 一致 |
| histogram_shape | [99, 99] | [99, 99] | 一致 |
| histogram_sum | 1,923,054 | 1,923,054 | 一致 |
| **histogram_sha256** | `bacd25e6…d0cc677` | `bacd25e6…d0cc677` | **逐位一致** |
| 直方图 min/1%/5%/10%/mean/std/90%/95%/99%/max | 15 / 26 / 36 / 46 / 196.21 / 200.15 / 493 / 676 / 902 / 1025 | 同左 | 一致 |
| hit_coordinate_min / max | 见 metrics | 同左 | 一致 |
| 运行耗时 | 22.33 s | 67.53 s | 3.02 倍 |
| repeat_identical（同环境重复运行） | True | True | 一致 |

## 三、浮点字段差异

逐字段比对后，**只有 4 个浮点字段存在差异**，全部为求和顺序导致的末位差异：

| 字段 | 参考 | 目标 | 绝对差 |
|---|---|---|---|
| hit_coordinate_mean[0] | 0.0007782921386433898 | 0.0007782921386433894 | 4.3e-19 |
| hit_coordinate_mean[1] | 0.004137619690995617 | 0.004137619690995636 | 1.9e-17 |
| hit_coordinate_mean[2] | 3.7e-21 | −6.6e-20 | 7.0e-20（两值均≈0） |
| hit_coordinate_std[2] | 1.2879190611469105e-16 | 1.287186117671944e-16 | 7.3e-20（两值均≈0） |

其余字段（含整数计数、直方图全部统计量与哈希）**完全相同**。

阈值对照：仓库自设的跨环境容差为 `rtol=1e-6` / `atol=1e-8`
（见 metrics 中的 `comparison_policy`），本次实测最大绝对差 1.9e-17，**远优于阈值**。

## 四、结论

1. 目标环境的计算结果与参考环境**一致**：所有整数计数与直方图哈希逐位相同，
   浮点统计量差异在 1e-17 量级，满足（并远优于）"基本一致"的验收要求。
2. 关闭 JIT **只影响速度，不影响结果**：耗时约为参考的 3.0 倍
   （67.53 s vs 22.33 s；该差异同时包含"关闭 JIT"与"硬件不同"两个因素，未做拆分）。
3. 同环境重复运行结果一致（`repeat_identical=True`），可用于后续回归。
