# RayTracePy 参考环境与第一周测试基准

## 1. 基准结论

2号任务第一周基准建立在 Windows AMD64、CPython 3.10.21 和 GitHub 提交
`f09f19d898f0de8f3532df6ed03a66b142b1da32` 上。项目版本为
`raytracepy 0.0.1`，验收案例使用 `examples/single/single_light.py`。

完整案例固定随机种子为 `20260920`，运行 3,000,000 条光线。两次独立运行
的数值指标和直方图 SHA-256 完全一致，因此当前参考环境具有可重复性。

主要结果：

| 指标 | 结果 |
|---|---:|
| 总光线数 | 3,000,000 |
| 命中数 | 1,923,054 |
| 未命中数 | 1,076,946 |
| 命中率 | 0.641018 |
| 直方图尺寸 | 99 × 99 |
| 直方图最大值 | 1,025 |
| 直方图 SHA-256 | `bacd25e64f5ca3bd0d6df9bf7fa775d25d8410cea055368bebb3a0198d0cc677` |

完整数值、输入、图形和环境信息位于
`artifacts/reference/windows-python310/`。

## 2. PyPI 发布包验证

从 PyPI 安装 `raytracepy==0.0.1` 成功，但导入失败：

```text
ModuleNotFoundError: No module named 'raytracepy.utils'
```

发布 wheel 只包含 `raytracepy/*.py`，没有包含 `raytracepy/utils` 等子包。失败
日志保存在 `artifacts/reference/pypi-wheel-import.log`。因此本周数值基准使用
相同版本号、指定 Git commit 的 GitHub 源码；不得把源码环境的成功导入写成
“PyPI wheel 可以正常导入”。打包负责人和组长需要在后续集成时处理该问题。

## 3. 参考依赖选择

仓库 `requirements.txt` 固定 `numba~=0.53.1`，但该版本早于 Python 3.10
支持。参考环境采用 `requirements-reference.txt`，保留其余项目锁定版本，只将
Numba 调整为支持 Python 3.10 且兼容 NumPy 1.22 的 `0.56.4`。使用宽松 `setup.cfg` 直接解析会
安装 NumPy 2.x，而上游绘图代码使用已删除的 `np.NaN`，官方图形报告会失败。

关键版本：

| 依赖 | 版本 |
|---|---:|
| Python | 3.10.21 |
| RayTracePy | 0.0.1（GitHub source，editable） |
| NumPy | 1.22.0 |
| SciPy | 1.10.0 |
| Numba | 0.56.4 |
| llvmlite | 0.39.1 |
| Pandas | 1.4.1 |
| Plotly | 5.6.0 |
| Datashader | 0.13.0 |

完整传递依赖版本见
`artifacts/reference/windows-python310/installed-packages.txt`。

## 4. 固定输入和输出

自动化测试使用 10,000 条光线的缩小案例：

- 输入：`tests/fixtures/single_light_smoke_input.json`
- 期望输出：`tests/fixtures/single_light_smoke_expected.json`

完整官方案例使用原示例的 3,000,000 条光线和 100 × 100 分箱：

- 输入：`tests/fixtures/single_light_reference_input.json`
- 期望输出：`tests/fixtures/single_light_reference_expected.json`
- 压缩直方图：`artifacts/reference/windows-python310/single_light_histogram.npz`
- HTML 图形：`artifacts/reference/windows-python310/single_light_report.html`
- 运行日志：`artifacts/reference/windows-python310/single_light_run.log`

NumPy 和 Numba 维护不同的随机状态，基准脚本会同时设置两者。如果只调用
`numpy.random.seed`，不能保证 RayTracePy 的 JIT 随机采样可重复。

## 5. 复现方法

在仓库根目录执行：

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-reference.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\generate_reference_baseline.py `
  --input tests\fixtures\single_light_reference_input.json `
  --output-dir artifacts\reference\windows-python310 `
  --expected-output tests\fixtures\single_light_reference_expected.json `
  --plot `
  --verify-repeat
```

`--verify-repeat` 会运行案例两次，并要求所有记录指标完全一致。更新期望输出前
必须确认输入、源码 commit 和参考依赖没有意外变化。

## 6. 测试范围

第一周自动化测试覆盖：

- 包导入、版本号和公开 API；
- `Plane`、`Light` 和 `RayTrace` 的最小计算路径；
- 命中数组形状和有限值检查；
- 直方图总数与命中数一致性；
- 固定输入与固定期望输出回归；
- NumPy/Numba 双重固定种子后的重复运行一致性。

当前参考环境及全新重建环境的命令结果均为 `7 passed`。异常输入、鸿蒙实机结果和最终误差阈值
属于第二周工作。

## 7. 数值比较策略

参考环境内使用完全一致比较，以便尽早发现回归。鸿蒙端暂定使用
`rtol=1e-6`、`atol=1e-8` 比较连续数值，并单独比较总光线数、命中数和
直方图总和。该阈值只是第二周测试的起点，必须根据鸿蒙端重复运行证据重新
校准，不能为了通过测试而任意放宽。
