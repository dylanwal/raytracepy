# RayTracePy 鸿蒙 PC 移植进度

状态统一使用：`未开始`、`进行中`、`阻塞`、`待审核`、`已完成`。

| 负责人 | 工作项 | 分支 | 截止节点 | 状态 | PR / 结果 |
|---|---|---|---|---|---|
| 1号（组长） | 基线、仓库管理、PR 审核、最终集成 | `port/harmonyos-pc` | 第 10 天 | 进行中 | 已建立基线文档 |
| 2号 | 参考环境、固定输入输出、自动化测试 | `feature/baseline-tests` | 第 2 / 8 天 | 进行中 | 参考基线、fixtures、自动化测试已完成；参考 vs 目标数值对比已完成（直方图 sha256 逐位一致），见 `docs/harmonyos-pc/REFERENCE_COMPARISON.md` |
| 3号 | 鸿蒙原生 Python 环境检查 | `feature/python-environment` | 第 3 天 | 待审核 | 脚本与报告已提交 PR #3，环境已在设备上实测通过（检测脚本 `RESULT: OK`；`NUMBA_DISABLE_JIT=1` 时测试 7 passed）。所用 CPython 3.10.15 系 Alpine Linux 构建、**非为鸿蒙编译**（原生执行、无模拟层），验收标准待老师裁定，见 `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md` 与探测报告第十、十一节 |
| 4号 | 依赖兼容矩阵和安装脚本 | `feature/dependencies` / `feature/dependencies-fix` | 第 4 / 9 天 | 待审核 | 依赖清单、兼容矩阵与安装脚本已完成；安装脚本已在干净 venv 上端到端验证（`feature/dependencies-fix` 含修正，建议先合） |
| 5号 | 源码兼容性和官方示例 | `feature/harmonyos-delivery` | 第 7 天 | 待审核 | 官方示例 `examples/single/single_light.py` 完整运行；源码仅需 1 处环境适配（`NUMBA_DISABLE_JIT` 守卫，PR #4 已合并）；Windows 专用 `.pyd` 不影响运行 |
| 6号 | 打包、安装文档、报告和演示 | `feature/harmonyos-delivery` | 第 9 天 | 待审核 | wheel/sdist 已构建（`artifacts/release/`）；`docs/harmonyos-pc.md`、移植报告、已知问题、演示脚本齐全；全新环境复测 7 passed |

## 第一周检查点

- [x] 创建 `port/harmonyos-pc` 集成分支；
- [x] 确定源码、版本、许可证和项目范围；
- [x] 2号提交参考环境版本、示例输入输出和运行日志；
- [x] 3号确认鸿蒙 PC 系统、CPU 架构、原生 CPython 和 pip
      （探测已完成：系统 HarmonyOS 6.1.0、架构 aarch64 已确认；
      设备**不存在为鸿蒙编译的 CPython**，现用解释器为 Alpine Linux 构建、
      在设备上原生执行；环境已实测通过，见 `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md`
      与探测报告第十、十一节）；
- [ ] 4号提交主要依赖的安装与导入结果；
- [ ] 5号确认 Windows 专用 `.pyd` 是否影响正常运行；
- [ ] 第 5 天完成“继续移植 / 申请调整范围”的阶段结论。

## 第二周检查点

- [x] 至少一个官方示例在鸿蒙 PC 上完整运行；
- [x] 参考环境与鸿蒙 PC 的数值结果完成对比；
- [ ] 所有功能 PR 已审核并合并到 `port/harmonyos-pc`；
- [x] 在全新环境完成安装、导入和示例复测；
- [x] 安装包、测试报告、移植报告、已知问题和演示材料齐全；
- [ ] 记录最终分支、commit 和仓库链接。

## PR 规则

1. 每位成员只在按“姓名 + 功能”命名的个人分支工作；已经创建的分支保持原名，不必重新命名；
2. PR 目标分支统一为 `port/harmonyos-pc`；
3. PR 必须写明修改内容、测试方法、运行结果和遗留问题；
4. 通过相关负责人交叉验证和组长审核后再合并；
5. `master` 保留上游基线，项目验收完成前不合并鸿蒙专属改动。

## 每日汇报格式

```text
负责人：
今日完成：
验证证据（commit / PR / 日志）：
当前阻塞：
明日计划：
```
