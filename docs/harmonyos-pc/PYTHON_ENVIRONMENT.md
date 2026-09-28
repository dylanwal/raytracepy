# 鸿蒙 PC Python 环境说明

- 整理：1号（在目标设备上实测）
- 日期：2026-09-26
- 设备：HUAWEI MateBook Pro（HAD-W32），HarmonyOS 6.1.0，API 23，aarch64，32 GB
- 相关文档：`PYTHON_ENVIRONMENT_PROBE.md`（探测过程与解释器来源）、
  `../../artifacts/environment/harmonyos-pc/`（原始证据）

## 一、现状（实测）

项目根目录下的 `.venv` 即目标环境，`raytracepy` 以 editable 方式安装（`-e .`，源码树 `src/`）：

| 组件 | 版本 |
|---|---|
| Python | 3.10.15（CPython，aarch64，musl，`~/.local/alpine-py310`） |
| 目标三元组 | `aarch64-alpine-linux-musl` |
| pip / setuptools / wheel | 26.2.1 / 65.5.0 / 0.48.0 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.2.3 |
| numba | 0.61.2 |
| llvmlite | 0.44.0（LLVM 15.0.7） |
| plotly | 5.6.0 |
| datashader | 0.16.3 |
| pytest / pytest-cov | 9.1.1 / 7.1.0 |
| 工具链 | `/data/service/hnp/bin` 下 `clang`/`clang++` 15.0.4、`make`、`cmake`、`ninja` |

## 二、验证（实测）

```bash
cd <repo>
source .venv/bin/activate
python scripts/check_environment.py        # RESULT: OK，无阻塞、无警告
NUMBA_DISABLE_JIT=1 pytest -q tests/       # 7 passed
```

- 环境检测输出：`artifacts/environment/harmonyos-pc/check_environment.log`、
  同目录 `environment.json`（键集覆盖参考基线文件，可直接对比）
- 测试证据：`artifacts/environment/harmonyos-pc/pytest-evidence.log`

注意：不加 `NUMBA_DISABLE_JIT=1` 时 `pytest` 会段错误（numba JIT 产物无法执行），
且该变量需要 `fix/numba-disable-jit-override` 的源码守卫才会生效，见第四节。

## 三、可重复执行的环境准备步骤

> 完整准备工具位于目标设备 `~/.local/ohos-python-tools/`（`setup_py310.sh`、
> `fixall.py`、`fixcompat.py`、`fixwheels.py`、`selfsign.py`、`add_stlshim.py`、
> `stlshim.cpp`、`pyapks/`、`wheels/`）。
> **该目录在仓库之外，不是持久交付物**，建议归档后再作为验收依据（见第四节第 5 条）。

依据 `setup_py310.sh`，环境由以下步骤构成：

1. 准备一份**带签名的 musl loader 副本**：OpenHarmony 拒绝直接执行未签名的 ELF，
   而 pip / setuptools / packaging 会调用 loader 探测 musl 版本。
2. 解压 Alpine 3.10.15 的 apk 集合到 `~/.local/alpine-py310`
   （python3、python3-dev、readline、sqlite、xz、zlib、openssl 等）。
3. **替换 libffi（3.4.4 → 3.8.0）**：Alpine 的 3.4.4 在本内核上
   `ffi_closure_alloc()` 返回 NULL，ctypes 回调（numba 依赖）因此不可用。
4. **平台补丁**：`sysconfig.get_platform()` → `linux-aarch64`，
   `platform.system()` → `Linux`（与 harmonybrew 构建保持一致，使 pip 产出 musllinux 标签）。
5. **`patchelf` 全量修补**：设置 rpath；把 `libc.musl-aarch64.so.1` 换成 `libc.so`
   （否则会加载第二份 libc）；给 `lib-dynload/*.so` 追加 `libpython3.10.so.1.0` 依赖；
   把 `bin/python3.10` 的解释器指向签名后的 loader。
6. **对所有 ELF 逐个签名**（OpenHarmony 的 fs-verity 代码签名要求）。
7. **安装预编译 wheel**：`llvmlite-0.44.0-cp310-cp310-linux_aarch64.whl`、
   `numba-0.61.2-cp310-cp310-linux_aarch64.whl`；`stlshim` 兼容 shim 仅挂在
   numba 的 4 个扩展与 pillow 的 libzstd 上。
8. 创建 `.venv`，安装 numpy / scipy / pandas / plotly / datashader / pytest / pytest-cov
   与 `-e .`。

选 3.10 的原因：参考栈（numba 0.56.4 系列）只覆盖到 Python 3.10；
harmonybrew 只提供 `python@3.12/3.13/3.14`，因此改用 Alpine 的 CPython 3.10 二进制。

## 四、已知限制

1. **解释器不是为鸿蒙编译的**：Alpine 官方 3.10 二进制 + ELF 元数据修补，
   在设备上原生执行（aarch64 指令、链接系统 musl，无虚拟机、无容器、无模拟层）。
   验收是否接受该口径待老师裁定，见 `PYTHON_ENVIRONMENT_PROBE.md` 第十节。
2. **numba JIT 产物无法执行**：JIT 能编译，执行编译产物时段错误
   （`src/raytracepy/raytrace.py:175`）。纯 Python 路径结果与参考**逐位一致**，
   耗时约为参考的 2.87 倍（见探测报告第十节第 3 小节）。
3. **需显式关闭 JIT**：`NUMBA_DISABLE_JIT=1` 原本被
   `src/raytracepy/__init__.py` 的 `numba.config.DISABLE_JIT = False` 覆盖；
   修复分支 `fix/numba-disable-jit-override` 只在该变量显式设置时才尊重它，
   默认行为不变。
4. **Windows 专用产物不适用**：`src/raytracepy/compile/math_custom.cp310-win_amd64.pyd`
   在非 Windows/AMD64 上不可用（待 5号 确认是否影响运行）。
5. **准备工具在仓库外**：`~/.local/ohos-python-tools/` 目前不随仓库交付，
   换设备无法复现，建议归档（可放 `scripts/harmonyos/` 或作为发布附件）。

## 五、结论

目标设备上的 CPython 3.10.15 环境**可以安装、导入并运行 RayTracePy**：
检测脚本判定 `OK`，`NUMBA_DISABLE_JIT=1 pytest -q tests/` 为 `7 passed`，
数值结果与参考环境逐位一致。剩余两项——"是否为鸿蒙编译"与"是否必须修复 JIT 段错误"——
属验收口径问题，需老师裁定。
