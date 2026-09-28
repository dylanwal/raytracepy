# 鸿蒙 PC Python 环境准备工具（归档说明）

本目录用于归档目标设备 Python 环境的准备工具，使环境**可复现**。
可重复的准备步骤见 `docs/harmonyos-pc/PYTHON_ENVIRONMENT.md` 第三节。

## 状态

- 工具目前位于仓库外的 `~/.local/ohos-python-tools/`。
- **脚本本体尚未归档进本目录**：需先确认各文件的作者与许可证（见下），
  由 1号 确认后再提交，避免把来源不明的代码或第三方工具混入仓库。
- 已归档：`toolchain-manifest.txt`（各文件的 sha256 前 16 位与大小）。

## 文件与来源

| 文件 | 用途 | 来源 |
|---|---|---|
| `setup_py310.sh` | 主流程：解压 Alpine 包 → 替换 libffi → 平台补丁 → patchelf → 签名 | 自研（1号） |
| `fixcompat.py` | 兼容修复（ELF 元数据、`PT_INTERP` 等） | 自研 |
| `fixwheels.py` | 让 wheel 安装的扩展模块在 OpenHarmony 上可导入 | 自研 |
| `fixall.py` | 批量执行上述修复 | 自研 |
| `signall.py` | 批量签名 | 自研 |
| `add_stlshim.py` / `stlshim.cpp` | 给 numba 的 4 个扩展与 pillow 的 libzstd 挂兼容 shim | 自研 |
| `selfsign.py` | ELF 自签名（OpenHarmony 要求代码签名） | **第三方**：[hqzing/ohos-selfsign](https://github.com/hqzing/ohos-selfsign)，0BSD |
| `pyapks/*.apk` | Alpine 3.10.15 的 18 个包（python3、python3-dev、libffi、openssl、readline、sqlite、xz、zlib 等） | Alpine Linux 官方仓库 |
| `wheels/llvmlite-0.44.0-cp310-cp310-linux_aarch64.whl` | llvmlite（PyPI 无 musllinux wheel） | 由工具链构建/修补 |
| `wheels/numba-0.61.2-cp310-cp310-linux_aarch64.whl` | numba（同上） | 由工具链构建/修补 |

另外需从 harmonybrew 取得 `libffi.so.8.5.0`，用于替换 Alpine 的 3.4.4
（后者在本内核上 `ffi_closure_alloc()` 返回 NULL）。

## 已验证的端到端流程

在干净 venv 上实测通过（结果见 `PORTING_REPORT.md` 第六节）：

```bash
export PIP_INDEX_URL=https://pypi.org/simple
python -m venv .venv-fresh
SP=.venv-fresh/lib/python3.10/site-packages

pip install numpy==1.26.4
pip install wheels/llvmlite-0.44.0-cp310-cp310-linux_aarch64.whl \
            wheels/numba-0.61.2-cp310-cp310-linux_aarch64.whl
pip install -r ../../requirements-harmonyos.txt
fixall.py "$SP"        # 1) ELF 元数据 + musl 兼容  2) 代码签名
add_stlshim.py "$SP"   # 3) numba 的 4 个扩展链接 stl shim
pip install ../../dist/raytracepy-0.0.1-py3-none-any.whl

NUMBA_DISABLE_JIT=1 pytest -q ../../tests/     # 期望 7 passed
```

## 运行时依赖（重要）

安装后的扩展带有指向**仓库外**目录的 RUNPATH：

| 目录 | 提供 | 缺失后果 |
|---|---|---|
| `~/.local/alpine-llvm15/usr/lib` | `libstdc++.so.6` | numba 导入失败 |
| `~/.local/ohos-python-tools/lib` | `libstlshim.so`、`libmusl_compat.so` | numba / pillow 导入失败 |

因此这两个目录（尤其 `lib/` 下的两个 `.so`）必须随环境保留或一并归档，
否则即使 venv 完好，numba 也无法导入。

## 待办

1. 把自研脚本归档到本目录（作者与许可证确认后）。
2. 确认第三方工具的许可证与分发方式：`selfsign.py` 为 0BSD，可再分发，
   但需保留版权与许可证声明；Alpine 包各自适用其许可证。
3. 决定二进制产物的归档方式：整个工具目录约 210 MB（含 apk 与 wheel），
   建议作为 release 附件或使用 Git LFS，不建议直接提交进普通提交历史。
   另外 `lib/libstlshim.so` 与 `lib/libmusl_compat.so` 是**运行时依赖**，
   必须归档（见上节）。
4. 归档后在本目录补一份端到端脚本（调用 `setup_py310.sh` + 建 venv + 装依赖），
   使其可在干净设备上一键复现。

## 校验

`toolchain-manifest.txt` 记录每个文件的 sha256 **前 16 位**与大小，用于快速判断
工具目录是否被改动：

```bash
cd ~/.local/ohos-python-tools
sha256sum setup_py310.sh fixcompat.py fixwheels.py   # 与清单前 16 位比对
```

归档脚本本体后，应在清单中补全完整 sha256，并用 `sha256sum -c` 校验。
