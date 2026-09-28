# 鸿蒙原生 Python 环境探测报告

- 负责人：3号
- 日期：2026-09-25（2026-09-26 更新第十、十一节）
- 状态：待裁定 —— 阻塞已解除，环境已实测通过（第十一节）；所用解释器非为鸿蒙编译，验收标准待老师认定

## 一、结论

> **本节为 2026-09-25 探测当日的结论。** 2026-09-26 设备侧环境已安装完成，
> 阻塞在事实上已解除，故本节"无可用运行时"的判定仅对当日成立，见第十节。
> 另需注意本节中"原生"一词的含义与第十节的限定不同，见第十节第 1 小节。

目标设备上**不存在可供本项目使用的原生 CPython 运行时**。

该结论包含两层：

1. 经系统化探测，未发现任何 Python 解释器、包管理器或构建工具链。剩余不确定性仅存在于应用沙箱不可访问的系统目录中。
2. 该设备上的终端运行于应用沙箱内，即使系统隐藏分区中存在 Python，沙箱也无法访问，对本项目不可用。此点已经证据确认，结论确定。

因此，**"无可用运行时"的判定成立**，不依赖未访问目录的探测结果。

## 二、设备信息

| 项目 | 值 |
|---|---|
| 型号名称 | HUAWEI MateBook Pro |
| 型号代码 | HAD-W32 |
| HarmonyOS 版本 | 6.1.0 |
| 软件版本 | 6.1.0.117 (SP68C00E100R13P3) |
| 处理器 | HUAWEI Kirin X90 |
| CPU 架构 | aarch64 |
| 运行内存 | 32 GB |
| 内核版本 | HongMeng Kernel 1.12.0 |
| API 版本 | 6.1.0(23) |
| OpenHarmony 版本 | OpenHarmony 6.1 |
| 系统服务版本 | 6.23.0.100 |
| 安全补丁标签 | 2026/05/01 |

（设备序列号已省略，不写入公开文档。）

要点说明：

- **内核为 HongMeng Kernel，不是 Linux 内核**，因此常见 POSIX 工具不保证存在。
- 芯片为 Kirin X90，架构 aarch64，与预期一致。

## 三、探测环境

| 项目 | 值 |
|---|---|
| 终端程序 | Alacritty（第三方应用，非系统终端） |
| Shell | zsh（依据：命令未找到时的报错前缀均为 `zsh:`） |
| 用户 | USER=100（非 root） |
| 家目录 | /storage/Users/currentUser |
| 当前目录 | /storage/Users/currentUser/Desktop |
| PATH | /usr/local/bin:/data/app/bin:/data/service/hnp/bin:/usr/bin:/vendor/bin |

## 四、探测结果

### 1. 系统架构

```
$ uname -m
aarch64
```

与 Kirin X90 一致。

### 2. PATH 内各目录存在性

```
$ ls /usr/local/bin
ls: /usr/local/bin: No such file or directory

$ ls /vendor/bin
ls: /vendor/bin: No such file or directory

$ ls /data/app/bin
ls: /data/app/bin: No such file or directory

$ ls /data/service/hnp/bin
ls: /data/service/hnp/bin: No such file or directory

$ ls /usr/bin
（存在，共 179 项，完整清单见附录）
```

PATH 中 5 个目录，仅 `/usr/bin` 真实存在。

> 本节及第四、五节均为 2026-09-25 探测当日的快照。此后状态已变化，见第十节。

### 3. /usr/bin 内容核查

完整清单中不含以下任何一项：

- `python`、`python2`、`python3`
- 任何 `python3.x` 形式
- `pip`、`pip3`

已核查的具体名称：awk, base64, basename, bunzip2, bzcmp, bzdiff, bzegrep, bzfgrep, bzgrep, bzip2, bzip2recover, bzmore, cal, cat, chmod, cksum, clear, cmp, comm, count, cp, cpio, crc32, curl, cut, date, dd, df, diff, dirname, dos2unix, du, echo, egrep, env, expand, expr, factor, false, fgrep, file, find, flock, fmt, free, ftpget, ftpput, getconf, getfattr, grep, gunzip, gzip, head, help, hexedit, hidumper, hilog, hiperf, hiprofiler_cmd, hisysevent, hitrace, hostname, iconv, ifconfig, iotop, ipcs, join, kill, killall, ln, loh, ls, lsattr, lsmod, mcookie, md5sum, mkdir, mkpasswd, mktemp, more, mount, mountpoint, mv, netcat, netstat, nl, nohup, nproc, od, openssl, param, paramshell, paste, patch, pgrep, pidof, ping, ping6, pkill, pr, printenv, printf, ps, pwd, pwdx, readahead, readlink, realpath, reboot, reset, rev, rm, rmdir, route, scp, sed, seq, setfattr, sftp, sh, sha1sum, sha256sum, sha384sum, sha512sum, shred, sleep, sort, split, ssh, ssh-keygen, stat, strings, sudo, sync, sysctl, tac, tail, tar, taskset, tee, telnet, test, time, timeout, timestamps, toe, top, touch, tr, traceroute, traceroute6, true, truncate, tsort, tty, uname, uniq, unix2dos, unlink, unzip, uptime, usleep, uudecode, uuencode, uuidgen, vi, view, vim, vmstat, watch, wc, wget, which, xargs, xxd, yes, zcat, zip, zsh

补充说明：

- 存在鸿蒙系统级工具：`hilog`、`hidumper`、`hitrace`、`hiperf`、`hisysevent`、`param`、`paramshell`。
- 存在网络与压缩工具：`curl`、`wget`、`ssh`、`scp`、`sftp`、`tar`、`zip`、`unzip`、`openssl`。
- **不存在任何 C 编译器或构建工具**：无 `cc`、`gcc`、`clang`、`cl`、`make`。

### 4. 包管理器

```
$ for c in apt apt-get dnf yum apk pacman rpm dpkg hpm pkgcmd brew; do command -v "$c"; done
（无输出）
```

11 种包管理器中，无一种存在。系统不提供任何软件安装手段。

### 5. 文件系统全盘搜索

```
$ find / -maxdepth 4 -name "*python*" 2>/dev/null
（无输出）

$ find /usr /storage -name "*python*" 2>/dev/null
（无输出）
```

在可访问的文件系统范围内，无任何名称含 `python` 的文件或目录。

### 6. /usr/lib 内容

```
$ ls /usr/lib
libcrypto.so.3  libncursesw.so.6  libssl.so.3  libtinfo.so.6  libz.so.1  libz.so.1.2.13
```

仅 6 个基础共享库，**无 libpython**。

### 7. 基础系统信息命令

```
$ id
zsh: command not found: id
```

连 `id` 这样的基础 coreutils 命令都不存在。

### 8. 运行环境判定（关键证据）

```
$ env
ALACRITTY_WINDOW_ID=window85_terminal0
HAP_DEBUGGABLE=false
XDG_CACHE_HOME=/data/storage/el2/base/haps/entry/files
XDG_CONFIG_HOME=/data/storage/el2/base/haps/entry/files
__LIBACE_ENTRY_POINT=59f6cc9a34
USER=100
HNP_PRIVATE_HOME=/data/app
HNP_PUBLIC_HOME=/data/service/hnp
...
```

判定依据：

- `XDG_CACHE_HOME` 与 `XDG_CONFIG_HOME` 指向 `/data/storage/el2/base/haps/entry/files`。`el2` 与 `haps`（Harmony Ability Package）为 OpenHarmony 应用沙箱路径特征。
- `__LIBACE_ENTRY_POINT` 表明进程运行在 ACE（ArkUI）运行时内。
- `HAP_DEBUGGABLE=false` 表明该应用包不可调试。
- `ALACRITTY_WINDOW_ID` 表明该终端为第三方应用 Alacritty，而非系统终端。
- `USER=100`，非 root。

**结论：所使用终端是运行在应用沙箱内的一个鸿蒙应用，不是系统 shell。**

这解释了以下权限受限现象：

```
$ ls /
ls: /: Permission denied

$ ls /bin
ls: /bin: Permission denied

$ ls /system
ls: /system: Permission denied

$ ls /data
ls: /data: Permission denied

$ ls -la /storage/Users/currentUser/appdata
ls: /storage/Users/currentUser/appdata: Operation not permitted
```

### 9. HNP 机制

```
$ command -v hnp
（无输出）
```

环境变量 `HNP_PUBLIC_HOME=/data/service/hnp` 存在，但该目录不存在。说明 HNP 机制在系统中存在，但当前未安装任何 HNP 包。

## 五、阻塞影响

以下为 2026-09-25 探测当日的状态。当日该机既无 Python 运行时，也无包管理器和编译器，因此：

| 负责人 | 工作项 | 受影响情况 |
|---|---|---|
| 3号 | 鸿蒙 Python 环境 | 阻塞，无法完成环境搭建 |
| 4号 | 依赖验证（NumPy/SciPy/Numba 等） | 阻塞，无解释器可安装依赖 |
| 5号 | 源码适配与示例运行 | 阻塞，无法在本机运行 |
| 6号 | 打包与全新环境复测 | 阻塞，无法在本机安装验证 |

第 5 天决策门条件不满足。

补充说明：即使存在裸 Python 解释器，由于缺少 C 编译器与 `make`，numpy、scipy、numba 等依赖在 aarch64 鸿蒙平台上大概率无预编译 wheel，需从源码构建，当前工具链无法支持（numba 另需 LLVM，构建难度更高）。因此本机不具备完成移植所需的完整工具链。

## 六、未验证项

1. **开发者模式**：未开启、未验证。
   - 未验证原因：该设备为官方借用设备，需归还。开启开发者模式存在引入系统故障的风险，经评估后主动放弃。
   - 需说明：`HAP_DEBUGGABLE=false` 已表明当前沙箱应用不可调试；即便开启开发者模式，也不会新增 Python 运行时，仅可能提升对 `/system`、`/data` 目录的读取权限。
   - 该未验证项不影响主要结论，原因见第一节。

2. **华为是否以 HNP 形式提供 Python 运行时**：未确认。

3. **应用市场是否存在官方原生 Python 应用**：未确认。

## 七、交付物情况

| 计划交付物 | 状态 |
|---|---|
| `scripts/check_environment.py` | 已完成，已在 Windows + CPython 3.10.21 上验证通过 |
| 鸿蒙 PC Python 环境说明 | 以本报告替代，待组长确认 |
| 环境安装和验证日志 | 以本报告第四节的探测记录替代，待组长确认 |
| 可重复执行的环境准备步骤 | 无法产出，前提条件不成立 |

关于 `check_environment.py`：

- 该脚本为纯标准库实现，用于检测目标机是否具备运行 RayTracePy 的条件，输出文本报告和 JSON。
- 已在 Windows + CPython 3.10.21 上完成验证，覆盖依赖存在与缺失两种情形；修正后另在 Windows + CPython 3.13.13 上复验，并单独验证了安装有 numba 0.67.0 时的 `@njit` 冒烟探针。
- **未能在目标设备上执行**，原因是设备不存在 Python 解释器。此事实本身构成一条证据。

## 八、待决事项

1. 是否向老师申请调整验收范围。
2. 上表中三项交付物的替代方案是否获批。
3. 3号后续任务如何重新分配（阻塞已于 2026-09-26 解除，现由第十节第 4 小节的验收标准问题决定）。
4. ~~是否通过官方渠道确认 HNP 形式的 Python 运行时或原生 Python 应用。~~
   已于 2026-09-26 以 harmonybrew + Alpine CPython 方案绕开，不再是阻塞项；
   但该方案对"原生"的限定需先澄清，见第十节第 1 小节。

## 九、边界声明

依据项目分工文档，本人在确认无可用运行时后，未采取以下操作：

- 未安装非官方 Python 二进制。
- 未尝试开启 root 或规避系统权限。
- 未改用虚拟机、容器或兼容层方案。
- 未伪造任何设备端执行记录。

## 十、后续进展（待确认项已由组长答复）

2026-09-26 组长告知：目标设备的 Python 环境已安装完成，第一节与第五节所述阻塞在事实上已解除。组长所述现状如下（本文未在设备上复核）：

| 项目 | 现状 |
|---|---|
| 包管理器 | harmonybrew（`brew`）已安装 |
| 工具链 | `/data/service/hnp/bin` 下 hnp 工具链约 452 项，含 `clang`/`clang++` 15.0.4、`make`、`cmake`、`llvm-config`、binutils、`patchelf`、`gdb`/`lldb` |
| 解释器 | CPython 3.10.15，位于 `~/.local/alpine-py310` |
| 虚拟环境 | 项目 `.venv` 基于上述解释器 |

因此，第五节“无包管理器和编译器”与第三节“`/data/service/hnp/bin` 不存在”的表述，均只对 2026-09-25 当日成立，不再适用于当前状态。

### 1. 解释器来源：原生执行，但不是为鸿蒙编译

原问询事项为：`~/.local/alpine-py310` 中的 CPython 3.10.15 是鸿蒙原生构建，还是基于
Alpine Linux / musl 的 Linux 构建？**2026-09-26 组长答复如下。**

- **来源**：Alpine Linux 官方的 CPython 3.10 二进制包，**不是为鸿蒙编译的**。此点组长明确不作含糊表述。
- **选用原因**：版本约束。harmonybrew 只有 `python@3.12` / `3.13` / `3.14`，没有 `python@3.10`，而参考栈固定为 3.10。
- **运行方式**：在设备上**原生执行** —— aarch64 本机指令，链接系统自带的 musl libc。**无虚拟机、无容器、无模拟层。**
- **改动范围**：仅 ELF 元数据（签名、libc 名称、libpython 依赖、替换 libffi、`PT_INTERP`）与两处 stdlib 文本。
- **兼容 shim**：仅 **2 个**，只挂在 **numba 的 4 个扩展**和 **pillow 的 libzstd** 上；numpy / scipy / pandas / datashader 一个都没用。

据此修正第九节相关表述：本次安装**不构成"以兼容层替代原生"**（无模拟层），
但**也不满足"为鸿蒙编译"**。这是两件需要分开判断的事，故本节不作结论，
升级为待决事项（见第 4 小节）。第九节"未改用虚拟机、容器或兼容层方案"描述的是
本人探测当日的行为，不涉及本次由他人完成的安装。

若验收要求必须是原生编译，组长给出的现成方案只有 harmonybrew 的 `python@3.12`，
代价是版本从 3.10 变为 3.12，且 llvmlite 需重新编译。**该取舍需老师裁定。**

> 补充一条**未经验证的推测**，不作为结论：兼容 shim 恰好挂在 numba 的 4 个扩展上，
> 而观察到的故障恰好是 numba JIT 产物执行时段错误。二者是否相关，值得在排查 JIT
> 段错误时优先检查 shim；但现有证据不足以认定因果关系。

### 2. numba JIT 的实际状态：编译可用，产物不可执行

原问询事项为：关闭 JIT 是否意味着 numba 路径一次也未真正执行？
**组长答复：不是"一次都没跑起来"。** 准确的分层是：

| 层次 | 结果 |
|---|---|
| `import numba` | 成功 |
| JIT 编译 | 成功 |
| **执行编译产物** | **段错误** |

因此准确表述是**编译能工作、产物不能执行**；实际执行的是同一份代码的纯 Python 路径。

由此确定影响范围：

- 7 个测试验证的是 RayTracePy 的**算法与数值**，**不是** numba 的 JIT，不能作为 JIT 可用的证据。
- 但有一项比"数值一致"更强的证据：300 万光线的参考负载（种子 `20260920`），
  在设备上以纯 Python 运行得到的 `histogram_sha256`，与 Windows **开启 numba JIT**
  的参考值**逐位相同**。
- 结论：**关闭 JIT 只影响速度，不影响结果。**

本人已核对 `artifacts/reference/windows-python310/`，该参考值确为 Windows 侧产出、未被污染
（`git status` 无未提交改动，7 个文件时间戳同为 2026-09-25 13:34，为单次运行产出）：

```text
seed=20260920
total_rays=3000000
histogram_sha256=bacd25e64f5ca3bd0d6df9bf7fa775d25d8410cea055368bebb3a0198d0cc677
duration_seconds=22.328691
repeat_identical=True
```

同目录 `environment.json` 记录 `machine=AMD64`、`system=Windows`、
`platform=Windows-10-10.0.26200-SP0`、`python_version=3.10.21`，与参考环境一致。

### 3. 耗时差异的更正

上一版本节把"设备 66 秒"与参考的 22.33 秒并列比较，**该比较不成立**，现更正：

- 66 秒那次运行的是 `examples/single/single_light.py`，**未固定种子**，与参考的 22 秒
  **不是同一次测量**。把两者当作同一负载的对比是本文的错误。
- 同一输入下的实际对比是：**设备 64.07 秒 vs Windows 22.33 秒，约 2.87 倍。**
- 该差异包含两个因素：**关闭 JIT**、**硬件不同**（参考为 Intel AMD64，设备为 Kirin X90 / aarch64）。
- 在 JIT 能在设备上跑起来之前，**这两个因素无法定量拆分**。
- 数值一致性：逐字段差异 ≤ 2e-17，优于仓库自设的 `rtol=1e-6` / `atol=1e-8`。

### 4. 待决事项（需老师裁定，非本人可决定）

1. **验收标准是否要求"为鸿蒙编译"。** 现用解释器原生执行但非鸿蒙编译。
   若标准是"原生执行"，现有环境即满足；若标准是"为鸿蒙编译"，则需改用
   harmonybrew 的 `python@3.12`，并承担版本变更与 llvmlite 重编的代价。
2. **numba JIT 段错误是否必须修复。** 现状可交付：结果与 Windows 开 JIT 逐位相同，
   代价是约 2.87 倍耗时。若不接受该耗时，则需继续排查段错误。
3. 第八节其余事项（交付物替代方案、3号后续任务分配）仍待处理。

### 5. 后续动作

`scripts/check_environment.py` 已按组长审查意见完成两轮修正，并新增 `@njit` 编译并调用
的一次冒烟探针，用于把"可导入但不能执行"这类情形自动检出。

**下一步应在设备上重跑该脚本**，用其输出替换第一节的旧结论，并作为第 1 小节所述
解释器来源的独立佐证（脚本会记录 `libc`、解释器路径与 `@njit` 探针结果）。

**该重跑已于 2026-09-26 完成（由 1号 执行），结果见第十一节。** 需更正一处：
`@njit` 冒烟探针只覆盖"简单函数可编译并执行"，**不能**代表 RayTracePy 自身的
JIT 路径可用——完整测试在 JIT 开启时仍段错误，见第十一节第 2 小节。

### 6. 移交：对照脚本的一处覆盖风险（属 2号 交付物）

组长在核对设备与参考结果时触发此问题：`scripts/generate_reference_baseline.py` 的
`--expected-output` **是输出参数，不是输入参数** —— 它把本次运行的指标写到该路径，
而不是读入该路径做对比。`write_json()`（第 190 行）不检查文件是否存在、不备份，
会**静默覆盖**已有文件；`parse_args()`（第 276 行）也未提供 `help` 文本。

风险在于：若把它指向 `artifacts/reference/windows-python310/single_light_metrics.json`，
参考基线会被设备数字覆盖，此后所有跨平台对比都变成循环论证。组长已还原。

本人**未修改该文件**（属 2号 的 `feature/baseline-tests` 交付物，改动需经其确认）。
建议的最小保护是在覆盖已存在的文件时要求显式 `--force`，并把参数名改为能体现
"写出"含义的名称。

> 该风险不影响本节第 2 小节的结论：本人已单独核对参考目录，未见污染。

## 十一、环境验证（2026-09-26 实测，1号 补充）

### 1. `scripts/check_environment.py` 判定 OK

```console
$ .venv/bin/python scripts/check_environment.py --boot-banner
label        : linux-python310
system       : Linux
machine      : aarch64
host gnu type: aarch64-alpine-linux-musl
pip          : present 26.2.1
setuptools   : 65.5.0 (>=42: True)
[required] numpy / numba / plotly / pandas  present
[important] scipy / llvmlite                present
[optional] datashader / pytest              present
njit compile+call: OK   returned 45.0   numba 0.61.2
blockers     : none
warnings     : none
RESULT       : OK
```

完整输出见 `artifacts/environment/harmonyos-pc/check_environment.log`，机器可读结果见
同目录 `environment.json`（键集覆盖 `artifacts/reference/windows-python310/environment.json`
的全部键，可直接对比）。

### 2. 完整测试：JIT 开启段错误，纯 Python 通过

```console
$ pytest -q tests/                       # 默认，JIT 开启
..Fatal Python error: Segmentation fault
  File ".../src/raytracepy/raytrace.py", line 175 in run
退出码 139

$ NUMBA_DISABLE_JIT=1 pytest -q tests/
7 passed in 0.72s
```

- 第十节第 2 小节"编译可用、产物不可执行"的结论在完整测试上复现，崩溃点为
  `src/raytracepy/raytrace.py:175`。
- 第十节第 2 小节"关闭 JIT 只影响速度，不影响结果"成立：`7 passed` 中包含
  `histogram_sha256` 逐位断言，即纯 Python 结果与参考完全一致。
- 但**关闭 JIT 需要额外条件**：`NUMBA_DISABLE_JIT=1` 原先不生效，因为
  `src/raytracepy/__init__.py` 在导入时无条件执行 `numba.config.DISABLE_JIT = False`
  （实测：设置变量后 `before 1` / `after False`）。已提交修复分支
  `fix/numba-disable-jit-override`：默认行为不变，仅在显式设置该变量时尊重它。
  加上该守卫后，上表第 2 条命令才成立。
- 据此，第十节第 2 小节"实际执行的是同一份代码的纯 Python 路径"需补充说明：
  该路径**不能**仅靠环境变量得到，必须配合上述源码守卫。

证据文件：`artifacts/environment/harmonyos-pc/pytest-evidence.log`。

### 3. 遗留

- 验收口径（是否要求"为鸿蒙编译"、是否必须修复 JIT 段错误）见第十节第 4 小节，待老师裁定。
- 环境准备工具目前位于仓库外的 `~/.local/ohos-python-tools/`，建议归档后再作为验收依据，
  见 `PYTHON_ENVIRONMENT.md` 第三节与第四节第 5 条。

## 附录：/usr/bin 完整清单

```
awk           cp        file            hitrace   mount       printenv   sha1sum     telnet       uptime
base64        cpio      find            hostname  mountpoint  printf     sha256sum   test         usleep
basename      crc32     flock           iconv     mv          ps         sha384sum   time         uudecode
bunzip2       curl      fmt             ifconfig  netcat      pwd        sha512sum   timeout      uuencode
bzcmp         cut       free            iotop     netstat     pwdx       shred       timestamps   uuidgen
bzdiff        date      ftpget          ipcs      nl          readahead  sleep       toe          vi
bzegrep       dd        ftpput          join      nohup       readlink   sort        top          view
bzfgrep       df        getconf         kill      nproc       realpath   split       touch        vim
bzgrep        diff      getfattr        killall   od          reboot     ssh         tr           vmstat
bzip2         dirname   grep            ln        openssl     reset      ssh-keygen  traceroute   watch
bzip2recover  dos2unix  gunzip          loh       param       rev        stat        traceroute6  wc
bzmore        du        gzip            ls        paramshell  rm         strings     true         wget
cal           echo      head            lsattr    paste       rmdir      sudo        truncate     which
cat           egrep     help            lsmod     patch       route      sync        tsort        xargs
chmod         env       hexedit         mcookie   pgrep       scp        sysctl      tty          xxd
cksum         expand    hidumper        md5sum    pidof       sed        tac         uname        yes
clear         expr      hilog           mkdir     ping        seq        tail        uniq         zcat
cmp           factor    hiperf          mkpasswd  ping6       setfattr   tar         unix2dos     zip
comm          false     hiprofiler_cmd  mktemp    pkill       sftp       taskset     unlink       zsh
count         fgrep     hisysevent      more      pr          sh         tee         unzip
```
