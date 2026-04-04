# 任务2测试报告：GLM-4.7-Flash 在当前基线不可运行（可稳定复现）

## 1. 报告目的

本报告用于完成任务2交付，覆盖以下目标：

- 当前代码基线下，模型是否可运行。
- 当模型不可运行时，是否可清晰定位阻塞点。
- 提供完整证据链：测试步骤、错误日志、根因分析、适配差距（GAP）结论。
- 按要求披露 AI 辅助分析方式（agent / prompt）。

---

## 2. 执行环境

### 2.1 代码与版本

- 仓库路径：`/root/jupyter/workspace/vllm-ascend`
- Git Commit：`0fccd729`
- Python：`3.11.13`
- vLLM：`0.11.0`
- 操作系统：Linux

### 2.2 说明

- 本次基于回滚后的真实基线复现，不引入临时修复。
- 目标是确认当前状态下首个阻塞点，并给出系统性适配结论。

---

## 3. 测试范围与方法

### 3.1 测试目标

验证 e2e 用例是否可启动 GLM-4.7-Flash 评测流程：

- 用例：`tests/e2e/models/test_lm_eval_correctness.py`
- 配置：`tests/e2e/models/configs/GLM-4.7-Flash.yaml`

### 3.2 执行命令

```bash
cd /root/jupyter/workspace/vllm-ascend
/usr/local/python3.11.13/bin/python -m pytest tests/e2e/models/test_lm_eval_correctness.py --config tests/e2e/models/configs/GLM-4.7-Flash.yaml --tp-size 1 -q
```

### 3.3 结果概览

- 结果：失败
- 退出码：`4`
- 失败阶段：pytest 加载 conftest 阶段（尚未进入模型执行主体）

---

## 4. 错误日志（关键摘录）

```text
INFO 04-04 16:05:31 [__init__.py:36] Available plugins for group vllm.platform_plugins:
INFO 04-04 16:05:31 [__init__.py:38] - ascend -> vllm_ascend:register
INFO 04-04 16:05:31 [__init__.py:41] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 04-04 16:05:31 [__init__.py:207] Platform plugin ascend is activated
WARNING 04-04 16:05:31 [_custom_ops.py:20] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
ImportError while loading conftest '/root/jupyter/workspace/vllm-ascend/tests/e2e/conftest.py'.
tests/e2e/conftest.py:56: in <module>
    from vllm.utils.network_utils import get_open_port
E   ModuleNotFoundError: No module named 'vllm.utils.network_utils'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

---

## 5. 首个阻塞点与直接根因

### 5.1 首个阻塞点

- 文件：`tests/e2e/conftest.py:56`
- 语句：`from vllm.utils.network_utils import get_open_port`
- 异常：`ModuleNotFoundError: No module named 'vllm.utils.network_utils'`

### 5.2 直接根因

`get_open_port` 在 vLLM 0.11 中已迁移至 `vllm.utils`，旧路径 `vllm.utils.network_utils` 不再可用。

### 5.3 证据

- 实际符号：`vllm.utils.get_open_port`
- 定义位置：`/vllm-workspace/vllm/vllm/utils/__init__.py`

### 5.4 非环境损坏判定

- `import vllm` 正常，版本识别为 0.11.0。
- 同名函数存在，仅路径发生迁移。
- 结论为 API 漂移，不是包损坏。

---

## 6. 深度追踪分析（整合版）

当前首错表现为单行导入失败，但该问题在兼容排查中属于“版本漂移链”的首个可见断点，而非唯一问题。

### 6.1 复杂度来源

- 模块路径迁移
- 符号删除/重命名
- 配置字段变更
- dataclass 基类字段顺序变化
- 可选功能启动期硬导入
- NPU 与 CUDA 运行分支差异

上述问题会导致“修复一个阻塞点后，错误向更深层推进”的串行暴露特征。

### 6.2 历史排查轨迹（按层级）

1. 测试入口与基础导入层
   - `vllm.utils.network_utils` 路径失效。
   - `vllm.v1.attention.backend` 等路径在目标环境不可用。

2. Model Runner 初始化层
   - 多个符号缺失：`ECConnectorOutput`、`make_empty_encoder_model_runner_output`、`mamba_utils`、`cp_utils`、`maybe_create_ubatch_slices`。
   - 多个配置字段漂移：`profiler_config`、`is_mm_prefix_lm`、`prefill_context_parallel_size`。

3. Attention 与 Context Parallel 层
   - `cdiv` 路径迁移、backend registry 位置变化。
   - `get_pcp_group` 缺失，需要 DCP/单卡兼容分支。
   - `CommonAttentionMetadata` 字段变化引发 dataclass 结构冲突。

4. 算子注册与可选模块层
   - `ApplyRotaryEmb` 路径/结构变化，影响 OOT custom op。
   - `process_eagle_weight` 缺失。
   - quantization 模块硬导入触发 `CompilationMode` 缺失。

5. 运行时平台行为层
   - `fork` 与 NPU 冲突，需要 `spawn`。
   - CUDA 假设内存探测在无 CUDA 环境失败。
   - distributed backend 类型规范不一致导致 `Unknown backend type`。

---

## 7. 根因树（Root Cause Tree）

### 7.1 一级根因

- 版本组合不一致：vllm-ascend 预期接口与 vLLM 0.11 实际接口未完全对齐。

### 7.2 二级根因

- 模块重构后调用点未同步。
- 接口签名/返回/枚举位置变化。
- 配置字段改名或移除。
- NPU 平台启动与探测策略差异。
- 可选能力与主链路耦合过深。

### 7.3 三级表现

- 表面是离散报错，实质是跨版本契约失配的连续暴露。

---

## 8. 适配差距（GAP）结论

### GAP-1（P0）测试入口兼容缺失

- 范围：`tests/e2e/conftest.py`
- 影响：e2e 在初始化阶段失败，无法进入模型执行。

### GAP-2（P0）基础启动链兼容不足

- 范围：worker/model_runner/attention/distributed 初始化路径
- 风险：即使修复入口，仍会在引擎启动阶段中断。

### GAP-3（P1）可选模块耦合过深

- 范围：quantization、draft、特定算子
- 风险：未启用能力仍可阻塞主流程。

### GAP-4（P1）平台分支默认策略不稳

- 范围：NPU multiprocessing、memory profiling
- 风险：环境变化后稳定性下降。

### GAP-5（P2）测试入口缺少版本兼容层

- 范围：tests/e2e/conftest 与上游 API
- 风险：测试在核心逻辑前提前失败，不能反映真实模型能力。

---

## 9. 复杂度结论

本问题不属于单点简单错误。

虽然当前首错是单一导入失败，但该错误仅是兼容链条的首个可见断点。后续阻塞连续分布于测试入口、引擎初始化、attention、分布式、量化与平台运行时。排查耗时主要来自逐层回归与深层阻塞点串行暴露。

---

## 10. 建议修复方向（供任务3或后续处理参考）

针对当前首错，建议先采用最小侵入兼容导入：

```python
try:
    from vllm.utils import get_open_port
except ImportError:
    from vllm.utils.network_utils import get_open_port
```

说明：本报告为任务2交付，上述内容为建议，不表示已在当前基线提交修复。

---

## 11. 可复现性说明

可复现条件：

- 使用本报告中相同 Python / vLLM 版本。
- 使用 commit `0fccd729`。
- 执行第 3.2 节命令。

预期结果：

- 在 conftest 阶段出现相同 `ModuleNotFoundError`。

---

## 12. 任务2交付清单对应关系

已完成并覆盖：

- 测试步骤：第 3 节
- 错误日志：第 4 节
- 根因分析：第 5 至第 7 节
- GAP 结论：第 8 节

---

## 13. AI 辅助分析记录（按要求披露）

### 13.1 使用方式

- 本地复现实验与日志解析
- 只读代码检索与路径验证
- Agent 辅助交叉验证（Explore）

### 13.2 Agent 信息

- Agent 名称：`Explore`
- 任务类型：只读分析（quick）

### 13.3 使用 Prompt

```text
请对仓库 /root/jupyter/workspace/vllm-ascend 做只读分析（quick）。目标：确认 tests/e2e/conftest.py 中 `from vllm.utils.network_utils import get_open_port` 在当前环境(vLLM 0.11)为什么失败，并给出最小兼容建议。请返回：1) 失败根因一句话；2) vLLM 0.11 中 get_open_port 的实际定义路径；3) 建议的兼容导入代码片段（try/except）；4) 为什么这是 API 漂移而非环境损坏。
```

### 13.4 AI 分析结论摘要

- 根因：`get_open_port` 模块路径在 vLLM 0.11 发生迁移。
- 新路径：`vllm.utils.get_open_port`。
- 定性：API 漂移，不是运行环境损坏。

---

## 14. 最终结论

在 commit `0fccd729` 与 vLLM `0.11.0` 环境下，GLM-4.7-Flash 对应 e2e 测试链路无法启动。

结论具备以下特征：

- 首错可稳定复现且证据完备。
- 深度排查显示该问题属于系统性兼容链，不是单点偶发。
- 满足任务2“模型无法直接运行，但可清晰定位问题”的交付条件。