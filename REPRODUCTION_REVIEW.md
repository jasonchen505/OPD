# Rethink OPD 复现项目审查报告

审查对象：`/home/chenyizhou/OPD`

审查日期：2026-06-10

审查范围：本次审查覆盖远程项目中已有代码、tracked 本地改动、复现新增脚本、评测产物、训练日志和 `reproduction_learning_notes/` 下新增文档。报告只记录问题和建议，不修改训练代码。

## 结论摘要

当前复现已经跑通了核心链路：OPD 训练 checkpoint 记录到 step 1119，最终评测产物包含 AIME24 480 条、AIME25 480 条、AMC23 640 条 JSONL 输出，`grading_results.json` 中记录了 `MAX_TOKENS=16384` 的评测结果。

但项目仍有几类会影响复现可信度和后续维护的问题：

- 高优先级问题 4 个：flash-attn fallback API 不兼容、评测生成脚本可能静默产出不完整结果、运行入口依赖错误 Python、训练长度与评测长度口径混用。
- 中优先级问题 6 个：评分 verifier 分支错误、硬编码路径/GPU/模型、Ray 与调试变量副作用、GRPO/OPD 文档证据错位、模型下载记录过期、论文对比结论缺少同口径证据。
- 低优先级问题 4 个：评测长度 tokenizer 口径不一致、`seed` 字段命名误导、SFT demo 配置容易被误读为正式复现、环境版本与依赖锁定不足。

建议优先修复可导致代码路径崩溃或指标失真的问题，再整理文档证据链。

## 已核验证据

### 代码与脚本状态

- tracked 本地改动集中在 4 个文件：`scripts/val/eval/gen_vllm.py`、`scripts/val/eval/grade.py`、`verl/verl/utils/attention_utils.py`、`verl/verl/workers/fsdp_workers.py`。
- 新增复现相关文件包括：`opd_3090.sh`、`grpo_3090.sh`、`LlamaFactory/examples/train_full/qwen3_base_full_sft_3090.yaml`、`reproduction_learning_notes/`。
- `bash -n opd_3090.sh grpo_3090.sh` 通过。
- `.venv/bin/python -m py_compile scripts/val/eval/gen_vllm.py scripts/val/eval/grade.py` 通过。
- 裸 `python3 --version` 是 `Python 3.5.4`，`.venv/bin/python --version` 是 `Python 3.12.3`。

### 训练与评测产物

- OPD checkpoint tracker：`checkpoint/.../latest_checkpointed_iteration.txt = 1119`。
- 最终评测 JSONL 行数：
  - AIME24：480 行。
  - AIME25：480 行。
  - AMC23：640 行。
- `grading_results.json` 的 `MAX_TOKENS=16384` 结果：
  - AMC23：`mean_score=0.721875`，`best_score=0.975`，`solve_all=16`，`format_error_rollouts=114`。
  - AIME24：`mean_score=0.28125`，`best_score=0.6333333333333333`，`solve_all=1`，`format_error_rollouts=206`。
  - AIME25：`mean_score=0.23958333333333334`，`best_score=0.4666666666666667`，`solve_all=1`，`format_error_rollouts=199`。

### 关键日志事实

- OPD 日志 `logs/opd_3090_20260530_190228.log`：
  - step 1：`actor/pg_loss=0.2616005335`，`critic/score/mean=-0.261600554`，`topk/overlap_ratio=0.7222576141`。
  - step 2：`actor/pg_loss=0.2745051011`，`critic/score/mean=-0.2730915248`，`topk/overlap_ratio=0.7318731546`。
  - step 10：`actor/pg_loss=0.2737934291`，`critic/score/mean=-0.2613928020`，`topk/overlap_ratio=0.7240221500`。
  - step 1119：`actor/pg_loss=0.2205111142`，`critic/score/mean=-0.2039521337`，`topk/overlap_ratio=0.7624691129`。
  - 实际命令包含 `trainer.save_freq=200`。
- GRPO 日志 `logs/grpo_3090_20260530_170411.log`：
  - step 4：`actor/pg_loss=0.0049668346`，`critic/score/mean=0.03125`。
  - step 10：`actor/pg_loss=0.0043777823`，`critic/score/mean=0.03125`。
  - 实际命令包含 `trainer.save_freq=20`。

## 高优先级问题

### H1. flash-attn fallback API 不兼容，部分 verl 路径会直接失败

位置：

- `verl/verl/utils/attention_utils.py:48-53`
- 受影响调用示例：`verl/verl/models/qwen2/megatron/modeling_qwen2_megatron.py`、`verl/verl/models/llama/megatron/modeling_llama_megatron.py`

问题：

`attention_utils.py` 中新增的纯 PyTorch fallback 只返回：

```python
return index_first_axis(hidden_states, indices), indices, seqlen
```

但 flash-attn / transformers / `npu_utils.py` 的兼容语义是返回至少 5 项：`hidden_states`、`indices`、`cu_seqlens`、`max_seqlen_in_batch`、`used_seqlens_in_batch`。Megatron 模型路径会按：

```python
input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(...)
```

解包，当前 fallback 会触发 `ValueError: not enough values to unpack`。

另外，fallback 中的：

```python
def rearrange(x, pattern):
    return einops.rearrange(x, pattern)
```

不支持 `einops.rearrange(x, "...", b=batch)` 这种 kwargs 调用。虽然当前代码主路径多数直接导入 `einops.rearrange`，但这个 wrapper 的公开行为仍然和 flash-attn 提供的 `rearrange` 不一致。

原因：

fallback 只按当前 FSDP OPD 路径的最小需求实现，未对齐 `flash_attn.bert_padding.unpad_input` 的完整接口。

影响：

- 当前 FSDP 路径因为大量使用 `input_ids_rmpad, indices, *_ = unpad_input(...)`，可以侥幸跑通。
- 切到 Megatron、部分 SFT、sequence parallel 或其他依赖 `cu_seqlens` 的路径时会崩溃。
- 这是兼容性 bug，不只是性能退化。

潜在解法：

- 优先使用 transformers 已提供的 fallback：
  - `transformers.modeling_flash_attention_utils._unpad_input`
  - `transformers.modeling_flash_attention_utils._pad_input`
  - `transformers.modeling_flash_attention_utils._index_first_axis`
- 或参考 `verl/verl/utils/npu_utils.py:99-128`，补全 `cu_seqlens`、`max_seqlen_in_batch`、`used_seqlens_in_batch`。
- `rearrange` wrapper 应改为 `def rearrange(x, pattern, **axes_lengths): return einops.rearrange(x, pattern, **axes_lengths)`。
- 增加一个轻量单测：无 flash-attn 环境下调用 `unpad_input`，断言返回值数量、`cu_seqlens` dtype/shape、`pad_input(unpad_input(...))` 可 round-trip。

### H2. 评测生成脚本会吞掉 worker 异常，可能静默保存不完整结果

位置：

- `scripts/val/eval/gen_vllm.py:185-187`
- `scripts/val/eval/gen_vllm.py:291-312`

问题：

`worker_process` 捕获所有异常后只打印 `Critical Error`，然后返回当前已有 `results`。主进程也捕获 future 异常并继续，最后只要 `all_results` 非空就写文件。

脚本没有检查 `len(all_results) == len(samples) * N`，也没有检查每个 `example_id` 是否恰好有 N 个 rollout。

原因：

为避免多进程评测中断，异常处理过于宽松，没有把数据完整性作为硬约束。

影响：

- 某个 GPU worker OOM、tokenizer 报错或 vLLM 初始化失败时，脚本仍可能生成 JSONL。
- 后续 `grade.py` 会按现有输出计算结果，导致指标在不完整样本上被错误解释。
- 复现报告中指标可信度依赖人工检查行数，不够稳健。

潜在解法：

- worker 异常应重新抛出，主进程收到任一失败 future 后 `raise SystemExit(1)`。
- 保存前断言：
  - 总数等于 `len(samples) * N`。
  - 每个 `example_id` 的响应数等于 N。
  - 所有任务的输出文件行数符合预期。
- 在 JSONL 中额外写入 `model_name`、`task_name`、`max_tokens`、`temperature`、`top_p`、`rollout_id`，便于后续审计。

### H3. 运行入口依赖裸 `python3`，远程系统 Python 版本不满足项目要求

位置：

- `opd_3090.sh:89`
- `grpo_3090.sh` 中同类调用
- 文档 `reproduction_learning_notes/07_reproduction_log.md:36-44`

问题：

脚本使用：

```bash
python3 -m verl.trainer.main_ppo
```

但远程机器当前裸 `python3` 是 `Python 3.5.4`，而项目实际可用解释器是 `.venv/bin/python` 的 `Python 3.12.3`。

原因：

脚本假定调用前已经激活 `.venv`，但没有在脚本内验证环境。

影响：

- 新 shell、tmux 重启、cron 或他人复现实验时，可能直接使用 Python 3.5.4 运行，导致语法错误或模块缺失。
- 评测脚本含 f-string 和 Python 3.9+ 类型标注，在裸 `python3` 下无法编译。

潜在解法：

- 在脚本顶部加入：

```bash
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 12)'
```

- 将所有 `python3 -m ...` 改为 `"$PYTHON_BIN" -m ...`。
- 报告和 README 中明确必须从 `/home/chenyizhou/OPD` 运行，并使用 `.venv/bin/python`。

### H4. 训练响应长度和最终评测长度口径混用，结论容易被过度解读

位置：

- `opd_3090.sh:48-49`
- `scripts/val/eval/gen_vllm.py:57`
- `reproduction_learning_notes/07_reproduction_log.md:212-256`
- `reproduction_learning_notes/README.md:32-38,126`

问题：

训练时：

```bash
MAX_RESP_LENGTH=2048
MAX_VAL_RESP_LENGTH=2048
```

最终离线评测时：

```python
MAX_TOKENS = 16384
```

文档正确记录了 2048 与 16384 的评估差异，但有些总结位置仍把 16384 的高分当作复现成功主结论，另一些 checklist 仍保留 2048 的旧分数。

原因：

为适配 3090，训练上下文大幅缩短；为降低 format error，最终评测又拉长生成上限。这两个设置解决的是不同问题，但文档没有稳定区分“训练设置”“verl 内置验证设置”“离线评测设置”。

影响：

- 读者可能误以为训练过程也使用了 16K 响应长度。
- 与论文或上游结果对比时，若没有相同 max tokens、rollout 数、评分方式，就会出现口径不一致。

潜在解法：

- 在报告/README 中新增一个固定表格，分开列出：
  - train `MAX_RESP_LENGTH=2048`
  - verl validation `MAX_VAL_RESP_LENGTH=2048` 且 `trainer.test_freq=-1`
  - offline eval `MAX_TOKENS=16384`
- 所有结论中统一写成：“在 2048 响应长度训练后，使用 16384 离线生成评测得到 ...”。
- 与论文对比前补齐同口径 baseline：同模型、同 N、同 max tokens、同评分器。

## 中优先级问题

### M1. `grade.py --enable_model_verifier` 分支存在 prompt 丢失和输出协议错误

位置：

- `scripts/val/eval/grade.py:63-75`
- `scripts/val/eval/grade.py:124-128`
- `scripts/val/eval/grade.py:169-171`

问题：

`process_jsonl_file` 只保存 `gt` 和 `responses`，没有保留 JSONL 中已有的 `prompt`。因此 JSONL 评分路径下：

```python
question = df[i].get("question", "")
```

永远得到空字符串。

同时 verifier prompt 中既列出 `A/B/C`，又写“reply with either CORRECT, INCORRECT, or INVALID”，但代码只判断：

```python
model_score = "A" == judgement
```

原因：

JSONL 数据结构和 verifier prompt 是后续补丁式修改，未做端到端验证。

影响：

- 默认规则评分不受影响。
- 一旦启用 `--enable_model_verifier`，模型 verifier 会在缺少题目的情况下判断，且如果它按提示返回 `CORRECT`，代码会误判为 False。

潜在解法：

- JSONL 聚合时保留 `prompt`：

```python
results[id] = {"gt": None, "question": data.get("prompt", ""), "responses": []}
```

- 统一 verifier 输出协议：要么 prompt 只要求 `A/B/C`，要么代码同时接受 `A` 和 `CORRECT`。
- 对 verifier 分支增加一个小样本回归测试。

### M2. 评测脚本硬编码模型、GPU、max tokens 和显存比例

位置：

- `scripts/val/eval/gen_vllm.py:48`
- `scripts/val/eval/gen_vllm.py:57`
- `scripts/val/eval/gen_vllm.py:124-128`
- `scripts/val/eval/gen_vllm.py:236-238`
- `scripts/val/eval/grade.py:36-45`

问题：

评测脚本把模型路径、`MAX_TOKENS=16384`、GPU 列表 `[0..7]`、`gpu_memory_utilization=0.7`、长度 tokenizer 等全部写死。

原因：

脚本从一次性复现实验逐步改造而来，还没有变成可配置入口。

影响：

- 评测另一个 checkpoint 时需要手改源码，容易污染 git diff。
- 在 GPU 数量不同或只想评单卡/部分任务时不可复用。
- `grade.py` 用 Qwen3 tokenizer 统计 DeepSeek 输出长度，`avg_output_length` 只能作为近似值。

潜在解法：

- 增加 CLI 参数：`--model`, `--tasks`, `--gpu-ids`, `--max-tokens`, `--gpu-memory-utilization`, `--output-dir`, `--replace`。
- `grade.py` 增加 `--eval-dir`、`--length-tokenizer`、`--output-file`。
- 文档记录本次实际命令，避免“改代码即配置”的复现方式。

### M3. OPD/GRPO 脚本会影响同机服务，并默认开启重调试变量

位置：

- `opd_3090.sh:18-22`
- `opd_3090.sh:84-85`
- `grpo_3090.sh` 同类位置

问题：

脚本硬编码：

```bash
export RAY_TMPDIR=/mnt/sdb2/chenyizhou/OPD/ray_tmp
ray stop --force
export CUDA_LAUNCH_BLOCKING=1
```

原因：

这些设置来自排障过程，后来进入了正式复现脚本。

影响：

- `ray stop --force` 可能停止同一用户的其他 Ray 任务。
- `CUDA_LAUNCH_BLOCKING=1` 会显著改变性能表现，不适合作为默认训练配置。
- 硬编码 `/mnt/sdb2/...` 降低迁移性。

潜在解法：

- 改成可覆盖变量：

```bash
RAY_TMPDIR=${RAY_TMPDIR:-"$PWD/ray_tmp"}
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
STOP_RAY_BEFORE_RUN=${STOP_RAY_BEFORE_RUN:-1}
```

- 在脚本启动时打印所有关键环境变量。

### M4. OPD 文档中混入了 GRPO 指标表

位置：

- `reproduction_learning_notes/07_reproduction_log.md:196-199`

问题：

文档在 OPD 章节中写：

```markdown
| `actor/pg_loss` | 0.0 | 0.0 | 0.0 |
| `critic/score/mean` | 0.0 | 0.0 | 0.0 |
```

但实际 OPD 日志 step 1、2、10 分别为非零 KL reward 和非零 pg loss。这个 0 表更像 GRPO 早期稀疏 reward 的记录。

原因：

复现日志中先做 GRPO，再做 OPD，整理文档时把两类实验指标混到同一段。

影响：

- 直接削弱 OPD “token-level reward 有效”的证据链。
- 后续读者可能误判 OPD 前 10 步没有训练信号。

潜在解法：

- 删除该表或移动到 GRPO 章节。
- OPD 章节只保留日志中已核验的 step 1/2/10/1119 数据。
- 在表标题中明确实验名、日志文件名和 step。

### M5. GRPO 文档写 `save_freq=200`，但最终 GRPO 日志实际为 20

位置：

- `reproduction_learning_notes/08_grpo_baseline.md:56`
- `logs/grpo_3090_20260530_170411.log`

问题：

文档表格写 GRPO 将 `save_freq` 从 20 改成 200，但最终日志实际命令包含 `trainer.save_freq=20`。

原因：

可能是脚本后来改过，或文档记录的是目标配置而不是最终运行配置。

影响：

- 磁盘占用评估和复现命令记录不一致。
- 如果用户按最终日志复跑，checkpoint 频率会和文档不同。

潜在解法：

- 文档中区分“计划配置”和“实际运行日志配置”。
- 若当前脚本已是 200，则说明最终 GRPO 实验不是用当前脚本运行的，需要补一句“最终 GRPO 日志来自旧脚本版本”。

### M6. 模型下载记录过期，JustRL 状态前后矛盾

位置：

- `reproduction_learning_notes/07_reproduction_log.md:74-80`
- `reproduction_learning_notes/07_reproduction_log.md:142-143`
- `model/JustRL-DeepSeek-1.5B/README.md`

问题：

前文写 `JustRL-DeepSeek-1.5B` 不可用，可能是私有模型；后文正式 OPD 又使用 `hbx/JustRL-DeepSeek-1.5B`，本地目录也确实存在该模型 README 和权重。

原因：

下载过程分阶段更新，早期失败记录没有改成历史状态。

影响：

- 新复现者可能误以为 teacher 模型无法获取。
- teacher 来源会影响实验可解释性，应准确记录。

潜在解法：

- 改成时间线式记录：
  - 初次尝试：ModelScope/HF 未找到或访问失败。
  - 最终使用：`hbx/JustRL-DeepSeek-1.5B`。
- 记录下载命令、commit/hash 或模型文件列表。

## 低优先级问题

### L1. `seed` 字段实际是 rollout id，不保证随机可复现

位置：

- `scripts/val/eval/gen_vllm.py:159-180`

问题：

代码注释明确“不设置 request-level seeds”，但输出 JSONL 仍写：

```python
"seed": rollout_id
```

原因：

沿用了旧字段名，用于标记第几个 rollout。

影响：

字段名会误导读者以为评测可按 seed 精确复现。

潜在解法：

- 字段改名为 `rollout_id`。
- 若需要可复现，另加真实采样 seed 并确认不会触发 vLLM 性能退化。

### L2. SFT 配置是环境验证，不是论文 SFT 复现

位置：

- `LlamaFactory/examples/train_full/qwen3_base_full_sft_3090.yaml`
- `reproduction_learning_notes/07_reproduction_log.md:87-135`

问题：

SFT 使用 `alpaca_en_demo`、1 个 epoch、999 样本，结果主要证明环境可用，不构成论文中 Qwen3 SFT 数据集复现。

影响：

如果摘要写“1.7B SFT 已完成”，容易被误读为正式 SFT recipe 已复现。

潜在解法：

- 统一写成“LlamaFactory SFT smoke test 已完成”。
- 正式 SFT 另列为未完成或未评估。

### L3. `avg_output_length` tokenizer 口径需要注明

位置：

- `scripts/val/eval/grade.py:45`

问题：

`length_tokenizer` 硬编码为 `/home/chenyizhou/OPD/model/Qwen3-1.7B-Base`，但被评模型是 DeepSeek-R1-Distill-Qwen-1.5B-OPD-final。

影响：

由于两者同属 Qwen2 vocab，误差可能不大；但报告中 `avg_output_length` 不应被视为被评模型 tokenizer 的严格 token 数。

潜在解法：

- 使用被评模型 tokenizer。
- 或将字段改名为 `avg_output_length_by_qwen3_tokenizer`。

### L4. 环境版本未形成可重建锁定

位置：

- `reproduction_learning_notes/07_reproduction_log.md:31-64`
- `.venv/` 为本地环境，未形成 lock/report 文件。

问题：

文档记录了安装步骤，但没有保存完整 `pip freeze`、CUDA 驱动、vLLM、transformers、torch、verl editable commit 等版本清单。

影响：

后续在另一台 3090/4090 上重建环境时，可能因为 vLLM/transformers/torch 版本漂移导致不同错误。

潜在解法：

- 生成并提交或归档：
  - `pip freeze`
  - `python -c "import torch, transformers, vllm; print(...)"`
  - `nvidia-smi`
  - `git rev-parse HEAD`
- 在 README 中写明使用 `.venv/bin/python` 与关键包版本。

## 文档结论需要收敛的地方

### 1. README checklist 仍保留旧评估分数

位置：

- `reproduction_learning_notes/README.md:126`

问题：

README 前文展示的是 `MAX_TOKENS=16384` 的高分，但 checklist 写：

```markdown
- [x] 评估（AIME24: 1.9%, AIME25: 1.3%, AMC23: 23.6%）
```

这是 2048 max tokens 下的旧结果。

建议：

改为同时列出两组：

- 训练同长离线评测 `MAX_TOKENS=2048`：AIME24 1.9%，AIME25 1.3%，AMC23 23.6%。
- 长生成离线评测 `MAX_TOKENS=16384`：AIME24 28.1%，AIME25 24.0%，AMC23 72.2%。

### 2. `resource_estimation.md` 末尾和前文评估表冲突

位置：

- `reproduction_learning_notes/resource_estimation.md:493-520`

问题：

同一文档中已经列出 16384 的评估提升，但末尾仍写“训练效果：AMC23 上 23.6%，AIME 上约 1.5%”和“需要增加到 8192+”。

建议：

将末尾改成“旧结论”或删除，使用 16384 结果作为当前最终离线评估；同时保留 2048 结果作为截断影响分析。

### 3. “与论文预期基本一致”需要更谨慎

位置：

- `reproduction_learning_notes/07_reproduction_log.md:246-256`

问题：

文档用“论文预期”而不是明确论文表格数值和同口径设置，且同时出现 mean_score 与 best_score。best_score 接近 pass@16，不应和 mean accuracy 直接混用。

建议：

改成：

“在本地 1.5B、N=16、MAX_TOKENS=16384、规则评分口径下，AIME24 mean_score=28.1%。该结果显示模型具备较强数学推理能力，但尚未与论文官方设置做严格同口径复现对比。”

## 建议修复顺序

1. 修复 `attention_utils.py` fallback API。
   - 对齐 flash-attn/transformers 返回值。
   - 增加无 flash-attn 环境下的 unit test。
2. 修复 `gen_vllm.py` 的异常处理和完整性校验。
   - 任一 worker 失败即退出。
   - 保存前校验总条数和每题 rollout 数。
3. 固化运行入口。
   - `opd_3090.sh`、`grpo_3090.sh` 使用 `.venv/bin/python` 或 `PYTHON_BIN`。
   - 参数化 `RAY_TMPDIR`、GPU、debug env。
4. 修复 `grade.py --enable_model_verifier`。
   - 保留 prompt。
   - 统一 verifier 输出协议。
   - 参数化 eval dir 和 tokenizer。
5. 整理文档。
   - 移除 OPD 章节中的 GRPO 0 loss 表。
   - 同步 README、resource estimation、reproduction log 的 2048/16384 两套结果。
   - 把 JustRL 下载状态改成最终可复现时间线。
6. 增加环境快照。
   - 保存关键包版本和 GPU/CUDA 信息。
   - 记录最终训练、评测、评分命令。

## 验收建议

修复后至少运行以下检查：

```bash
cd /home/chenyizhou/OPD
bash -n opd_3090.sh grpo_3090.sh
.venv/bin/python -m py_compile scripts/val/eval/gen_vllm.py scripts/val/eval/grade.py
.venv/bin/python - <<'PY'
import torch
from verl.utils.attention_utils import unpad_input, pad_input
x = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
unpadded, indices, cu_seqlens, max_seqlen, *_ = unpad_input(x, mask)
assert cu_seqlens.tolist() == [0, 2, 4]
assert max_seqlen == 2
assert torch.equal(pad_input(unpadded, indices, 2, 4)[mask.bool()], x[mask.bool()])
print("attention fallback smoke test passed")
PY
```

评测脚本修复后，应额外跑一个小规模 smoke test，例如每个任务只取 1-2 个样本、`N=2`，并断言输出行数等于 `样本数 * N`。

## 总体判断

这次复现的核心训练和离线评测产物是存在的，不能简单判为“没跑通”。更准确的结论是：当前项目完成了一次 8x3090 上的 1.5B OPD 可行性复现，但工程化复现入口、flash-attn 替代实现、评测完整性检查和文档证据链仍需要修正。特别是 flash-attn fallback 和评测静默失败这两类问题，属于会影响后续复现实验可靠性的实质缺陷，应优先处理。
