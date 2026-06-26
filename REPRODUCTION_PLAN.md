# OPD 完整全流程复现计划 (8×RTX 3090)

> 基于对项目代码、文档、已有复现日志的深度分析  
> 硬件约束：8×NVIDIA RTX 3090 24GB，系统内存 ~251GB  
> 目标：在 3090 上完成论文核心流程的可信复现，并修复已知工程问题

---

## 一、现状评估

### 已完成

| 阶段 | 状态 | 产物 | 问题 |
|------|------|------|------|
| 环境搭建 | ✅ | `.venv` Python 3.12 | 裸 `python3` 是 3.5.4，脚本未验证环境 |
| 模型下载 | ✅ | DS-R1-1.5B, JustRL-1.5B, Qwen3-1.7B-Base | JustRL 下载记录前后矛盾 |
| SFT 验证 | ✅ smoke test | alpaca_en_demo 999 样本 | 非论文正式 SFT 复现 |
| GRPO baseline | ⚠️ 131 步中断 | 日志存在 | 未完成，信号稀疏 |
| OPD 训练 | ✅ 1119 步 | checkpoint + 日志 | `MAX_RESP_LENGTH=2048` 偏短 |
| 离线评估 | ✅ | AIME24/25/AMC23 JSONL | 训练/评测长度口径不一致 |
| 代码审查 | ✅ | `REPRODUCTION_REVIEW.md` | 14 个问题待修复 |

### 关键已知问题（来自 `REPRODUCTION_REVIEW.md`）

1. **H1**: `attention_utils.py` flash-attn fallback API 不兼容（只返回 3 值，标准返回 5 值）
2. **H2**: `gen_vllm.py` 会吞掉 worker 异常，可能静默保存不完整结果
3. **H3**: 脚本依赖裸 `python3`（3.5.4），未验证 `.venv` 环境
4. **H4**: 训练 `MAX_RESP_LENGTH=2048` vs 评测 `MAX_TOKENS=16384` 口径不一致
5. **M1**: `grade.py --enable_model_verifier` 分支 prompt 丢失
6. **M2**: 评测脚本硬编码模型/GPU/max_tokens
7. **M3**: 脚本硬编码 `RAY_TMPDIR`、`CUDA_LAUNCH_BLOCKING=1`

---

## 二、两条复现路径选择

### 路径 A：基于已有 OPD checkpoint 继续（推荐）

**优势**：已有 1119 步训练产物，只需修复工程问题 + 补充实验  
**劣势**：`MAX_RESP_LENGTH=2048` 偏短，与论文 `7168` 有差距

### 路径 B：使用 verl 新 distillation API 重新训练

**背景**：`verl_example/opd.sh` 使用了 verl 新增的 `distillation.*` 配置 API，与旧 `on_policy_distillation.sh`（通过 `reward_model.enable=True` 走 reward model worker 路径）架构不同。

**优势**：更干净的 API、更好的可维护性  
**劣势**：需要验证新 API 在 3090 上的兼容性，可能有新 bug

### 推荐：路径 A 为主，路径 B 作为探索

---

## 三、完整复现流程（路径 A）

### Phase 0: 环境修复与工程问题修复（~2 小时）

#### 0.1 修复 Python 环境入口

```bash
# 在所有脚本顶部加入
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 12)'
```

将 `opd_3090.sh`、`grpo_3090.sh` 中的 `python3` 替换为 `"$PYTHON_BIN"`。

#### 0.2 修复 flash-attn fallback API

**文件**：`verl/verl/utils/attention_utils.py`

当前问题：`unpad_input` fallback 只返回 3 值，标准接口返回 5 值。

**修复方案**：使用 transformers 已提供的 fallback：
```python
from transformers.modeling_flash_attention_utils import _unpad_input, _pad_input, _index_first_axis
```

或参考 `verl/verl/utils/npu_utils.py:99-128` 补全返回值。

**验证**：
```bash
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

#### 0.3 修复评测脚本完整性校验

**文件**：`scripts/val/eval/gen_vllm.py`

修复方案：
- worker 异常重新抛出，主进程收到任一失败 future 后 `raise SystemExit(1)`
- 保存前断言 `len(all_results) == len(samples) * N`
- 每个 `example_id` 的响应数等于 N

#### 0.4 参数化评测脚本

**文件**：`scripts/val/eval/gen_vllm.py`、`scripts/val/eval/grade.py`

增加 CLI 参数：`--model`, `--tasks`, `--gpu-ids`, `--max-tokens`, `--gpu-memory-utilization`

#### 0.5 清理脚本中的调试变量

- `CUDA_LAUNCH_BLOCKING=1` → `CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}`
- `RAY_TMPDIR=/mnt/sdb2/...` → `RAY_TMPDIR=${RAY_TMPDIR:-"$PWD/ray_tmp"}`
- `ray stop --force` → 条件化 `STOP_RAY_BEFORE_RUN=${STOP_RAY_BEFORE_RUN:-1}`

---

### Phase 1: 环境快照与基线验证（~1 小时）

#### 1.1 保存环境快照

```bash
cd /home/chenyizhou/OPD
source .venv/bin/activate

# 保存环境信息
{
  echo "=== Python ==="
  python --version
  echo "=== PyTorch ==="
  python -c "import torch; print(torch.__version__); print('CUDA:', torch.version.cuda)"
  echo "=== vLLM ==="
  python -c "import vllm; print(vllm.__version__)"
  echo "=== transformers ==="
  python -c "import transformers; print(transformers.__version__)"
  echo "=== verl ==="
  cd verl && git rev-parse HEAD && git status --short && cd ..
  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "=== CUDA ==="
  nvcc --version
} > reproduction_learning_notes/environment_snapshot.txt

pip freeze > reproduction_learning_notes/requirements_freeze.txt
```

#### 1.2 验证已有 OPD checkpoint

```bash
# 检查 checkpoint 完整性
ls -la checkpoint/token_reward_direct_DAPO-Math-17k_*/global_step_1119/actor/
cat checkpoint/token_reward_direct_DAPO-Math-17k_*/latest_checkpointed_iteration.txt
# 应输出: 1119
```

#### 1.3 验证已有评估产物

```bash
# 检查 JSONL 行数
wc -l scripts/val/eval/justrl_eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B-OPD-final/*.jsonl
# AIME24: 480, AIME25: 480, AMC23: 640

# 检查 grading_results.json
cat scripts/val/eval/justrl_eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B-OPD-final/grading_results.json
```

---

### Phase 2: 补充 GRPO Baseline 完整训练（~15 小时）

**目的**：与 OPD 做同口径对比，验证 dense reward vs sparse reward 的差异

#### 2.1 修复 grpo_3090.sh

基于已有的 `grpo_3090.sh`，修复以下问题：

```bash
# 修复 Python 入口
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}

# 参数化调试变量
RAY_TMPDIR=${RAY_TMPDIR:-"$PWD/ray_tmp"}
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

# 确保 enable_thinking=False（非 thinking 模型）
+data.apply_chat_template_kwargs.enable_thinking=False
```

#### 2.2 GRPO 训练配置

```bash
# 关键配置（基于 grpo_3090.sh）
ADV_ESTIMATOR=grpo
LOG_PROB_TOP_K=0              # GRPO 不需要 top-K
N_RESPONSES=2                 # 3090 适配
MAX_RESP_LENGTH=2048          # 3090 适配
MINI_BATCH_SIZE=16
MODEL_DTYPE=bfloat16
gpu_memory_utilization=0.5
param_offload=True
optimizer_offload=True
reward_model.enable=False     # 使用规则奖励
```

#### 2.3 预期结果

- 训练步数：~1119 步（与 OPD 同口径）
- 每步时间：~45-50 秒
- 总时长：~14 小时
- 预期：GRPO 早期 signal 稀疏（大部分 step `actor/pg_loss=0`），后期可能改善

---

### Phase 3: OPD 超参 Ablation（~50 小时）

基于已有 1119 步 OPD checkpoint，做以下 ablation：

#### 3.1 Top-K ablation

| 实验 | `LOG_PROB_TOP_K` | 预计时间 | 目的 |
|------|-------------------|----------|------|
| OPD-K0 | 0 (sampled-token) | ~14h | 退化为 sampled-token OPD |
| OPD-K8 | 8 | ~14h | 更激进的近似 |
| OPD-K16 | 16 | 已完成 | baseline |
| OPD-K32 | 32 | ~14h | 更精确的近似 |

#### 3.2 Top-K Strategy ablation

| 实验 | `TOP_K_STRATEGY` | 预计时间 | 目的 |
|------|-------------------|----------|------|
| OPD-only-stu | only_stu | 已完成 | baseline |
| OPD-only-tch | only_tch | ~14h | teacher-driven |
| OPD-union | union | ~14h | 最完整覆盖 |
| OPD-intersection | intersection | ~14h | 保守策略 |

#### 3.3 Teacher Temperature ablation

| 实验 | `TEACHER_TEMPERATURE` | 预计时间 | 目的 |
|------|------------------------|----------|------|
| OPD-T0.7 | 0.7 | ~14h | 锐化 teacher |
| OPD-T1.0 | 1.0 | 已完成 | baseline |
| OPD-T1.3 | 1.3 | ~14h | 软化 teacher |

#### 3.4 N_RESPONSES ablation

| 实验 | `N_RESPONSES` | 预计时间 | 目的 |
|------|---------------|----------|------|
| OPD-N2 | 2 | 已完成 | baseline |
| OPD-N4 | 4 | ~14h | 更多样本 |

#### 3.5 MAX_RESP_LENGTH ablation

| 实验 | `MAX_RESP_LENGTH` | 预计时间 | 目的 |
|------|--------------------|----------|------|
| OPD-2048 | 2048 | 已完成 | baseline |
| OPD-4096 | 4096 | ~14h | 更长推理 |

**注意**：3090 上 `MAX_RESP_LENGTH=4096` 需要进一步降低 `gpu_memory_utilization` 或 `N_RESPONSES`。

---

### Phase 4: 统一评估口径（~10 小时）

#### 4.1 评估所有 checkpoint

对 Phase 2-3 产生的所有 checkpoint 做统一评估：

```bash
# 评估配置（统一口径）
MAX_TOKENS=16384    # 足够长，避免截断
N=16                # 16 rollouts per prompt
temperature=0.7
top_p=0.95
```

#### 4.2 评估任务

- AIME24 (30 题)
- AIME25 (30 题)
- AMC23 (40 题)
- MATH-500 (可选)
- GPQA (可选)

#### 4.3 评估指标

- `mean_score`：所有 rollout 平均正确率（主要指标）
- `best_score`：每题 N 个 rollout 至少一个正确（pass@N）
- `solve_all`：N 个 rollout 全部正确的题数
- `format_error_rollouts`：缺少 boxed 答案的 rollout 数

---

### Phase 5: 文档整理与证据链收敛（~3 小时）

#### 5.1 修复文档问题

1. 移除 `07_reproduction_log.md` 中 OPD 章节的 GRPO 0 loss 表
2. 同步 README、resource estimation 的 2048/16384 两套结果
3. 把 JustRL 下载状态改成最终可复现时间线
4. 区分"训练设置"、"verl 内置验证设置"、"离线评测设置"

#### 5.2 生成复现报告

包含：
- 环境快照
- 训练配置对比表（原始 A800 vs 3090 适配）
- 训练曲线（pg_loss, overlap_ratio, entropy）
- 评估结果对比表（所有 ablation）
- 与论文对比（同口径 vs 不同口径）
- 已知问题与修复方案

---

## 四、时间估算

| Phase | 内容 | 预计时间 | 依赖 |
|-------|------|----------|------|
| 0 | 环境修复 | ~2h | 无 |
| 1 | 环境快照 + 基线验证 | ~1h | Phase 0 |
| 2 | GRPO baseline 完整训练 | ~15h | Phase 1 |
| 3 | OPD ablation | ~50h | Phase 1 |
| 4 | 统一评估 | ~10h | Phase 2, 3 |
| 5 | 文档整理 | ~3h | Phase 4 |
| **总计** | | **~81h** | |

**并行化**：Phase 2 和 Phase 3 可以串行运行（共用 GPU），总时间 ~65h。

---

## 五、关键风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 3090 OOM | 训练中断 | 降低 `N_RESPONSES`、`MAX_RESP_LENGTH`、`gpu_memory_utilization` |
| Ray /tmp 磁盘满 | 训练中断 | 设置 `RAY_TMPDIR` 到大容量磁盘，定期清理 |
| flash-attn fallback bug | Megatron 路径崩溃 | Phase 0 修复 |
| 评测脚本静默失败 | 指标失真 | Phase 0 修复完整性校验 |
| GRPO 训练不收敛 | 无法对比 | 增加训练步数或调整 lr |
| checkpoint 磁盘占用 | 磁盘满 | `save_freq=200`，定期清理中间 checkpoint |

---

## 六、验收标准

### 最小验收（必须完成）

- [x] OPD 1.5B 训练 1119 步
- [x] 评估 AIME24/25/AMC23
- [ ] 修复 flash-attn fallback API
- [ ] 修复评测脚本完整性校验
- [ ] 修复 Python 环境入口
- [ ] GRPO baseline 完整训练（同口径对比）
- [ ] 环境快照保存

### 完整验收（尽量完成）

- [ ] Top-K ablation (K=0/8/32)
- [ ] Strategy ablation (only_tch/union/intersection)
- [ ] Teacher temperature ablation (0.7/1.3)
- [ ] 统一评估口径（所有 checkpoint 用 MAX_TOKENS=16384）
- [ ] 复现报告（含与论文对比）
- [ ] 文档证据链收敛

### 超额验收（有时间再做）

- [ ] 使用 verl 新 distillation API 重新训练
- [ ] SFT → OPD 两阶段复现
- [ ] 更大数据集（DeepMath-103K）实验
- [ ] MATH-500/GPQA 评估

---

## 七、核心代码路径速查

| 模块 | 文件 | 关键行 |
|------|------|--------|
| Token-level advantage | `verl/verl/trainer/ppo/core_algos.py` | 854-880 |
| GRPO advantage | `verl/verl/trainer/ppo/core_algos.py` | 265-328 |
| Hybrid advantage | `verl/verl/trainer/ppo/core_algos.py` | 883-924 |
| Policy loss (PPO) | `verl/verl/trainer/ppo/core_algos.py` | 1058-1197 |
| Loss aggregation | `verl/verl/trainer/ppo/core_algos.py` | 942-978 |
| KL penalty | `verl/verl/trainer/ppo/core_algos.py` | 1633-1694 |
| Student top-K | `verl/verl/workers/actor/dp_actor.py` | 203-271 |
| Distillation reward | `verl/verl/workers/actor/dp_actor.py` | 450-624 |
| Actor update | `verl/verl/workers/actor/dp_actor.py` | 806-931 |
| Teacher top-K | `verl/verl/workers/fsdp_workers.py` | 1830-1917 |
| Teacher temperature | `verl/verl/workers/fsdp_workers.py` | 2015 |
| Training loop | `verl/verl/trainer/ppo/ray_trainer.py` | 967-2353 |
| RL data loading | `verl/verl/utils/dataset/rl_dataset.py` | 152-377 |
| Evaluation grading | `scripts/val/eval/utils.py` | 485-494 |
| Teacher rollout | `scripts/infer/vllm_rollout.py` | 384-523 |
| Data dedup | `scripts/infer/dedup_deepmath.py` | 36-109 |

---

## 八、关键配置参数速查

| 参数 | 原始值 (A800) | 3090 适配值 | 说明 |
|------|---------------|-------------|------|
| `MODEL_DTYPE` | fp32 | bfloat16 | rollout 权重减半 |
| `MAX_RESP_LENGTH` | 7168 | 2048 | 减少 KV cache |
| `N_RESPONSES` | 4 | 2 | 减少 rollout 显存 |
| `MINI_BATCH_SIZE` | 64 | 16 | 匹配更短序列 |
| `gpu_memory_utilization` | 0.8 | 0.4 | 留出 FSDP 空间 |
| `param_offload` | False | True | 参数卸载到 CPU |
| `optimizer_offload` | False | True | 优化器卸载到 CPU |
| `save_freq` | 20 | 200 | 减少磁盘占用 |
| `flash_attn` | flash_attention_2 | sdpa | flash-attn 未安装 |
| `trainer.test_freq` | -1 | -1 | 禁用内置验证 |

---

## 九、实测性能数据（已有）

### OPD 训练（1119 步）

| 指标 | 值 |
|------|-----|
| 总时间 | 13h44m |
| 每步时间 | ~45s |
| 显存分配 | 21-22 GB |
| 显存预留 | 26.5 GB |
| CPU 内存 | ~230 GB |
| 吞吐量 | ~181-197 tokens/s |

### 训练信号

| Step | pg_loss | score/mean | overlap_ratio |
|------|---------|------------|---------------|
| 1 | 0.2616 | -0.2616 | 0.7223 |
| 2 | 0.2745 | -0.2731 | 0.7319 |
| 10 | 0.2738 | -0.2614 | 0.7240 |
| 1119 | 0.2205 | -0.2040 | 0.7625 |

### 评估结果（MAX_TOKENS=16384）

| 任务 | mean_score | best_score | solve_all | format_error |
|------|------------|------------|-----------|--------------|
| AMC23 | 72.2% | 97.5% | 16/40 | 114/640 (18%) |
| AIME24 | 28.1% | 63.3% | 1/30 | 206/480 (43%) |
| AIME25 | 24.0% | 46.7% | 1/30 | 199/480 (41%) |

---

## 十、下一步行动

1. **立即执行**：Phase 0（修复工程问题）
2. **短期**：Phase 1（环境快照）+ Phase 2（GRPO baseline）
3. **中期**：Phase 3（OPD ablation）+ Phase 4（统一评估）
4. **长期**：Phase 5（文档整理）+ 超额验收

---

*最后更新：2026-06-22*
*基于对 `/home/chenyizhou/OPD` 项目的深度代码分析与文档审查*
