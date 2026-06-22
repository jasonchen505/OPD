# LLM & Agent 后训练面试深度准备手册

> 基于 OPD (On-Policy Distillation) 项目复现，面向 LLM 算法实习面试  
> 项目路径：`/home/chenyizhou/OPD`  
> 最后更新：2026-06-22

---

## 目录

- [第一部分：项目概述与核心理解](#第一部分项目概述与核心理解)
- [第二部分：verl 框架深度解析](#第二部分verl-框架深度解析)
- [第三部分：后训练算法原理与对比](#第三部分后训练算法原理与对比)
- [第四部分：OPD 核心实现细节](#第四部分opd-核心实现细节)
- [第五部分：分布式训练与系统工程](#第五部分分布式训练与系统工程)
- [第六部分：Agent 与多轮对话训练](#第六部分agent-与多轮对话训练)
- [第七部分：评估体系与方法论](#第七部分评估体系与方法论)
- [第八部分：高频面试问题深度解析](#第八部分高频面试问题深度解析)
- [第九部分：业务场景与工程落地](#第九部分业务场景与工程落地)
- [第十部分：面试官视角的能力考察点](#第十部分面试官视角的能力考察点)

---

## 第一部分：项目概述与核心理解

### 1.1 项目背景

**论文**：*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* (arXiv:2604.13016)

**核心问题**：LLM 后训练中，传统蒸馏存在 distribution mismatch，而 GRPO 等 outcome-level RL 在数学推理早期 reward 稀疏。

**OPD 的核心思想**：
- 让 student 自己 rollout（on-policy）
- 在 student 访问到的状态上，用 teacher 的 token-level KL 信号做 dense reward
- 用 PPO 风格的 policy update 更新 student

### 1.2 一句话总结（面试用）

> OPD 把蒸馏放到 student 的 on-policy 分布上，用 teacher 的 token-level 分布替代稀疏的 outcome reward，解决了传统 KD 的 distribution mismatch 和 GRPO 的 reward 稀疏问题。

### 1.3 项目实际完成的工作

| 维度 | 内容 |
|------|------|
| 算法 | OPD (token_reward_direct) + GRPO baseline |
| 框架 | verl v0.7.0 + LlamaFactory |
| 模型 | Student: DeepSeek-R1-Distill-Qwen-1.5B, Teacher: JustRL-DeepSeek-1.5B |
| 数据 | DAPO-Math-17k (数学推理) |
| 硬件 | 8x RTX 3090 24GB（原论文 8x A800 80GB） |
| 训练 | 1119 步 OPD 训练完成 |
| 评估 | AIME24/AIME25/AMC23, N=16 rollouts |

### 1.4 关键实验数据（必须背诵）

```
硬件：8x RTX 3090 24GB
Student：DeepSeek-R1-Distill-Qwen-1.5B
Teacher：JustRL-DeepSeek-1.5B
训练步数：1119 steps
Top-K：16, Strategy: only_stu, Weight: student_p
训练 response length：2048
N responses：2

训练信号：
- step 1:  actor/pg_loss=0.2616, critic/score/mean=-0.2616, topk/overlap_ratio=0.7223
- step 1119: actor/pg_loss=0.2205, critic/score/mean=-0.2040, topk/overlap_ratio=0.7625

离线评估（MAX_TOKENS=16384）：
- AIME24: mean_score=28.1%, best_score=63.3%
- AIME25: mean_score=24.0%, best_score=46.7%
- AMC23: mean_score=72.2%, best_score=97.5%
```

---

## 第二部分：verl 框架深度解析

### 2.1 verl 整体架构

verl 是 ByteDance 开源的 RL 训练框架，核心是 **Hybrid Engine** 设计：

```
RayPPOTrainer (主控进程)
  |
  +-- ActorRolloutRefWorker (colocate 在同一组 GPU)
  |     +-- Actor: FSDP 包装的 student，有 optimizer
  |     +-- Rollout: vLLM 实例，高吞吐生成
  |     +-- Ref: FSDP reference policy，用于 KL
  |
  +-- RewardModelWorker (OPD 中作为 Teacher)
        +-- Teacher model: FSDP CausalLM
        +-- 计算 token-level log-probs
```

**关键代码位置**：
- 训练主循环：`verl/verl/trainer/ppo/ray_trainer.py:967` (`fit()`)
- Actor worker：`verl/verl/workers/actor/dp_actor.py`
- FSDP workers：`verl/verl/workers/fsdp_workers.py`
- 核心算法：`verl/verl/trainer/ppo/core_algos.py`

### 2.2 Colocation 设计（重要考点）

**为什么 Actor/Rollout/Ref 共享 GPU？**

```python
# fsdp_workers.py 中的 context switching
def rollout_mode():
    # gather FSDP weights → 推送到 vLLM → offload
    pass

def trainer_mode():
    # 释放 vLLM KV cache → 设置模型为 .train()
    pass
```

**优势**：
- 最小化 GPU 显存开销
- 避免跨 GPU 通信
- 权重原地传输

**代价**：
- context switching 有开销
- 显存峰值要兼顾 vLLM KV cache 和 FSDP 训练

**面试回答模板**：
> verl 的 hybrid engine 把 Actor/Rollout/Ref colocate 在同一组 GPU 上。rollout 时 gather 权重给 vLLM，training 时释放 KV cache 回到 FSDP。这样同一组 GPU 可以交替做生成和训练，避免两套 GPU 的浪费。代价是 context switching 有开销，显存峰值要兼顾两边。

### 2.3 训练一步的数据流

```text
1. generate_sequences()     → vLLM 生成 N 个 responses
2. compute_log_prob()       → FSDP actor 计算 old_log_probs + student top-K
3. compute_rm_score()       → teacher forward 计算 teacher log-probs
4. compute_distillation_reward() → 计算 KL reward
5. compute_advantage()      → token_reward_direct: adv = reward * mask
6. update_actor()           → PPO clipped loss 更新 student
```

**代码位置**：`ray_trainer.py:1049-1269`

### 2.4 FSDP 配置与显存管理

```python
# 关键配置（opd_3090.sh）
actor_rollout_ref.actor.fsdp_config.param_offload=True      # 参数卸载到 CPU
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True   # 优化器卸载到 CPU
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16     # 使用 bf16
actor_rollout_ref.rollout.gpu_memory_utilization=0.4         # vLLM KV cache 预算
```

**显存分配分析**：

| 阶段 | Allocated | Reserved | 用途 |
|------|-----------|----------|------|
| Rollout | ~15 GB | ~20 GB | vLLM 模型权重 + KV cache |
| Training | ~18.6 GB | ~27.6 GB | FSDP 模型 + 优化器 + 激活值 |

**3090 适配策略**：
1. bf16：降低权重和激活显存
2. param/optimizer offload：用 CPU 内存换 GPU 显存
3. gradient checkpointing/activation offload：用计算换激活显存
4. 降低 response length（2048）和 N responses（2）
5. 降低 vLLM gpu_memory_utilization（0.4）

### 2.5 Ray 分布式调度

```bash
ray start --head  # 启动 Ray 集群
# verl 通过 Ray 管理 worker 生命周期和 RPC 调用
```

**关键概念**：
- WorkerGroup：管理一组同类 worker
- Dispatch Mode：`ONE_TO_ALL`（广播）、`MESH`（分片）等
- DataProto：verl 的数据传输协议，包含 batch tensors 和 meta_info

---

## 第三部分：后训练算法原理与对比

### 3.1 四种后训练方法对比表（面试必备）

| 方法 | 数据来源 | Reward/Loss | On-Policy | 信号密度 | 优点 | 风险 |
|------|----------|-------------|-----------|----------|------|------|
| SFT | 固定示范数据 | CE loss | 否 | token-level | 稳定、便宜 | distribution mismatch |
| DPO | preference pair | preference obj | 否 | pair-level | 不需在线 rollout | 依赖偏好数据质量 |
| GRPO | student rollout | outcome reward | 是 | response-level | 对最终目标直接 | 早期 reward 稀疏 |
| OPD | student rollout + teacher | token-level KL | 是 | token-level | 每个 token 有信号 | teacher 兼容性和成本 |

### 3.2 GRPO 原理

**核心思想**：Group Relative Policy Optimization，组内相对优势估计。

```python
# core_algos.py:265-328
@register_adv_est("grpo")
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)  # 每个 response 的总 reward
    # 组内标准化
    for idx in id2score:
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

**GRPO 的问题**：
- 数学题初始正确率低，很多 batch reward 全 0
- 组内标准化后优势信号很弱
- 早期训练 policy gradient 几乎为 0

**实测数据**：
```
GRPO step 4:  actor/pg_loss=0.00497, critic/score/mean=0.03125（32 个样本约 1 个正确）
GRPO step 10: actor/pg_loss=0.0, critic/score/mean=0.0（全错）
```

### 3.3 OPD 原理

**核心公式**：
```
r_t = -KL(π_student || π_teacher) at position t
```

**Top-K 近似**：
```
KL_t ≈ Σ_{a ∈ K_t} w(a) * [log π_student(a|s_t) - log π_teacher(a|s_t)]
reward_t = -KL_t
```

**为什么用 reverse KL？**
- `KL(π_student || π_teacher)` 是 mode-seeking
- 惩罚 student 在 teacher 低概率 token 上放太多概率
- 更适合 on-policy 修正 student 当前错误分布

**为什么用 top-K？**
- 完整词表 KL 太贵（vocab 10 万+）
- 高概率 token 集承载 97-99% 概率质量
- K=16 是精度与效率的 trade-off

### 3.4 OPD vs GRPO 训练信号对比

```
OPD step 1:   actor/pg_loss=0.2616（非零，dense signal）
GRPO step 1:  actor/pg_loss=0.0（零，稀疏 signal）

OPD step 10:  actor/pg_loss=0.2738（稳定）
GRPO step 10: actor/pg_loss=0.0（仍然稀疏）
```

**面试结论**：
> OPD 的 dense reward 解决了 GRPO 早期信号稀疏的问题，但公平比较需要同模型、同步数、同评测口径的完整实验。

### 3.5 PPO Clipped Loss

```python
# core_algos.py:1058-1197
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, ...):
    ratio = torch.exp(log_prob - old_log_prob)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1-cliprange, 1+cliprange)
    pg_losses = torch.where(advantages < 0,
                            torch.min(pg_losses3, clip_pg_losses1),  # dual-clip
                            torch.max(pg_losses1, pg_losses2))       # standard clip
```

**PPO 在 OPD 中的作用**：
- 即使 reward 来自 KL，policy update 仍需要约束
- PPO clipping 限制新旧策略变化
- 避免一次更新过大导致训练不稳定

### 3.6 Advantage Estimator 对比

| Estimator | 代码位置 | 适用场景 | 特点 |
|-----------|----------|----------|------|
| `gae` | core_algos.py:200 | 标准 PPO | 需要 value function |
| `grpo` | core_algos.py:265 | GRPO | 组内标准化，outcome reward |
| `token_reward_direct` | core_algos.py:855 | OPD | 直接用 token reward，不需要 baseline |
| `token_reward_direct_plus_grpo` | core_algos.py:883 | OPD+GRPO 混合 | dense + sparse 结合 |

---

## 第四部分：OPD 核心实现细节

### 4.1 Top-K 策略实现

**Student Top-K 计算**（`dp_actor.py:86-189`）：
```python
def _forward_micro_batch(self, micro_batch, temperature, top_k=0, student_top_k_ids=None):
    output = self.actor_module(input_ids=input_ids_rmpad, ...)
    logits = output.logits / temperature
    topk_logits, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_log_probs = topk_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return entropy, log_probs, topk_ids, topk_log_probs
```

**Teacher Top-K 计算**（`fsdp_workers.py:1831-1918`）：
```python
def _compute_teacher_top_k_log_probs(self, logits, student_ids, top_k, strategy="only_stu"):
    t_logits, t_ids = torch.topk(logits_chunk, k=top_k, dim=-1)
    t_logsumexp = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
    t_log_probs_top_k = t_logits - t_logsumexp

    # 计算 student token 在 teacher 中的 log-prob
    chunk_log_probs = torch.gather(logits_chunk, dim=-1, index=student_ids_chunk) - t_logsumexp

    # 计算 overlap mask
    matches = (s_ids_exp == t_ids_exp)
    is_in_teacher = matches.any(dim=-1)
```

### 4.2 Top-K 策略对比

| 策略 | 代码实现 | KL 来源 | 适用场景 |
|------|----------|---------|----------|
| `only_stu` | dp_actor.py:560-564 | `S_logp - T_on_S` | 默认，学生驱动 |
| `only_tch` | dp_actor.py:566-571 | `S_on_T - T_logp` | 教师驱动 |
| `intersection` | dp_actor.py:573-578 | 仅重叠 token | 保守 |
| `union` | dp_actor.py:580-599 | 连接 S & T | 最完整 |
| `union-intersection` | dp_actor.py:601-621 | 对称差 | 聚焦分歧 |

### 4.3 Reward Weight 计算

```python
# dp_actor.py:522-556
def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
    if weight_mode == "student_p":
        log_probs = S_logp  # 按学生信念加权
    elif weight_mode == "teacher_p":
        log_probs = T_logp  # 按教师信念加权
    elif weight_mode == "none":
        log_probs = torch.zeros_like(S_logp)  # 均匀权重

    if normalize:
        norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        weights = torch.exp(norm_log_weights)
```

**为什么用 `student_p`？**
- student 高置信的 token 对 KL reward 贡献更大
- 如果 student 把高概率放错了，teacher 能强烈纠正
- 低概率 token 影响较小，减少噪声

### 4.4 Distillation Reward 计算

```python
# dp_actor.py:451-624
def compute_distillation_reward(self, data: DataProto) -> DataProto:
    # only_stu 策略
    if strategy == "only_stu":
        kl_val = S_logp - T_on_S  # 每 token 每 top-K 的反向 KL
        norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
        rm_scores = -kl_val * norm_weights  # 负 KL = 对齐奖励
```

### 4.5 Token Reward Direct Advantage

```python
# core_algos.py:855-880
@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(token_level_rewards, response_mask, ...):
    # 最简单的 advantage estimator
    # 不需要 gamma、lambda、value baseline
    advantages = token_level_rewards * response_mask
    returns = advantages.clone()
    return advantages, returns
```

**为什么不需要 critic？**
- reward 本身已经是每个 token 的 dense teacher signal
- 不需要 value function 估计 baseline
- verl 框架保留了 critic 组件，但 OPD 这条路径不依赖它

### 4.6 Teacher Temperature

```python
# fsdp_workers.py:2015
logits_rmpad = logits_rmpad.div_(teacher_temperature)
```

**效果**：
- T < 1：锐化教师分布（更自信，信号更强）
- T > 1：软化教师分布（更均匀，更宽容）
- T = 1.0：无缩放（默认）

### 4.7 PPO 3D Top-K 处理

```python
# core_algos.py:1118-1155
if log_prob.dim() == 3 and old_log_prob.dim() == 3:
    # Top-K 情况下 advantage 是 (batch, seq_len, K)
    # 使用 memory-efficient formulation
    # ∇L ≈ -A × ∇log π = ∇(-A × log π)
    # 所以 L = -Σ_x [A(x) × log π_θ(x)]
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses = torch.sum(pg_losses, dim=-1)  # sum across K tokens
```

---

## 第五部分：分布式训练与系统工程

### 5.1 FSDP 原理

**FSDP (Fully Sharded Data Parallel)**：
- 参数、梯度、优化器状态分片到多卡
- 计算时 all-gather 收集完整参数
- 计算后 reduce-scatter 更新分片

**关键配置**：
```python
param_offload=True      # 参数卸载到 CPU
optimizer_offload=True  # 优化器状态卸载到 CPU
forward_prefetch=True   # 预取下一层参数
model_dtype=bfloat16    # 使用 bf16
```

### 5.2 vLLM 集成

**vLLM 在 verl 中的角色**：
- 高吞吐生成 responses
- PagedAttention 管理 KV cache
- 与 FSDP actor 共享 GPU

**关键配置**：
```python
gpu_memory_utilization=0.4  # KV cache 预算
max_model_len=3072          # 最大序列长度
tensor_model_parallel_size=1 # 张量并行
```

**面试要点**：
> vLLM 的 `gpu_memory_utilization` 控制 KV cache 预算，不是模型权重。模型权重由 FSDP 管理，vLLM 只负责推理时的 KV cache 和调度。

### 5.3 显存优化策略

| 策略 | 效果 | 代价 |
|------|------|------|
| bf16 | 权重/激活减半 | 精度略有损失 |
| param offload | 参数→CPU | PCIe 通信开销 |
| optimizer offload | 优化器→CPU | PCIe 通信开销 |
| gradient checkpointing | 重计算换显存 | 计算时间增加 |
| activation offload | 激活→CPU | CPU-GPU 搬运 |
| 降低 response length | 减少 KV cache | 训练信号减少 |
| 降低 N responses | 减少 batch tokens | 方差增大 |
| 降低 gpu_memory_utilization | 留空间给训练 | 推理吞吐降低 |

### 5.4 3090 适配经验

**原始配置**（8x A800 80GB）：
```bash
MAX_RESP_LENGTH=7168
N_RESPONSES=4
MODEL_DTYPE=fp32
gpu_memory_utilization=0.8
param_offload=False
optimizer_offload=False
```

**3090 适配**（8x RTX 3090 24GB）：
```bash
MAX_RESP_LENGTH=2048
N_RESPONSES=2
MODEL_DTYPE=bfloat16
gpu_memory_utilization=0.4
param_offload=True
optimizer_offload=True
```

**适配逻辑**：
1. bf16：降低 rollout 权重和激活显存
2. MAX_RESP_LENGTH=2048：减少 KV cache 和每步 token 数
3. N_RESPONSES=2：减少每个 prompt 的 rollout 数
4. gpu_memory_utilization=0.4：给 FSDP 和 reward model 留显存
5. param/optimizer offload：用 CPU 内存换 GPU 显存

### 5.5 Context Switching 开销

```python
# fsdp_workers.py:653-752
def rollout_mode():
    # gather FSDP weights → 推送到 vLLM
    # 时间：取决于模型大小和 GPU 数量

def trainer_mode():
    # 释放 vLLM KV cache → 设置模型为 .train()
    # 时间：KV cache 释放 + FSDP 设置
```

**实测观察**：
- rollout 和 training 交替进行
- 每步有固定的 context switching 开销
- 但比两套 GPU 的方案更节省显存

---

## 第六部分：Agent 与多轮对话训练

### 6.1 verl 中的 Agent 支持

verl 框架支持多种 Agent 训练场景：

```
verl/recipe/
├── langgraph_agent/     # LangGraph React Agent
├── collabllm/           # CollabLLM 多轮交互
├── retool/              # 工具调用 RL
├── sglang_multiturn/    # SGLang 多轮对话
└── flowrl/              # FlowRL
```

### 6.2 LangGraph React Agent

```python
# verl/recipe/langgraph_agent/react_agent_loop.py
async def call_model(state: MessagesState, config: RunnableConfig):
    model = config["configurable"]["model"]
    message = await model.ainvoke(state["messages"], sampling_params=sampling_params)
    return {"messages": [message]}

def should_continue(state: MessagesState, config: RunnableConfig) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END  # 没有工具调用，结束
    return "tools"  # 继续调用工具
```

**Agent Loop 设计**：
- 基于 LangGraph 的状态机
- 支持多轮工具调用
- 可配置最大轮次防止无限循环

### 6.3 CollabLLM

```python
# verl/recipe/collabllm/collabllm_agent_loop.py
class CollabLLMAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        # 1. 首先生成模型响应
        await self._handle_pending_state(agent_data, sampling_params)
        # 2. 收集交互 rollout
        num_repeats = self.config.actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts
        # 3. 交互式训练
```

**CollabLLM 的特点**：
- 支持用户交互式训练
- 多轮 rollout 收集
- 结合任务正确性和交互质量

### 6.4 Agent 训练的关键问题

**面试可能问到的问题**：

1. **多轮对话的 credit assignment 问题**
   - 最终 reward 如何分配到每一轮？
   - OPD 的 token-level reward 天然适合多轮场景

2. **工具调用的 reward 设计**
   - 工具调用成功/失败如何量化？
   - 如何结合 outcome reward 和 process reward？

3. **Agent 的 exploration vs exploitation**
   - 如何鼓励 agent 探索新工具？
   - 如何避免 agent 陷入重复调用？

### 6.5 Multi-Turn Rollout

```python
# verl/examples/sglang_multiturn/
# 支持多轮对话的 rollout 配置
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_turns=5
```

**技术挑战**：
- 每轮的 KV cache 管理
- 多轮的 token 预算控制
- 工具调用的异步处理

---

## 第七部分：评估体系与方法论

### 7.1 评估流程

```text
1. gen_vllm.py: vLLM 生成 N=16 个 response
2. grade.py: 抽取 \boxed{}，归一化，sympy 等价检查
3. grading_results.json: 汇总指标
```

### 7.2 评估指标

| 指标 | 含义 | 使用场景 |
|------|------|----------|
| `mean_score` | 所有 rollout 平均正确率 | 主要指标，接近单次 accuracy |
| `best_score` | 每题 N 个 rollout 至少一个正确 | 类似 pass@N，衡量采样潜力 |
| `solve_all` | N 个 rollout 全部正确的题数 | 衡量稳定性 |
| `format_error_rollouts` | 缺少 boxed 答案的 rollout 数 | 诊断生成问题 |

**面试要点**：
> `best_score` 不能直接当 accuracy。它是多次采样中至少一次答对的比例，类似 pass@16。简历和面试里应重点讲 `mean_score`，同时说明 `best_score` 反映多样采样潜力。

### 7.3 Max Tokens 的影响（重要 confounder）

```
MAX_TOKENS=2048:  AIME24 mean=1.9%, format error ~98%
MAX_TOKENS=16384: AIME24 mean=28.1%, format error 大幅下降
```

**原因分析**：
- 数学推理需要长 CoT
- 2048 会截断推理，没有最终 `\boxed{}`
- 评分直接 format error

**面试回答**：
> 一开始用 2048 做离线评测，AIME 指标非常低。但查看输出发现大量 response 没有最终 boxed 答案，本质是生成被截断。把评测 max tokens 调到 16384 后，format error 大幅下降，指标也上来了。这说明评估时 max tokens 是数学推理任务里的重要 confounder。

### 7.4 评估脚本的完整性问题

**已知问题**（`gen_vllm.py:185-187`）：
- worker 异常只 print，不退出
- 没有校验 `len(results) == samples * N`
- 可能保存不完整 JSONL

**修复建议**：
```python
# 保存前断言
assert len(all_results) == len(samples) * N, f"Expected {len(samples) * N}, got {len(all_results)}"
# 每个 example_id 恰好有 N 个 rollout
for example_id, count in rollout_counts.items():
    assert count == N, f"Example {example_id} has {count} rollouts, expected {N}"
```

### 7.5 评估口径一致性

**训练 vs 评估设置**：
```
训练：MAX_RESP_LENGTH=2048, N_RESPONSES=2
评估：MAX_TOKENS=16384, N=16
```

**严谨表述**：
> 在 2048 响应长度训练后，使用 16384 离线生成评测得到 AIME24 mean_score=28.1%。这不是和论文完全同口径的长上下文训练复现。

---

## 第八部分：高频面试问题深度解析

### Q1: 什么是 On-Policy Distillation？

**回答模板**：
> 普通 KD 通常是 teacher 在固定数据集上生成答案，student 去拟合这些样本，这是 off-policy 的。OPD 是 student 当前策略先 rollout，teacher 在 student 访问到的状态上给 token-level 反馈。这样 student 学的是"在自己会走到的状态上，teacher 会怎么分布"，更贴近 RL 中 on-policy 的思想。

**深挖点**：
- Off-policy KD 的 distribution mismatch 问题
- OPD 降低 mismatch 的代价是每步都要 teacher forward

### Q2: 为什么用 reverse KL 而不是正向 KL？

**回答**：
> 正向 KL `KL(teacher || student)` 更 mode-covering，鼓励 student 覆盖 teacher 的所有模式。reverse KL `KL(student || teacher)` 更 mode-seeking，惩罚 student 在 teacher 低概率 token 上放概率。OPD 的 on-policy 设定里，重点纠正 student 自己会选择的分布，所以 reverse KL 更自然。

**深挖点**：
- mode-covering vs mode-seeking 的直觉
- 在什么场景下正向 KL 更好？

### Q3: top-K 的 K 怎么选？

**回答**：
> K 太小，KL 估计偏差大，可能漏掉 teacher 重要但 student 不关注的 token。K 太大，更接近 full KL，但 teacher forward 后 gather/log-softmax 显存和计算更重。论文观察高概率质量集中在少数共享 token 上（97-99%），K=16 是精度与效率的 trade-off。

**深挖点**：
- 做 K=0/8/16/32 的 ablation
- K 和词表大小的关系

### Q4: OPD 和 DPO 的区别？

**回答**：
> DPO 用 preference pair 直接优化偏好，不需要在线 rollout，不需要 reward model。OPD 更像把 distillation 嵌进 on-policy RL 框架，用 teacher distribution 替代稀疏 outcome reward。DPO 是 pair-level signal，OPD 是 token-level dense signal。

### Q5: 为什么 OPD 还要 PPO clipping？

**回答**：
> 即使 reward 来自 teacher KL，也是在更新 policy。PPO clipping 限制新旧策略变化，避免一次更新过大导致训练不稳定。尤其 token-level reward 很密集，如果没有约束，可能快速把 student 拉向 teacher 分布，损害探索和原有能力。

### Q6: OPD 什么情况下会失败？

**回答**：
> 论文指出两个条件很重要：1）student 和 teacher 要有兼容 thinking pattern；2）teacher 要提供 student 没有的新能力。如果 teacher 和 student 格式不兼容，token-level KL 会强行对齐不合适的轨迹。如果 teacher 只是同分布略强，student 可能学不到有价值的东西。

### Q7: 如何判断 OPD 训练确实在工作？

**回答**：
> 我会看：1）`actor/pg_loss` 是否非零；2）`critic/score/mean` 作为负 KL 是否向 0 靠近；3）`topk/overlap_ratio` 是否上升；4）response length 和 format error 是否改善；5）最终离线 benchmark 是否改善。

**实测数据**：
```
step 1:    actor/pg_loss=0.2616, topk/overlap_ratio=0.7223
step 1119: actor/pg_loss=0.2205, topk/overlap_ratio=0.7625
```

### Q8: `critic/score/mean` 是负的，reward 怎么会负？

**回答**：
> 这里 score 是负 KL。KL 非负，所以负 KL 小于等于 0。越接近 0 表示 student 和 teacher 越对齐。它和 GRPO 里 0/1 correctness reward 不是一个量纲。

### Q9: flash-attn fallback 问题是什么？

**回答**：
> 远程环境 CUDA 13.1 和 flash-attn 预编译 wheel 不兼容。我加了一个纯 PyTorch fallback，但审查发现 `unpad_input` 只返回 3 个值，标准接口返回 5 个值。当前 FSDP 路径靠 `*_` 能跑，但 Megatron 路径会失败。这是临时 workaround，不是完整修复。

### Q10: 为什么不用 LoRA？

**回答**：
> 论文和项目目标是 full-parameter 后训练，要研究 OPD 动态和 token-level alignment。LoRA 可以节省显存，但会改变优化空间和复现口径。1.5B full fine-tuning 在 8x3090 上通过 FSDP/offload 已经可行，所以优先保持 full-param。

---

## 第九部分：业务场景与工程落地

### 9.1 OPD 的业务价值

**降本**：
- 用强 teacher 指导小 student
- 线上部署小模型，降低推理成本

**提质**：
- 长推理任务中 dense signal 提高训练效率
- 适合数学、代码、工具调用

### 9.2 业务落地方案

```text
1. 收集真实业务 prompt，脱敏、分桶、去重
2. student 当前线上模型生成多样 response
3. teacher 对相同 prompt 给 token-level feedback
4. 混合 outcome evaluator：业务规则、偏好模型、安全规则
5. 用 OPD 或 OPD+GRPO 训练 student
6. 离线评估：业务指标、安全指标、人工抽检
7. 灰度上线，监控 hallucination、拒答率、latency、成本
```

### 9.3 Hybrid Reward 设计

```python
total_reward = token_KL_reward + lambda * outcome_reward + safety_penalty + format_reward
```

**各部分作用**：
- `token_KL_reward`：dense learning signal
- `outcome_reward`：保证最终业务目标
- `safety_penalty`：保证安全合规
- `format_reward`：保证结构化输出

### 9.4 上线监控指标

| 类别 | 指标 |
|------|------|
| 效果 | 任务成功率、正确率、用户满意度 |
| 安全 | hallucination rate、安全违规率、拒答率 |
| 性能 | P50/P95/P99 latency、QPS |
| 成本 | 每千次请求成本、GPU 利用率 |

### 9.5 灰度与回滚

**灰度策略**：
- 小流量 A/B test
- Shadow evaluation（新模型结果不影响用户）
- 逐步扩大流量

**回滚方案**：
- 保留上一稳定模型 bundle（权重 + tokenizer + config + template）
- 关键指标超阈值自动切回
- 数据和训练版本可追溯

---

## 第十部分：面试官视角的能力考察点

### 10.1 五类能力

| 能力 | 考察方式 | 你的抓手 |
|------|----------|----------|
| 底层原理 | "为什么这么设计？" | OPD 解决 distribution mismatch + sparse reward |
| 实验验证 | "怎么证明有效？" | 训练信号、top-K overlap、评估结果、confounder 控制 |
| 问题定位 | "结果不符合预期怎么办？" | 指标层→数据层→算法层→系统层→环境层 |
| 工程落地 | "怎么上线？" | 训练 pipeline、部署、监控、灰度、回滚 |
| 业务理解 | "有什么价值？" | 强 teacher → 小 student 降本提质 |

### 10.2 面试官想看到的不是

- ❌ "我跑出了一个分数"
- ❌ "我背住了公式"
- ❌ "我完全复现了论文"

### 10.3 面试官想看到的是

- ✅ 我理解算法目标、系统资源和评估可信度的关系
- ✅ 我能从日志和指标判断训练是否健康
- ✅ 我能诚实承认边界和不足
- ✅ 我能设计 ablation 验证假设
- ✅ 我能把方案迁移到业务场景

### 10.4 回答的四段式框架

每个深挖题都用这个结构：

1. **问题背景**：这个方法解决什么真实问题
2. **核心机制**：具体怎么做，落到公式、代码、配置
3. **局限与风险**：主动讲不适用条件
4. **改进与验证**：如果继续做，如何证明或修复

**示例**：
> OPD 解决的是 GRPO 等 outcome-level RL 在数学推理早期 reward 稀疏、以及传统 off-policy distillation 存在 distribution mismatch 的问题。它让 student 自己 rollout，再用 teacher 在 student 访问到的 token 状态上给 dense KL reward。我的复现里通过 `token_reward_direct` 直接把 token reward 当 advantage，top-K=16，默认 only_stu 策略。但它也有局限，比如 teacher/student thinking pattern 不兼容时会错对齐，teacher 成本高，且我的训练 response length 是 2048，离线评估是 16384，口径要说明。

### 10.5 主动抛给面试官的技术点

1. **"这个项目里最关键的是 reward 的粒度。"**
   GRPO 是 response-level sparse reward，OPD 是 token-level dense reward。

2. **"最大的工程瓶颈是 hybrid engine 下 vLLM KV cache 和 FSDP 训练状态抢显存。"**

3. **"评估结果最容易被 max tokens 误导。"**

4. **"如果继续做，我会做 top-K strategy、teacher temperature、GRPO+OPD hybrid 的 ablation。"**

---

## 附录 A：核心代码速查表

| 主题 | 文件位置 | 行号 |
|------|----------|------|
| 训练主循环 | `verl/verl/trainer/ppo/ray_trainer.py` | 967 |
| GRPO advantage | `verl/verl/trainer/ppo/core_algos.py` | 265 |
| OPD advantage | `verl/verl/trainer/ppo/core_algos.py` | 855 |
| PPO loss | `verl/verl/trainer/ppo/core_algos.py` | 1058 |
| loss aggregation | `verl/verl/trainer/ppo/core_algos.py` | 942 |
| student top-K | `verl/verl/workers/actor/dp_actor.py` | 86 |
| distillation reward | `verl/verl/workers/actor/dp_actor.py` | 451 |
| teacher top-K | `verl/verl/workers/fsdp_workers.py` | 1831 |
| teacher forward | `verl/verl/workers/fsdp_workers.py` | 1926 |
| 数据加载 | `verl/verl/utils/dataset/rl_dataset.py` | - |
| 评估生成 | `scripts/val/eval/gen_vllm.py` | - |
| 评估评分 | `scripts/val/eval/grade.py` | - |
| OPD 训练脚本 | `on_policy_distillation.sh` | - |
| 3090 适配脚本 | `opd_3090.sh` | - |

## 附录 B：必须背诵的术语表

| 术语 | 解释 |
|------|------|
| On-Policy | 在当前策略分布上采样和训练 |
| Off-Policy | 在历史或其他策略分布上训练 |
| Reverse KL | `KL(π_student \|\| π_teacher)`，mode-seeking |
| Forward KL | `KL(teacher \|\| student)`，mode-covering |
| Top-K | 只取概率最高的 K 个 token 近似 KL |
| Token-Level Reward | 每个 token 位置都有 reward 信号 |
| Outcome Reward | 只在 response 结束时有 reward |
| FSDP | Fully Sharded Data Parallel，参数分片 |
| Colocation | Actor/Rollout 共享同一组 GPU |
| Hybrid Engine | rollout 和 training 交替使用 GPU |
| PPO Clipping | 限制 policy update 幅度 |
| GAE | Generalized Advantage Estimation |
| GRPO | Group Relative Policy Optimization |
| DAPO | Dynamic Sampling Policy Optimization |
| vLLM | 高吞吐 LLM 推理框架 |
| PagedAttention | vLLM 的 KV cache 管理技术 |
| Ray | 分布式计算框架 |
| bf16 | Brain Float 16，保留指数范围 |
| Gradient Checkpointing | 重计算换显存 |
| Activation Offload | 激活值卸载到 CPU |
| Format Error | 输出缺少 `\boxed{}` 等格式 |
| mean_score | 所有 rollout 平均正确率 |
| best_score | N 个 rollout 至少一个正确（pass@N） |

## 附录 C：面试前最后检查清单

### 必须能画的图
1. verl 组件图（Actor/Rollout/Ref/RewardModel）
2. OPD 训练数据流（7 步）
3. Top-K KL 计算流程
4. FSDP 显存分配

### 必须能解释的词
- on-policy vs off-policy
- reverse KL vs forward KL
- token-level reward vs outcome reward
- top-K approximation
- colocation / hybrid engine
- PPO clipping
- FSDP / vLLM / Ray
- format error / mean_score vs best_score

### 不能过度宣称
- ❌ 不说完全复现论文所有规模
- ❌ 不说 OPD 绝对优于 GRPO
- ❌ 不说 16384 评测就是训练时长上下文能力
- ❌ 不说 flash-attn fallback 已经完全解决
- ❌ 不说 SFT demo 是完整 SFT 复现

### 一句话总结（面试收尾用）

> 这个项目展示的是我能把 LLM 后训练论文读懂、跑通、拆解训练信号，并且能诚实分析复现结果和工程风险。

---

## 附录 D：延伸阅读

### verl 相关
- verl 论文：*HybridFlow: A Flexible and Efficient RLHF Framework* (arXiv:2409.19256)
- verl 文档：https://verl.readthedocs.io/
- verl 支持的算法：PPO, GRPO, DAPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, SPIN 等

### 后训练算法
- GRPO：*DeepSeekMath: Pushing the Limits of Mathematical Reasoning*
- DAPO：*DAPO: An Open-Source LLM Reinforcement Learning System*
- DPO：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- PPO：*Proximal Policy Optimization Algorithms*

### 蒸馏相关
- GKD：*Generalized Knowledge Distillation for LLMs*
- OPD 论文：*Rethinking On-Policy Distillation of Large Language Models*

### Agent 相关
- ReTool：*ReTool: Using multi-round conversations and code sandboxing*
- CollabLLM：*CollabLLM: Training LLMs for Collaborative Tasks*
- LangGraph：https://langchain-ai.github.io/langgraph/

---

> **最后提醒**：这份文档是面试准备材料，不是背诵材料。理解原理、能举例子、能讲清楚"为什么"比记住细节更重要。面试时保持诚实，主动讲边界和不足，比过度包装更能赢得信任。
