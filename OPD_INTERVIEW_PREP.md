# Rethink OPD 复现项目面试准备手册

面向：正在找 LLM 算法实习的 MS 在读候选人  
项目路径：`/home/chenyizhou/OPD`  
文档目标：把“我复现了 Rethinking OPD”转化成面试中能讲清楚、能被追问、能体现后训练理解和工程落地能力的材料。

---

## 0. 怎么使用这份文档

建议按三层准备：

1. **先背熟项目介绍稿**：30 秒、1 分钟、3 分钟三个版本，保证开场不飘。
2. **再掌握框架和训练链路**：能白板画出 verl 中 actor/rollout/ref/reward model 的数据流。
3. **最后准备拷打题**：面试官会追问 OPD 为什么有效、为什么会失败、与 GRPO/RLHF/DPO/SFT 的区别、如何迁移到业务和基模后训练。

这份文档中，“实际复现结果”都来自当前项目已有日志、评测文件和学习笔记；“建议说法”是面试表达模板，不要把未做过的 ablation 讲成已完成实验。

---

## 1. 简历上怎么写这个项目

### 1.1 中文简历 bullet 推荐

**项目名：Rethinking OPD 论文复现与 8x3090 后训练适配**

- 基于 verl v0.7.0 和 LlamaFactory 复现 Rethinking On-Policy Distillation 中的 1.5B OPD 后训练流程，理解并验证 student rollout、teacher token-level KL reward、top-K 蒸馏、PPO policy update 与离线数学评测链路。
- 将原 8xA800 80GB 配置适配到 8xRTX3090 24GB：使用 bf16、FSDP param/optimizer offload、gradient checkpointing、activation offload、降低 `MAX_RESP_LENGTH`/`N_RESPONSES`/`MINI_BATCH_SIZE`、调低 vLLM `gpu_memory_utilization`，完成 1119 步 OPD 训练。
- 在 AIME24/AIME25/AMC23 上进行 N=16 rollout 离线评测，定位 `MAX_TOKENS=2048` 导致严重 format error，改用 16384 生成上限后得到 AIME24 mean 28.1%、AIME25 mean 24.0%、AMC23 mean 72.2% 的规则评分结果。
- 审查并记录复现中的工程问题：flash-attn fallback API 不兼容、评测脚本可能静默保存不完整结果、训练/评测长度口径混用、文档指标错位，并给出修复建议。

### 1.2 英文简历 bullet 推荐

- Reproduced the 1.5B On-Policy Distillation pipeline from Rethinking OPD with verl and LlamaFactory, covering student on-policy rollout, teacher token-level KL reward, top-K distillation, PPO-style actor update, and offline math evaluation.
- Adapted the original 8xA800-80GB recipe to 8xRTX3090-24GB by enabling bf16, FSDP parameter/optimizer offload, activation offload, gradient checkpointing, shorter response length, smaller rollout count, and lower vLLM GPU memory utilization.
- Completed a 1119-step OPD run and evaluated the merged checkpoint on AIME24/AIME25/AMC23 with 16 rollouts per prompt; identified response truncation as the main cause of format errors and improved offline evaluation by increasing max generation length.
- Audited reproducibility risks in the codebase, including an incomplete flash-attn fallback API, silent failures in the vLLM evaluation script, Python environment mismatch, and inconsistent reporting between training and evaluation settings.

### 1.3 简历中不要这么写

不要写：

- “完全复现论文所有结果。”
- “证明 OPD 一定优于 GRPO。”
- “实现了一个生产级 OPD 训练框架。”
- “解决了 flash-attn 依赖问题。”

更稳妥的写法：

- “完成 1.5B 规模、8x3090 约束下的可行性复现和工程适配。”
- “对比分析了 GRPO 稀疏 outcome reward 与 OPD dense token reward 的训练信号差异。”
- “发现并记录了若干复现可靠性风险，给出修复方案。”

---

## 2. 面试开场逐字稿

### 2.1 30 秒版本

我简历里这个 Rethink OPD 项目主要是复现一篇关于 on-policy distillation 的 LLM 后训练工作。传统蒸馏通常是在 teacher 生成的固定数据上做 SFT，而 OPD 是让 student 自己 rollout，然后在 student 访问到的状态上，用 teacher 的 token-level log-prob 或 KL 信号做 dense reward，再用 PPO 风格的 policy update 更新 student。我在 8 张 3090 24GB 上把原来 A800 80GB 的配置缩小并跑通了 1.5B 模型的 OPD 训练和 AIME/AMC 离线评测，也审查了复现中的一些工程问题，比如 flash-attn fallback 和评测脚本完整性。

### 2.2 1 分钟版本

这个项目我主要想展示两个能力：一是理解 LLM 后训练算法，二是能把论文代码在受限硬件上真的跑起来。

算法上，OPD 的关键是把 distillation 放到 student 的 on-policy 分布上。每一步先用当前 student 通过 vLLM 生成响应，再用 student 和 teacher 在每个 token 位置的 top-K 分布计算近似 reverse KL，把负 KL 当作 token-level reward。这样 advantage 不再像 GRPO 那样依赖最终答案对错，而是每个 token 都有 dense signal。

工程上，原项目默认是 8xA800 80GB，我这边只有 8x3090 24GB，所以改了很多资源配置：bf16、FSDP offload、gradient checkpointing、activation offload、降低响应长度和 rollout 数、调低 vLLM KV cache 显存比例。最终完成了 1119 步 1.5B OPD 训练，并做了 AIME24/AIME25/AMC23 的 N=16 离线评测。过程中我也发现了复现文档和代码中的几个风险，比如评测 max tokens 太短会造成大量 format error，以及 flash-attn fallback 实现其实不是完整兼容。

### 2.3 3 分钟版本

我这个项目复现的是 Rethinking On-Policy Distillation。它讨论的问题是：LLM 后训练里，单纯用 outcome reward 做 RL，比如 GRPO，数学任务上信号很稀疏；而传统 off-policy distillation 又是在 teacher 或固定数据分布上训练，student 没有真正从自己的错误轨迹上学习。OPD 的想法是在 student 的 on-policy rollout 上做蒸馏：student 当前策略先生成 response，然后 teacher 在这些相同状态上给出 token-level 分布信号，训练目标本质上是让 student 在自己访问到的状态上对齐 teacher。

我重点理解了三块。第一是 reward：代码里 `token_reward_direct` 直接把 token-level KL reward 当 advantage，不走 GAE，也不需要 value baseline。第二是 top-K：完整词表 KL 太贵，所以只在 student 或 teacher 的 top-K token 集上近似，比如默认 `only_stu`，teacher 只需要给出 student top-K token 的 log-prob。第三是系统实现：verl 用 Ray 管理 worker，Actor/Rollout/Ref colocate 在一组 GPU 上，rollout 阶段用 vLLM，训练阶段切回 FSDP actor，reward model 作为 teacher forward 计算 token-level log-probs。

我的实际复现是在 8xRTX3090 24GB 上做的，硬件比原论文 A800 80GB 紧张很多。所以我把原配置从 fp32 改成 bf16，打开 actor 的 param offload 和 optimizer offload，启用 activation offload 和 gradient checkpointing，把 `MAX_RESP_LENGTH` 从 7168 降到 2048，把 `N_RESPONSES` 降到 2，把 mini batch 降到 16，并把 vLLM `gpu_memory_utilization` 调到 0.4。最终完成 1119 步训练，日志里 step 1/2 的 `actor/pg_loss` 大约 0.26/0.27，`topk/overlap_ratio` 大约 0.72，说明 token-level 蒸馏信号确实在工作。

评估上，我用 AIME24、AIME25、AMC23，N=16 rollouts，规则评分。一个重要发现是，如果评估 max tokens 仍用 2048，模型大量被截断，format error 很高，AIME 几乎看起来很差；把离线评测生成上限调到 16384 后，AIME24 mean_score 到 28.1%，AIME25 到 24.0%，AMC23 到 72.2%。但我也会在面试里强调：这不是严格同口径论文复现，因为训练响应长度是 2048，而离线评测 max tokens 是 16384，所以我会把它表述成 3090 约束下的可行性复现和工程分析。

---

## 3. 面试官听完后希望你体现什么

### 3.1 你不是只会跑脚本

你要体现：

- 知道 OPD 为什么是 on-policy。
- 知道 token-level KL reward 怎么进入 PPO loss。
- 知道 top-K 是为了降低 full-vocab KL 的计算和显存成本。
- 知道 verl 的 actor/rollout/ref/reward model 各自负责什么。
- 知道 3090 适配为什么要改响应长度、rollout 数、bf16、offload、vLLM 显存比例。
- 知道评估结果为什么容易被 `max_tokens`、format error、best_score/mean_score 口径影响。

### 3.2 你能承认边界

面试中主动说清楚边界会加分：

- 我完成的是 1.5B、8x3090 条件下的可行性复现，不是论文全部规模复现。
- SFT 部分是 LlamaFactory smoke test，不是完整 SFT 数据集复现。
- 16384 离线评测结果说明模型长生成能力释放出来，但训练时仍是 2048 response length。
- flash-attn fallback 当前只是让 FSDP 路径跑通，不是完整兼容实现。
- GRPO 只跑到 131 步左右，没有做完整公平对比。

这说明你有科研和工程的诚实感。

---

## 4. 一张白板讲清 OPD

### 4.1 核心公式

在 student 生成的 response token 位置 t 上，计算：

```text
r_t = - KL(pi_student(. | s_t) || pi_teacher(. | s_t))
```

因为完整词表太大，用 top-K token set 近似：

```text
KL_t ~= sum_{a in K_t} w(a) * [log pi_student(a | s_t) - log pi_teacher(a | s_t)]
reward_t = - KL_t
```

默认配置中：

- `K=16`
- `top_k_strategy=only_stu`
- `reward_weight_mode=student_p`

所以可以讲成：

> 我们先取 student 在当前位置最相信的 K 个 token，再让 teacher 对这些 token 给 log-prob。如果 student 高概率 token 在 teacher 那里也高概率，KL 小，reward 没那么负；如果 student 把概率放到 teacher 不认可的 token 上，负 KL reward 会惩罚它。

### 4.2 OPD 训练数据流

```text
prompt batch
  -> student actor / vLLM rollout 生成 responses
  -> actor forward 计算 old_log_probs、student top-K log_probs
  -> teacher reward model forward 计算 teacher on student top-K log_probs
  -> compute_distillation_reward 得到 token-level KL reward
  -> token_reward_direct: advantage = reward * response_mask
  -> PPO clipped loss 更新 actor
```

### 4.3 verl 组件图

```text
RayPPOTrainer
  |
  +-- ActorRolloutRefWorker
  |     +-- Actor: FSDP student, 有 optimizer，用于训练
  |     +-- Rollout: vLLM student，用于高吞吐生成
  |     +-- Ref: FSDP reference policy，用于可选 KL/reference log-prob
  |
  +-- RewardModelWorker
        +-- Teacher model: FSDP CausalLM，用于 token-level log-prob reward
```

你可以补一句：

> verl 的 hybrid engine 会在 rollout 和 training 两个模式之间切换，同一组 GPU 上先用 vLLM 生成，再释放 KV cache 切回 FSDP 训练，这就是 colocation 设计。

---

## 5. 框架学习地图

### 5.1 你应该熟悉的文件

| 主题 | 文件 | 面试中怎么讲 |
|---|---|---|
| 训练主循环 | `verl/verl/trainer/ppo/ray_trainer.py:967` | `fit()` 中串起 rollout、logprob、reward、advantage、actor update |
| GRPO advantage | `verl/verl/trainer/ppo/core_algos.py:265` | outcome reward 的组内标准化 |
| OPD advantage | `verl/verl/trainer/ppo/core_algos.py:855` | `token_reward_direct` 直接把 token reward 当 advantage |
| policy loss | `verl/verl/trainer/ppo/core_algos.py:1058` | PPO clipped loss，top-K 时 advantage 是 3D |
| loss aggregation | `verl/verl/trainer/ppo/core_algos.py:942` | token-mean、seq-mean-token-mean 等 |
| student top-K | `verl/verl/workers/actor/dp_actor.py:86` | actor forward 中取 student top-K log_probs |
| distillation reward | `verl/verl/workers/actor/dp_actor.py:451` | 由 student/teacher log-prob 算 KL reward |
| teacher top-K | `verl/verl/workers/fsdp_workers.py:1831` | teacher 计算 top-K、overlap mask、teacher-on-student log-probs |
| reward model forward | `verl/verl/workers/fsdp_workers.py:1926` | teacher forward，应用 teacher temperature |
| 数据加载 | `verl/verl/utils/dataset/rl_dataset.py` | parquet -> chat template -> tokenize -> left pad |
| 评估生成 | `scripts/val/eval/gen_vllm.py` | vLLM 多 GPU 生成 JSONL |
| 评估评分 | `scripts/val/eval/grade.py`、`utils.py` | boxed answer 抽取、归一化、sympy 等价 |

### 5.2 训练一步发生了什么

面试中可以按这个顺序回答：

1. `generate_sequences()`：当前 student policy 用 vLLM 对 prompt 生成 N 个 responses。
2. `compute_log_prob()`：FSDP actor 重新算这些 responses 的 token log-prob，以及 top-K log-prob。
3. `compute_rm_score()`：teacher reward model 在同样 token 序列上 forward，得到 teacher log-prob。
4. `compute_distillation_reward()`：对 student top-K token set 计算近似 KL reward。
5. `compute_reward()`：如果还有 rule-based reward 或 format reward，会在这里合并。
6. `compute_advantage()`：OPD 用 `token_reward_direct`，直接 `advantages = token_level_rewards * response_mask`。
7. `update_actor()`：用 PPO clipped loss 更新 student。

### 5.3 为什么 OPD 不像 GRPO 那样强依赖最终答案

GRPO 需要等完整 response 生成后，用规则判断答案对错。数学题上初始 1.5B 模型正确率低，很多 batch reward 全 0，`pg_loss` 也可能是 0。

OPD 的 teacher log-prob 在每个 token 位置都提供信号，即使最后答案错了，模型仍能从中间推理步骤的 token 分布差异得到梯度。

但要注意：

- dense reward 不代表一定正确。
- teacher 如果和 student thinking pattern 不兼容，token-level signal 可能拉偏。
- teacher 如果没有新能力，只是同分布更强一点，OPD 可能收益有限。

---

## 6. 你的实际复现怎么讲

### 6.1 实验设置

实际 OPD 复现配置：

- 硬件：8 x NVIDIA RTX 3090，24GB。
- Student：`DeepSeek-R1-Distill-Qwen-1.5B`。
- Teacher：`JustRL-DeepSeek-1.5B`。
- 数据：`datasets/dapo-math-17k.parquet`。
- 算法：`ADV_ESTIMATOR=token_reward_direct`。
- Top-K：`LOG_PROB_TOP_K=16`。
- Top-K strategy：`only_stu`。
- Reward weight：`student_p`。
- Teacher temperature：1.0。
- Response length：训练 `MAX_RESP_LENGTH=2048`。
- N responses：训练 `N_RESPONSES=2`。
- Mini batch：16。
- 模型精度：bf16。
- vLLM 显存比例：`gpu_memory_utilization=0.4`。
- Offload：actor param offload、optimizer offload、ref offload、reward model param offload。

### 6.2 为什么这么改

原始实验是 8xA800 80GB，3090 只有 24GB，最大的压力是：

- rollout 阶段 vLLM 需要模型权重和 KV cache。
- training 阶段 FSDP 需要权重、梯度、优化器、激活值。
- OPD 比 GRPO 多一个 teacher reward model forward。

所以改动的逻辑是：

- bf16：降低 rollout 权重和激活显存。
- `MAX_RESP_LENGTH=2048`：减少 KV cache 和每步 token 数。
- `N_RESPONSES=2`：减少每个 prompt 的 rollout 数，降低 batch token 压力。
- `MINI_BATCH_SIZE=16`：缩小 PPO 更新批量。
- `gpu_memory_utilization=0.4`：给 FSDP 和 reward model 留显存。
- param/optimizer offload：用 CPU 内存换 GPU 显存。
- gradient checkpointing/activation offload：用计算和 CPU 内存换激活显存。

### 6.3 实际训练信号怎么解读

日志中 OPD 的关键事实：

| Step | actor/pg_loss | critic/score/mean | topk/overlap_ratio | 解读 |
|---:|---:|---:|---:|---|
| 1 | 0.2616 | -0.2616 | 0.7223 | 训练一开始就有 KL reward，不是稀疏全 0 |
| 2 | 0.2745 | -0.2731 | 0.7319 | teacher/student top-K 有约 73% 重叠 |
| 10 | 0.2738 | -0.2614 | 0.7240 | 蒸馏信号稳定存在 |
| 1119 | 0.2205 | -0.2040 | 0.7625 | KL gap 降低，top-K overlap 增加 |

面试中可以这样讲：

> OPD 的 `critic/score/mean` 是负值，不代表模型变差，因为这里的 score 本质是负 KL reward。KL 越小，reward 越接近 0；所以从 -0.26 到 -0.20 可以理解为 student 和 teacher 的 token 分布更接近。同时 top-K overlap 从约 0.72 到 0.76，说明 student 高概率 token 集合和 teacher 更一致。

### 6.4 与 GRPO 早期训练对比

GRPO 实验早期：

- step 4：`critic/score/mean=0.03125`，32 个样本中约 1 个正确。
- step 10：`critic/score/mean=0.03125`。
- 大部分 step `actor/pg_loss=0`。

可讲结论：

> 在 1.5B 数学模型上，GRPO 的 outcome reward 很稀疏，早期很多 batch 没有正样本，policy gradient 信号很弱。OPD 用 teacher 分布提供 dense token reward，即使最后答案错，也能在每个 token 位置得到学习信号。

注意别讲成：

> OPD 一定比 GRPO 好。

更准确：

> 在我的复现实验和论文设定里，OPD 的 dense reward 解决了 GRPO 早期信号稀疏的一部分问题，但公平比较还需要同模型、同训练步数、同 response length、同评测口径的完整实验。

### 6.5 评估结果怎么讲

离线评估设置：

- Tasks：AIME24、AIME25、AMC23。
- N=16 rollouts per prompt。
- temperature=0.7，top_p=0.95。
- 规则评分：抽取 `\boxed{}`，归一化，sympy 等价检查。

关键结果：

| 任务 | MAX_TOKENS=2048 mean | MAX_TOKENS=16384 mean | 为什么重要 |
|---|---:|---:|---|
| AMC23 | 23.6% | 72.2% | AMC 更短，长生成后格式错误明显下降 |
| AIME24 | 1.9% | 28.1% | 2048 时大量被截断 |
| AIME25 | 1.3% | 24.0% | 同样受截断影响 |

怎么解释：

> 一开始我用 2048 做离线评测，AIME 指标非常低，但查看输出发现大量 response 没有最终 `\boxed{}`，本质是生成被截断。把评测 max tokens 调到 16384 后，format error 大幅下降，AIME 指标也上来了。这说明评估时 max tokens 是数学推理任务里的重要 confounder。

边界：

> 训练时 response length 仍是 2048，离线评测是 16384，所以这不能说完全复现了论文长上下文训练设置；更准确是短 response 训练后的长生成评测。

---

## 7. 面试可深挖点：算法

### Q1. 什么是 On-Policy Distillation？为什么不是普通 KD？

回答：

普通 KD 通常是 teacher 在固定数据集或 teacher 自己分布上生成答案，student 去拟合这些样本，这是 off-policy 的。OPD 是 student 当前策略先 rollout，teacher 在 student 访问到的状态上给 token-level 反馈。这样 student 学的是“在自己会走到的状态上，teacher 会怎么分布”，更贴近 RL 中 on-policy 的思想。

深挖：

- Off-policy KD 的问题是 distribution mismatch：student 训练时看到的是 teacher trajectories，部署时走的是自己的 trajectories。
- OPD 降低这个 mismatch，但代价是每一步都要生成 rollout 和 teacher forward，计算更贵。

### Q2. OPD 的 reward 是什么？

回答：

在每个 token 位置，reward 是 student 分布和 teacher 分布之间 KL 的负值：

```text
r_t = - KL(pi_student || pi_teacher)
```

实现时不会对全词表算 KL，而是在 top-K token set 上近似。默认取 student top-K，所以 teacher 只需要给这些 token 的 log-prob。

深挖：

- KL 小，说明 student 与 teacher 对下一个 token 的偏好一致，reward 接近 0。
- KL 大，说明 student 把概率放在 teacher 不认可的 token 上，reward 更负。
- 这是 dense reward，每个 response token 都有信号。

### Q3. 为什么用 reverse KL？

回答：

这里的形式是 `KL(student || teacher)`，更像 mode-seeking：鼓励 student 把概率质量集中在 teacher 高概率区域。对后训练来说，这能避免 student 在 teacher 认为不好的 token 上放太多概率。

补充：

- 正向 KL `KL(teacher || student)` 更 mode-covering，会鼓励 student 覆盖 teacher 的所有模式。
- reverse KL 更保守、更偏向 teacher 的高置信区域，但也可能降低 diversity。

### Q4. 为什么要 top-K？

回答：

LLM 词表可能 30K 到 150K，若每个 token 位置都对全词表算 KL，显存和计算量都很高。论文观察高概率质量集中在少数共享 token 上，所以用 top-K 近似 KL。我的配置中 K=16。

如果追问 K 太小怎么办：

- K 太小：KL 估计偏差大，可能漏掉 teacher 重要但 student 不关注的 token。
- K 太大：更接近 full KL，但 teacher forward 后 gather/log-softmax 显存和计算更重。
- K 是精度与效率的 trade-off。

### Q5. top-K strategy 有哪些？

回答：

常见策略：

- `only_stu`：取 student top-K，问 teacher 这些 token 的概率。默认配置，student-driven。
- `only_tch`：取 teacher top-K，看 student 对 teacher 高概率 token 的概率。teacher-driven。
- `intersection`：只在 student/teacher 都进入 top-K 的 token 上算，保守但可能稀疏。
- `union`：合并两边 top-K，覆盖更全但成本更高。
- `union-intersection`：关注双方分歧的 token。

我的复现用 `only_stu`，因为它只针对 student 当前会高概率选择的 token 给反馈，计算比较稳定。

### Q6. reward_weight_mode 为什么用 student_p？

回答：

`student_p` 是用 student 在 top-K 上的概率作为权重。这样 student 自己高置信的 token 对 KL reward 贡献更大。如果 student 把高概率放错了，teacher 能强烈纠正；低概率 token 影响较小，减少噪声。

对比：

- `teacher_p` 更强调 teacher 认为重要的 token，可能更强但也可能拉 student 去它当前完全不理解的区域。
- `none` 是 uniform weighting，简单但没有概率重要性。

### Q7. `token_reward_direct` 和 GAE/GRPO 的区别？

回答：

GAE/GRPO 通常从 outcome reward 或 value function 估计 advantage。`token_reward_direct` 很简单：

```python
advantages = token_level_rewards * response_mask
returns = advantages.clone()
```

它不需要 gamma、不需要 lambda、不需要 value baseline，因为 reward 本身已经是每个 token 的 dense teacher signal。

追问：那 critic 还需要吗？

> 对 OPD 这条路径不一定需要 critic。verl 代码是 PPO/RL 通用框架，会保留相关组件；真正 advantage 这里直接来自 token reward。

### Q8. PPO clipped loss 在 OPD top-K 下怎么工作？

回答：

普通 PPO 对每个 response token 有一个 log_prob 和 advantage。OPD top-K 时，advantage 变成 `(batch, seq_len, K)`，因为每个 token 位置有 K 个候选 token 的 KL reward。actor forward 也要重新算这些 top-K token 的 log_probs，然后 policy loss 在 K 维上聚合，再按 token mask 做 loss aggregation。

关键点：

- 不是只更新实际采样 token。
- 它对 student top-K token 分布整体做调整。
- 这更接近 distribution-level distillation，而不是 sample-level imitation。

### Q9. OPD 为什么可能失败？

回答：

论文指出两个条件很重要：

1. student 和 teacher 要有兼容 thinking pattern。
2. teacher 要提供 student 没有的新能力。

如果 teacher 和 student 格式不兼容，比如一个 thinking 模型、一个 non-thinking 模型，token-level KL 会强行对齐不合适的轨迹。如果 teacher 只是同分布略强，但没有真正新能力，student 可能学不到有价值的东西。

补充：

- 这也是为什么 chat template、`enable_thinking`、模型家族兼容性在复现里非常重要。

### Q10. OPD 和 DPO/RLHF 有什么区别？

回答：

- SFT：用标准答案或 teacher 数据做监督，主要是 imitation。
- DPO：用 preference pair 直接优化偏好，不需要在线 rollout，不需要 reward model 训练环路。
- RLHF/GRPO：通常用 outcome reward 或 reward model 标量奖励更新策略。
- OPD：teacher 不是给一个标量偏好，而是在 student rollout 的每个 token 位置给分布级 dense reward。

一句话：

> OPD 更像把 distillation 嵌进 on-policy RL 框架，用 teacher distribution 替代稀疏 outcome reward。

---

## 8. 面试可深挖点：系统与工程

### Q11. verl 里 actor、rollout、ref、reward model 分别是什么？

回答：

- Actor：FSDP 包装的 student，有 optimizer，负责训练。
- Rollout：vLLM 实例，用当前 student 权重高吞吐生成 responses。
- Ref：参考模型，用于可选 KL/reference log-prob。
- Reward model：在 OPD 中就是 teacher model，不是传统分类 reward model，而是用于 teacher log-prob。

### Q12. 为什么要 actor/rollout colocation？

回答：

训练时需要 FSDP，生成时需要 vLLM。如果为 actor 和 rollout 分配两套 GPU，会浪费显存。verl hybrid engine 把它们 colocate 在同一组 GPU 上，在 rollout mode 和 trainer mode 之间切换：rollout 时 gather 权重给 vLLM，training 时释放 KV cache 回到 FSDP。

代价：

- context switching 有开销。
- 显存峰值要兼顾 vLLM KV cache 和 FSDP 训练。

### Q13. 为什么 vLLM `gpu_memory_utilization` 要降到 0.4？

回答：

这个参数主要控制 vLLM KV cache 预算。原始 A800 80GB 可以给 vLLM 很大空间，但 3090 只有 24GB。OPD 同时还有 FSDP actor、ref、teacher reward model 和激活值压力，所以我把它降到 0.4，为训练阶段留空间。

### Q14. FSDP offload 为什么有用？

回答：

FSDP 已经把参数、梯度、优化器分片到多卡，但 24GB GPU 仍然紧张。param offload 和 optimizer offload 把部分状态放到 CPU，用 CPU 内存和 PCIe 通信换 GPU 显存。我的机器 CPU 内存有 200GB+，所以这是可行 trade-off。

### Q15. bf16 解决了什么？

回答：

bf16 能减少模型权重和激活的显存，尤其 rollout 阶段 vLLM 使用 gathered weights 时，1.5B 权重从 fp32 约 6GB 降到 bf16 约 3GB。同时 bf16 保留 exponent 范围，比 fp16 对训练稳定性更友好。

注意：

Adam optimizer 可能仍保留 fp32 master states，所以训练总状态不一定完全减半，但 rollout 权重和激活会明显受益。

### Q16. 你遇到的 flash-attn 问题是什么？

回答：

远程环境 CUDA 13.1 和 flash-attn 预编译 wheel 不兼容，从源码编译又有 nvcc 路径问题。为了让 FSDP + sdpa 路径跑通，项目中加了一个纯 PyTorch fallback。

但我后来审查发现这个 fallback 并不完整：`unpad_input` 只返回 3 个值，而 flash-attn 标准接口返回 `hidden_states, indices, cu_seqlens, max_seqlen_in_batch, used_seqlens`。当前 FSDP 路径靠 `*_` 能跑，但 Megatron 或 sequence-parallel 路径会失败。所以这应该被描述为临时 workaround，而不是彻底修复。

### Q17. 为什么评测脚本有静默失败风险？

回答：

`gen_vllm.py` 中 worker 捕获异常后只 print，然后返回已有结果；主进程只要 `all_results` 非空就保存 JSONL，没有校验行数是否等于 `num_samples * N`。这意味着某个 GPU OOM 或 tokenizer error 可能导致不完整评测文件，但评分脚本仍会计算指标。

我在审查报告里建议：

- worker 失败直接让主进程退出。
- 保存前校验每题 rollout 数。
- JSONL 中记录 task/model/max_tokens 等元信息。

### Q18. 裸 `python3` 为什么是问题？

回答：

远程裸 `python3` 是 3.5.4，但项目虚拟环境 `.venv/bin/python` 是 3.12.3。训练脚本里直接写 `python3 -m verl.trainer.main_ppo`，如果没有提前激活 venv，就会用错解释器，评测脚本里的 f-string 和类型标注会直接语法失败。

这是典型复现工程问题：能在我当前 shell 跑，不代表脚本本身可复现。

---

## 9. 面试可深挖点：数据与评估

### Q19. 训练数据是什么？

回答：

OPD 和 GRPO 主要用 `DAPO-Math-17k`，是数学题数据，项目中路径是 `datasets/dapo-math-17k.parquet`。数据通过 verl 的 RL dataset pipeline 加载，提取 prompt，应用 chat template，再 tokenize 和 left pad。

### Q20. 为什么 `enable_thinking=False` 重要？

回答：

Qwen3 等模型的 chat template 可能有 thinking/non-thinking 模式。如果训练 non-thinking 模型但忘记关 thinking，prompt/response 格式会被包上不匹配的 thinking 模板，导致 student 和 teacher 的 token 分布对齐错位。OPD 特别依赖 token-level 分布，所以 template 错比普通 SFT 更致命。

### Q21. 为什么要数据去重？

回答：

数学评测里如果训练集和测试集重叠，模型可能记住答案，指标虚高。项目里 `dedup_deepmath.py` 做两阶段去重：先 exact match，再用 Sentence-BERT + FAISS 做语义相似度过滤。虽然我的主实验用 DAPO-Math-17k，没有额外 DeepMath 混合，但面试中可以讲这是论文复现要关注的数据泄漏问题。

### Q22. 评估怎么做？

回答：

两阶段：

1. `gen_vllm.py` 用 vLLM 对 AIME24/AIME25/AMC23 生成 N=16 个 response。
2. `grade.py` 对每个 response 抽取最后的 `\boxed{}`，做 LaTeX/字符串归一化，再用规则和 sympy 等价判断。

指标：

- `mean_score`：所有 rollout 平均正确率。
- `best_score`：每题 N 个 rollout 里至少一个正确的比例，接近 pass@N。
- `solve_all`：某题 N 个 rollout 全部答对的题数。
- `format_error_rollouts`：缺少 boxed 答案的 rollout 数。

### Q23. 为什么 `best_score` 不能直接当 accuracy？

回答：

因为 best_score 是多次采样中至少一次答对的比例，类似 pass@N。它衡量“模型采样分布中有没有正确解”，不等于单次生成准确率。mean_score 更接近平均单次 rollout accuracy。

所以简历和面试里应该重点讲 mean_score，同时说明 best_score 反映多样采样潜力。

### Q24. 为什么 2048 max tokens 评估很差？

回答：

数学推理模型经常需要很长 CoT，2048 生成上限会截断推理，导致没有最终 `\boxed{}`，评分直接 format error。我的评测里 2048 时 AIME format error 接近 98%，把离线生成上限提高到 16384 后，format error 大幅下降，mean_score 明显提高。

### Q25. 评估结果如何严谨表述？

推荐说：

> 在我的本地 1.5B OPD checkpoint 上，使用 N=16、temperature=0.7、top_p=0.95、规则评分、MAX_TOKENS=16384 的离线评测，AIME24 mean_score 为 28.1%，AIME25 为 24.0%，AMC23 为 72.2%。但训练时 response length 是 2048，所以这不是和论文完全同口径的长上下文训练复现。

---

## 10. 业务与基模训练意义

### 10.1 OPD 对业务后训练有什么价值

可以这样回答：

> 业务场景中经常有一个更强但更贵的 teacher，比如大模型 API、专家模型或 ensemble，而线上要部署一个更小更便宜的 student。传统蒸馏只在固定数据上拟合 teacher 输出，无法覆盖 student 自己会犯错的状态。OPD 能让 student 在自己的 rollout 分布上接受 teacher 的 token-level 反馈，因此更适合修正 student 的真实行为分布。

具体价值：

- 降低部署成本：用强 teacher 改善小 student。
- 不完全依赖人工标注：teacher 分布可以提供 dense signal。
- 对长推理任务有帮助：中间 token 也有信号，不只最终对错。
- 可用于业务 prompt 分布：让 student 在真实业务请求上 rollout，再由 teacher 指导。

### 10.2 用到业务中会怎么设计

一个合理方案：

1. 收集真实业务 prompt，脱敏、分桶、去重。
2. student 当前线上或候选模型生成多样 response。
3. teacher 对相同 prompt/partial trajectory 给 token-level 或 step-level 反馈。
4. 混合 outcome evaluator：业务规则、偏好模型、安全规则。
5. 用 OPD 或 OPD+GRPO 训练 student。
6. 离线评估：业务指标、安全指标、人工抽检。
7. 灰度上线，监控 hallucination、拒答率、latency、成本。

### 10.3 对基模训练有什么意义

谨慎回答：

> OPD 本质是 post-training，不是预训练。它不会替代大规模 next-token pretraining，但可以作为基模后训练阶段的一种能力迁移和行为对齐方法。

可以用于：

- 将强 reasoning teacher 的能力迁移给小模型。
- 在特定任务域，例如数学、代码、工具调用，对 student 做 targeted post-training。
- 结合 prompt selection，让 teacher 在 student 薄弱分布上提供反馈。
- 作为 RLHF/RLAIF 中稀疏 reward 的补充。

不适合直接说：

- “OPD 可以替代 pretraining。”
- “OPD 能从无到有学会所有能力。”

### 10.4 OPD 在业务中的风险

- Teacher 成本高：每个 token 都要 teacher forward，成本可能很大。
- Teacher bias 会被蒸馏进 student。
- 格式不兼容会造成错误对齐。
- 如果业务答案没有明确 correctness，单靠 teacher KL 可能强化 teacher 的风格而非任务质量。
- 线上 prompt 分布变化时，需要持续监控和重采样。
- 安全场景需要额外 reward/evaluator，不能只看 teacher likelihood。

### 10.5 如何把 OPD 和业务 reward 结合

可以讲 hybrid：

```text
total_reward = token_KL_reward + lambda * outcome_reward + safety_penalty + format_reward
```

解释：

- token_KL_reward 提供 dense learning signal。
- outcome_reward 保证最终业务目标，例如答案正确、工具调用成功。
- safety_penalty 保证安全合规。
- format_reward 保证结构化输出。

但要调权重，避免 sparse outcome reward 完全压过 token KL，或 token KL 只学 teacher 风格不学业务目标。

---

## 11. 项目亮点如何包装

### 11.1 算法亮点

可以讲：

- 我不只是跑训练，还追到了 `token_reward_direct` advantage 的实现。
- 我理解 top-K KL 的计算路径：student top-K、teacher-on-student log-prob、reward weighting。
- 我理解 dense token reward 和 GRPO sparse outcome reward 的差异。
- 我能解释为什么 OPD 成败依赖 teacher/student thinking pattern。

### 11.2 工程亮点

可以讲：

- 把 A800 80GB 的配置迁移到 3090 24GB。
- 调整 FSDP offload、bf16、vLLM KV cache、response length、batch 和 rollout 数。
- 处理 CUDA/flash-attn/vLLM/Ray/tmp disk 等环境问题。
- 做了代码审查，发现评测脚本和 fallback 兼容性问题。

### 11.3 实验亮点

可以讲：

- 完成 1119 步训练。
- 观察到 OPD early steps 就有非零 policy gradient。
- 验证 top-K overlap 从约 0.72 到 0.76。
- 定位评估 max tokens 对 format error 的影响。

### 11.4 反思亮点

可以讲：

- 我会区分训练设置和评估设置。
- 我不会把 best_score 当单次 accuracy。
- 我不会把临时 fallback 说成完整工程修复。
- 我知道复现项目里文档和实际日志可能不一致，需要从证据出发。

---

## 12. 高频拷打题与参考回答

### Q26. 如果 teacher 比 student 强很多，`only_stu` 会不会漏掉 teacher 的关键 token？

会。`only_stu` 只看 student top-K，如果 teacher 的关键 token 不在 student top-K，就不会直接给这些 token 正向提升信号。它更像纠正 student 当前高概率 token 的分布。若 teacher 强很多，可以考虑 `only_tch` 或 `union`，但会增加计算并可能引入 student 当前难以学习的 token。

### Q27. 为什么不用 full-vocab KL？

full-vocab KL 最准确，但每个 token 位置都要对 15 万 vocab 做 log-softmax/聚合，训练长响应时显存和算力很贵。top-K 是效率近似，利用高概率 token 集中大部分概率质量的经验观察。

### Q28. `critic/score/mean` 是负的，面试官说 reward 怎么会负？

这里 score 是负 KL。KL 非负，所以负 KL 小于等于 0。越接近 0 表示 student 和 teacher 越对齐。它和 GRPO 里 0/1 correctness reward 不是一个量纲。

### Q29. OPD 会不会只学 teacher 风格，不学 correctness？

有这个风险。如果 teacher 分布高概率的是风格或格式，而不是正确推理，student 可能学到风格对齐。数学任务里可以结合 rule-based correctness reward 或 teacher-aligned prompt selection，保证能力信号不是纯风格信号。

### Q30. 如果 teacher 也会错，OPD 会怎样？

Teacher 错误会被蒸馏。OPD 假设 teacher 在 student 访问状态上提供更好的分布。如果 teacher 不可靠，可以：

- 只在高置信样本上蒸馏。
- 加 outcome verifier。
- 用多个 teacher ensemble。
- 对 teacher 输出做 rejection sampling 或 prompt selection。

### Q31. 为什么 response length 训练用 2048，评估用 16384？

训练用 2048 是硬件折中：3090 KV cache 和 FSDP 激活压力太大。评估用 16384 是为了避免数学推理被截断。这两个设置口径不同，所以我会明确说明这是受限硬件下训练、长生成下离线评估，不等于完全复现论文长上下文训练。

### Q32. 如果要做更严格实验，你下一步做什么？

我会做：

1. 同一模型的 baseline：base、GRPO、OPD 都用相同 max tokens 和 N 评估。
2. top-K ablation：K=0/8/16/32，比较 reward 分布和评估结果。
3. strategy ablation：only_stu vs only_tch vs union。
4. teacher temperature ablation。
5. 修复评测脚本完整性校验，保证指标可靠。
6. 若硬件允许，提高训练 `MAX_RESP_LENGTH`，减少训练/评估口径差异。

### Q33. 为什么 OPD 还要 PPO clipping？

即使 reward 是 teacher KL，也是在更新 policy。PPO clipping 限制新旧策略变化，避免一次更新过大导致训练不稳定。尤其 token-level reward 很密集，如果没有约束，可能快速把 student 拉向 teacher 分布，损害探索和原有能力。

### Q34. OPD 和 behavior cloning 的差别？

Behavior cloning 学 teacher 生成的 token 序列，目标是最大化 teacher sample 中真实 token 的 likelihood。OPD 学的是 teacher 在 student 当前状态上的分布，不一定只学 teacher 采样出的单条轨迹；top-K KL 让它对多个候选 token 的概率分布做对齐。

### Q35. 为什么同 family / thinking pattern 重要？

因为 OPD 是 token-level distribution alignment。如果 teacher 和 student 的 reasoning format 不同，同一个语义步骤对应的 token 分布差异可能很大，KL reward 会惩罚 student 的正常路径，导致训练失败或退化。

### Q36. 为什么 GRPO 早期 reward 稀疏？

数学题初始正确率低。GRPO 每个 prompt 采样 N 个 response，只有最终答案正确才有正 reward；如果一个 batch 几乎全错，组内标准化后优势信号很弱或为 0。OPD 每个 token 都有 teacher KL，所以早期更容易有梯度。

### Q37. 你如何判断 OPD 训练确实在工作？

我会看：

- `actor/pg_loss` 是否非零。
- `critic/score/mean` 作为负 KL 是否向 0 靠近。
- `topk/overlap_ratio` 是否上升。
- teacher/student entropy 是否合理。
- response length 和 format error 是否改善。
- 最终离线 benchmark 是否改善。

在我的日志里，step 1/2/10 `actor/pg_loss` 都约 0.26-0.27，step 1119 约 0.22，top-K overlap 从约 0.72 到 0.76。

### Q38. 你怎么解释 `response_length/clip_ratio` 高？

clip ratio 高说明很多 response 到达 max response length，被截断。数学推理任务中这会造成没有最终答案、没有 `\boxed{}`，进而 format error 高。它既影响训练数据质量，也影响评估结果解释。

### Q39. 为什么不用 LoRA？

论文和项目目标是 full-parameter 后训练，尤其要研究 OPD 动态和 token-level alignment。LoRA 可以节省显存，但会改变优化空间和复现口径。我的 1.5B full fine-tuning 在 8x3090 上通过 FSDP/offload 已经可行，所以优先保持 full-param。

### Q40. 如果上线业务模型，你会如何监控 OPD 后模型？

我会监控：

- 任务成功率 / 正确率。
- hallucination rate。
- format compliance。
- safety violation。
- latency 和成本。
- 与 teacher 的分布 drift。
- 用户反馈或人工标注抽检。

并且会用 shadow evaluation / A/B test，避免只看离线 benchmark。

---

## 13. 可以主动抛给面试官的技术点

如果面试官问得比较泛，你可以主动说：

1. **“这个项目里我觉得最关键的是 reward 的粒度。”**  
   GRPO 是 response-level sparse reward，OPD 是 token-level dense reward。

2. **“我复现时最大的工程瓶颈不是参数量，而是 hybrid engine 下 vLLM KV cache 和 FSDP 训练状态抢显存。”**  
   这能体现你理解系统。

3. **“我后来发现评估结果最容易被 max tokens 误导。”**  
   这能体现你不只盯训练 loss。

4. **“我不会说 flash-attn fallback 已经彻底修好，因为它只覆盖了当前 FSDP 路径。”**  
   这能体现你诚实且代码审查细。

5. **“如果继续做，我会做 top-K strategy、teacher temperature、GRPO+OPD hybrid 的 ablation。”**  
   这能体现科研延展能力。

---

## 14. 业务场景模拟回答

### 场景 1：客服小模型蒸馏

面试官问：如果我们有一个强大的客服大模型，但线上只能部署小模型，OPD 怎么用？

回答：

我会把真实客服 query 脱敏后作为 prompt，让当前小模型 student 生成多样 response。然后用大模型 teacher 对这些 student-generated trajectories 做 token-level feedback，同时结合业务规则，比如是否回答完整、是否触发合规禁区、是否引用正确知识库。OPD 的价值是 student 在自己真实会走到的状态上被 teacher 纠正，比单纯拿 teacher 标准答案做 SFT 更能修正 student 的线上错误分布。

### 场景 2：代码模型

面试官问：代码任务里 OPD 有什么用？

回答：

代码任务可以把 stronger code model 作为 teacher，用 student rollout 的代码 token 序列做 token-level distillation，同时结合单元测试 outcome reward。token KL 提供 dense signal，unit test 提供 correctness signal。风险是 teacher 风格不等于可运行正确性，所以一定要把执行反馈纳入 reward。

### 场景 3：安全对齐

面试官问：安全场景能不能只靠 OPD？

回答：

不建议只靠 OPD。Teacher distribution 能迁移安全风格和拒答模式，但安全对齐需要明确 policy constraints。更合理是 OPD + safety classifier/reward + red-team eval。否则 student 可能学到 teacher 的表面表达，而不是严格遵守安全边界。

---

## 15. 一分钟讲“我踩过的坑”

逐字稿：

这个项目里我踩过几个挺典型的复现坑。第一个是环境依赖，verl 的部分路径默认依赖 flash-attn，但我这台机器 CUDA 13.1 和 flash-attn wheel 不匹配，所以我加了 sdpa 和 fallback 方案；后来审查发现 fallback 只能覆盖当前 FSDP 路径，接口不完整。第二个是 24GB 显存适配，vLLM KV cache 和 FSDP training state 会抢显存，所以必须同时调 `gpu_memory_utilization`、response length、rollout 数和 offload。第三个是评估，2048 max tokens 会严重截断数学推理，导致 format error 很高，不能直接说模型不行。第四个是文档证据链，我发现一些复现笔记把 GRPO 和 OPD 指标混在了一起，所以后来我专门写了审查报告，把实际日志、checkpoint、评测文件逐项核对。

---

## 16. 面试中可画的表格：OPD vs GRPO vs SFT vs DPO

| 方法 | 数据来源 | reward / loss | 是否 on-policy | 信号密度 | 优点 | 风险 |
|---|---|---|---|---|---|---|
| SFT | 固定示范数据 | CE loss | 否 | token-level 但来自固定答案 | 稳定、便宜 | distribution mismatch |
| DPO | preference pair | preference objective | 否 | pair-level | 不需要在线 rollout | 依赖偏好数据质量 |
| GRPO | student rollout | outcome reward group norm | 是 | response-level sparse | 对最终任务目标直接 | 早期 reward 稀疏 |
| OPD | student rollout + teacher | token-level KL reward | 是 | token-level dense | 每个 token 有 teacher signal | teacher 兼容性和成本 |

---

## 17. 如果面试官要求你现场推导

### 17.1 从 KL 到 reward

```text
KL(pi_s || pi_t) = sum_a pi_s(a|s_t) [log pi_s(a|s_t) - log pi_t(a|s_t)]
reward_t = - KL(pi_s || pi_t)
```

Top-K 近似：

```text
K_t = TopK(pi_s(.|s_t))
reward_t ~= - sum_{a in K_t} w_a [log pi_s(a|s_t) - log pi_t(a|s_t)]
```

其中 `w_a` 可以是 `student_p` softmax 权重。

### 17.2 从 reward 到 PPO loss

```text
A_t,a = reward_t,a
ratio_t,a = exp(log pi_new(a|s_t) - log pi_old(a|s_t))
L = - mean_t sum_a min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```

解释：

- top-K 情况下 `a` 是 K 个候选 token。
- response mask 只保留 response 部分，不训练 prompt padding。
- PPO clipping 限制 policy update。

---

## 18. 准备面试前的最后检查清单

### 18.1 必须背熟的数字

- 硬件：8 x RTX 3090 24GB。
- OPD 训练：1119 steps，约 13h44m。
- Student：DeepSeek-R1-Distill-Qwen-1.5B。
- Teacher：JustRL-DeepSeek-1.5B。
- Dataset：DAPO-Math-17k。
- 训练 response length：2048。
- 训练 N responses：2。
- Top-K：16。
- `topk/overlap_ratio`：约 0.72 -> 0.76。
- 离线评估 N：16。
- 16384 评测结果：AIME24 28.1%，AIME25 24.0%，AMC23 72.2% mean_score。

### 18.2 必须能解释的词

- on-policy vs off-policy。
- reverse KL。
- top-K approximation。
- token-level reward。
- GRPO group advantage。
- PPO clipping。
- FSDP。
- vLLM KV cache。
- colocation / hybrid engine。
- `enable_thinking` / chat template。
- format error。
- mean_score vs best_score。

### 18.3 不能过度宣称

- 不说完全复现论文所有规模。
- 不说 OPD 绝对优于 GRPO。
- 不说 16384 评测就是训练时长上下文能力。
- 不说 flash-attn fallback 已经完全解决。
- 不说 SFT demo 是完整 SFT 复现。

---

## 19. 你可以问面试官的问题

面试后半段可以反问：

1. 你们团队现在后训练更关注 reward model、preference data，还是 RL 系统吞吐？
2. 业务里 teacher/student 蒸馏更多是离线数据蒸馏，还是会用线上 student rollout 做 on-policy 修正？
3. 你们如何处理数学/代码这种长推理任务的评估长度和 pass@k 口径？
4. 在资源受限的训练环境里，你们更常用 FSDP、DeepSpeed ZeRO，还是 Megatron 系列并行？
5. 如果一个小模型在业务分布上经常走到 teacher 没见过的状态，你们会更倾向做数据增强、RL，还是 on-policy distillation？

这些问题能把对话带到后训练、评估和系统落地上。

---

## 20. 最终推荐叙事

你这个项目最好的叙事不是“我跑出了一个分数”，而是：

> 我围绕一篇 LLM 后训练论文，完整理解了 OPD 从算法目标、token-level reward、top-K KL、PPO update 到 verl 分布式训练系统的链路；在 8x3090 的受限环境下做了可行性适配，完成 1.5B OPD 训练和离线数学评估；同时从复现工程角度审查了环境、评测脚本和文档口径中的问题。这个项目让我对 LLM 后训练里算法目标、系统资源和评估可信度三者之间的关系有了比较具体的经验。

如果只记一句：

> 这个项目展示的是我能把 LLM 后训练论文读懂、跑通、拆解训练信号，并且能诚实分析复现结果和工程风险。

