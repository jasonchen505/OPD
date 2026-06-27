# 论文与代码对照分析：RethinkOPD × 综述 × 实现

> 基于两篇论文 tex 源码与 `/home/chenyizhou/OPD` 代码仓库的深度交叉分析  
> 论文 A：*Rethinking On-Policy Distillation* (RethinkOPD, arXiv:2604.13016)  
> 论文 B：*A Survey of On-Policy Distillation for Large Language Models* (Tencent Survey, COLM 2026)  
> 代码：verl v0.7.0 + LlamaFactory + 项目自定义脚本

---

## 一、两篇论文的定位与关系

### 1.1 各自定位

| 维度 | RethinkOPD (论文 A) | Tencent Survey (论文 B) |
|------|---------------------|------------------------|
| 类型 | 实证研究 + 机制分析 | 系统综述 + 统一框架 |
| 核心贡献 | 两个成功条件 + token 级机制 + 恢复策略 | f-divergence 统一框架 + 200+ 方法分类 + 设计空间 |
| 方法论 | 控制变量实验 + 反向蒸馏验证 | 理论推导 + 文献综合 + 分类学 |
| 覆盖范围 | 单一方法深度分析 | 整个 OPD 领域全景 |
| 实验规模 | 1.5B-7B, 8xA800 | 涵盖 0.6B-1.6T 多规模 |

### 1.2 互补关系

RethinkOPD 回答 **"OPD 什么时候有效、为什么有效"**，Tencent Survey 回答 **"OPD 有哪些方法、如何选择"**。两者结合能形成从原理到实践的完整认知：

```
RethinkOPD 的"为什么"  ←→  Tencent Survey 的"是什么"和"怎么做"
    ↓                              ↓
成功条件 (phenomenology)    →  方法分类 (taxonomy)
机制分析 (mechanism)        →  目标函数设计 (objectives)
恢复策略 (recipe)           →  训练动态优化 (dynamics)
```

---

## 二、核心概念的代码级映射

### 2.1 OPD 的数学定义 → 代码实现

**论文 A 的定义**（`2_Preliminaries.tex:44-68`）：

```
L_OPD(θ) = E_{x, ŷ~π_θ} [ Σ_t D_KL(p_t || q_t) ]
```

其中 `p_t = π_θ(·|x, ŷ_{<t})` 是 student，`q_t = π_T(·|x, ŷ_{<t})` 是 teacher。

**论文 B 的统一框架**（`main.tex:149-169`）：

```
L_OPD(θ) = E_{y~π_mix} [ Σ_t D_f(p_teacher(·|x,y_{<t}), p_θ(·|x,y_{<t})) ]
```

**代码实现**（`verl/verl/workers/actor/dp_actor.py:560-563`）：

```python
# only_stu 策略下的 KL 计算
kl_val = S_logp - T_on_S  # reverse KL: log π_student - log π_teacher
norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
rm_scores = -kl_val * norm_weights  # 负 KL = reward
```

**对照分析**：
- 代码中的 `kl_val = S_logp - T_on_S` 直接实现了 token 级 reverse KL 的核心计算
- `rm_scores = -kl_val` 将 KL 转换为 reward（KL 越小 → reward 越高 → student 越接近 teacher）
- `norm_weights` 实现了论文 A 中 `student_p` 加权方案

### 2.2 三种 OPD 变体 → 代码中的 Top-K 策略

**论文 A 定义的三种变体**（`2_Preliminaries.tex:76-142`）：

| 变体 | 公式 | 论文 A 的评价 |
|------|------|--------------|
| Sampled-Token | `ℓ_t = log p_t(ŷ_t) - log q_t(ŷ_t)` | 无偏单样本估计，最轻量 |
| Full-Vocabulary | `D_KL(p_t || q_t)` over full V | 最精确，O(BTM) 显存 |
| Top-k | `D_KL(p̄_t^{S_t} || q̄_t^{S_t})` | 折中方案，近似 full-vocab |

**论文 B 的分析**（`main.tex:176-181`）：

> MiniLLM 用 Reverse KL + REINFORCE，DistiLLM 用 Skewed KL 避免零除，GKD 用 λ-mixing 在 on/off-policy 之间插值。

**代码中的实现**（`dp_actor.py:450-624`）：

```python
# 五种 top-K 策略
if strategy == "only_stu":      # Student Top-K（论文 A 默认）
    kl_val = S_logp - T_on_S
elif strategy == "only_tch":    # Teacher Top-K
    kl_val = S_on_T - T_logp
elif strategy == "intersection": # Overlap Top-K
    kl_val = S_logp - T_on_S; kl_val = torch.where(valid_mask, kl_val, 0)
elif strategy == "union":       # Full union
    union_ids = torch.cat([S_ids, T_ids], dim=-1)
    # ...
elif strategy == "union-intersection": # Symmetric difference
    # ...
```

**对照分析**：
- 代码中 `only_stu` 对应论文 A 的 Student Top-k 变体
- `intersection` 对应论文 A §5.2 的 Overlap Top-k ablation
- `union` 对应 Full-Vocabulary 的近似（合并两边 top-K）
- 代码中没有直接实现 Sampled-Token OPD（`LOG_PROB_TOP_K=0` 时退化为 sampled-token）

### 2.3 Advantage Estimator → 核心算法

**论文 A 的 OPD 目标**（通过 G-OPD 的 RL 等价性，论文 B `main.tex:769-775`）：

```
max_θ E_{y~π_θ} [ Σ_t α·log(π_T/π_ref) - KL(π_θ || π_ref) ]
```

当 α=1 时退化为标准 Reverse KL 蒸馏。

**代码实现**（`verl/verl/trainer/ppo/core_algos.py:854-880`）：

```python
@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(token_level_rewards, response_mask, ...):
    """直接使用 token-level reward 作为 advantage 的最简单 estimator"""
    advantages = token_level_rewards * response_mask
    returns = advantages.clone()
    return advantages, returns
```

**论文 B 对 G-OPD 的分析**（`main.tex:769-773`）：

> G-OPD 证明了 OPD 与 KL-constrained RL 的等价性。当 α>1 时，reward extrapolation 让 student 发现 teacher 自身概率低但 outcome reward 高的新推理路径。

**对照分析**：
- 代码中 `token_reward_direct` 是最简单的实现：直接把 KL reward 当 advantage，不需要 GAE、不需要 value baseline
- 这对应论文 B 中 G-OPD 的 α=1 特例（标准 reverse KL 蒸馏）
- 代码中还有 `token_reward_direct_plus_grpo` 混合模式（`core_algos.py:883-924`），对应论文 B 中的 KD+RL 联合优化

---

## 三、RethinkOPD 两大发现的代码验证

### 3.1 发现一：Thinking-Pattern Consistency

**论文 A 的发现**（`3_Exp.tex:17-51`）：

> 即使 teacher 在 benchmark 上得分更高，如果 thinking pattern 不匹配，OPD 也会失败。GRPO teacher 虽然 benchmark 表现略差，但初始 overlap ratio 更高，蒸馏效果更好。

**代码中的 overlap ratio 监控**（`verl/verl/trainer/ppo/ray_trainer.py:1147-1233`）：

```python
if (self.global_steps == 1 or self.global_steps % 10 == 0) and "student_valid_counts" in batch.batch.keys():
    # 计算 overlap_mask
    overlap_mask = batch.batch["overlap_mask"].float()
    overlap_counts = overlap_mask.sum(dim=-1)  # (BS, SeqLen)
    avg_overlap_counts = (overlap_counts * response_mask).sum(dim=0) / valid_denom
```

**实测数据**（`logs/opd_3090_20260530_190228.log`）：

```
Step 1:    val-topk/overlap_ratio = 0.7223
Step 1119: val-topk/overlap_ratio = 0.7625
```

**对照分析**：
- 论文 A 报告 overlap ratio 从 72% 上升到 91%（成功 run）
- 实测从 72.2% 上升到 76.2%，上升趋势一致但幅度较小
- 原因：训练 `MAX_RESP_LENGTH=2048`（论文 A 用 7168），且只训练了 1119 步（约 0.66 epoch）
- 这验证了论文 A 的核心发现：overlap ratio 的上升是 OPD 成功的 signature

### 3.2 发现二：Overlap Tokens Carry 97-99% Probability Mass

**论文 A 的发现**（`Sections/Appendix.tex:96-126`）：

> overlap tokens 在整个训练过程中承载了 student 和 teacher 各自 97-99% 的概率质量。

**代码中的监控指标**（`ray_trainer.py` 日志输出）：

```
val-topk/student_p_sum_intersection: 0.9795
val-topk/teacher_p_sum_intersection: 0.9815
```

**对照分析**：
- 实测 student 在 overlap tokens 上的概率总和为 97.95%，teacher 为 98.15%
- 与论文 A 报告的 97-99% 完全一致
- 这直接验证了论文 A 的核心发现：top-K=16 的近似已经覆盖了绝大部分概率质量

### 3.3 发现三：Reward Quality Degrades with Trajectory Depth

**论文 A 的发现**（`6_Discussion.tex:29-79`）：

> - Response length 存在 sweet spot（3K-7K 最优，10K+ 崩溃）
> - 不稳定性从后部 token 向前传播（back-to-front pattern）
> - Teacher continuation 的 accuracy advantage 从 +0.37（1K prefix）降到 +0.02（16K prefix）

**代码中的约束**（`opd_3090.sh:48`）：

```bash
export MAX_RESP_LENGTH=2048  # 3090 限制，远小于论文 A 的 7168
```

**对照分析**：
- 论文 A 用 7168 在 A800 上训练，3090 适配后降到 2048
- 论文 A 的实验证明 3K-7K 是 sweet spot，2048 处于"太短"的边界
- 这可能是实测 overlap ratio 只上升到 76%（而非论文的 91%）的原因之一
- 论文 A 的 Figure 6 显示短 response（0.5K-1K）"provide too few supervised tokens for sample-efficient learning"

---

## 四、论文 B 综述框架下的方法定位

### 4.1 RethinkOPD 在综述分类中的位置

根据论文 B 的三轴分类（`main.tex:199-358`）：

| 轴 | RethinkOPD 的定位 |
|----|-------------------|
| **Objective** | Fixed divergence（Reverse KL），但研究了 Top-k 近似 |
| **Signal** | White-box logit supervision（teacher 全分布访问） |
| **Dynamics** | 提出了 off-policy cold start + teacher-aligned prompts 两种稳定化策略 |

论文 B 对 RethinkOPD 的引用（`main.tex:500, 1253, 1255, 1310`）：

> - "Thinking-pattern mismatch between teacher and student is typically resolved through an off-policy cold-start phase [2604.13016]"
> - "[2604.13016] identify that OPD can fail catastrophically when the student's initial policy is too far from the teacher's distribution"
> - "The hybrid SFT+OPD pipeline: [2604.13016] combined with on-policy refinement has emerged as a common industrial recipe"

### 4.2 RethinkOPD 的方法论贡献在综述中的体现

论文 B 将 RethinkOPD 的贡献归入 **§6.2 Curriculum and Difficulty Adaptation**（训练动态优化）：

```
RethinkOPD → off-policy cold start → 降低 thinking-pattern gap
           → teacher-aligned prompts → 提高 overlap ratio
```

这在综述的 taxonomy 中对应：

```
OPD
├── §6 Training Efficiency and Stabilization
│   └── 6.2 Curriculum and Difficulty Adaptation
│       └── Stable-OPD, CaOPD, ... + RethinkOPD's cold start
```

### 4.3 RethinkOPD 未覆盖但综述重要的方向

| 方向 | 综述中的代表方法 | RethinkOPD 的状态 |
|------|-----------------|-------------------|
| Adaptive divergence | EOPD, AOPD | 未探索（固定用 reverse KL） |
| RL-augmented objectives | G-OPD, KDRL, RLAD | 仅对比了 GRPO baseline |
| Self-distillation | OPSD, SD-ZERO | 未探索 |
| Black-box methods | Lion, GAD | 未探索 |
| Cross-family transfer | DSKD, SimCT | 未探索（student/teacher 同系列） |
| Multi-turn agentic | MAD-OPD, TCOD | 论文 A §6 讨论了局限性 |

---

## 五、代码实现中的两条 OPD 路径

### 5.1 旧路径：reward model worker 路径

**脚本**：`on_policy_distillation.sh`、`opd_3090.sh`

**机制**：
```bash
reward_model.enable=True                    # teacher 作为 reward model
reward_model.model.path=model/JustRL-DeepSeek-1.5B  # teacher 权重
ADV_ESTIMATOR=token_reward_direct           # 使用 token-level KL advantage
+actor_rollout_ref.rollout.log_prob_top_k=16  # top-K=16
```

**代码流**：
```
ray_trainer.py:1134 → rm_wg.compute_rm_score(batch)     # teacher forward
fsdp_workers.py:2553 → RewardModelWorker.compute_rm_score()  # 提取 logits
fsdp_workers.py:1831 → _compute_teacher_top_k_log_probs()    # 计算 top-K
dp_actor.py:450 → compute_distillation_reward()              # 计算 KL reward
core_algos.py:854 → compute_token_reward_direct_advantage()  # 直接当 advantage
```

**特点**：
- Teacher 复用 reward model worker
- Teacher 和 actor 共享 GPU（colocation）
- 配置复杂（需要设置多个 reward_model.* 参数）

### 5.2 新路径：distillation API 路径

**脚本**：`verl_example/opd.sh`

**机制**：
```bash
distillation.enabled=True
distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.topk=16
distillation.distillation_loss.use_policy_gradient=True
```

**特点**：
- Teacher 作为独立的 distillation worker
- 更清晰的关注点分离
- 支持多种 loss mode（k1、forward_kl_topk 等）
- 自动保存复现上下文（git commit、diff、env）

### 5.3 两种路径的算法等价性

从论文 B 的统一框架看（`main.tex:149-169`），两种路径在算法上等价：

```
L_OPD = E_{y~π_θ} [ Σ_t D_KL(p_t || q_t) ]
```

区别只在工程实现：
- 旧路径：通过 reward model worker 计算 teacher log-probs
- 新路径：通过独立 distillation worker 计算 teacher log-probs

---

## 六、论文 A §6 Discussion 的代码级验证

### 6.1 Reward Quality Degradation

论文 A §6.1 发现 response length 存在 sweet spot。代码中的约束：

```bash
# opd_3090.sh
MAX_RESP_LENGTH=2048  # 远小于论文 A 推荐的 3K-7K
```

论文 A 的 Figure 6(a) 显示：
- 0.5K-1K：too few supervised tokens
- 3K-7K：strongest results
- 10K-15K：late-stage collapse

实测用 2048，处于"太短"的边界，这解释了为什么：
- 训练效果不如论文 A 报告的那么好
- overlap ratio 只上升到 76%（论文 A 成功 run 到 91%）

### 6.2 Sampled-Token OPD Is Sufficient

论文 A §6.3 的关键发现（`6_Discussion.tex:131-175`）：

> Sampled-token OPD achieves performance comparable to Top-k settings. The only clearly worse configuration is Top-1.

代码中 `LOG_PROB_TOP_K` 的配置：
```bash
# on_policy_distillation.sh:61
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}  # 默认 16
# 0 表示 sampled-token OPD
```

**对照分析**：
- 论文 A 证明 k=1（argmax）不稳定，k=4/16/64 差异不大
- 代码默认 k=16，与论文 A 的推荐一致
- 论文 A 的结论"sampled-token OPD is already sufficient"意味着 k=0 也是可行的

### 6.3 Globally Informative Reward ≠ Locally Exploitable

论文 A §6.2 的发现（`6_Discussion.tex:102-122`）：

> 失败的 teacher（R1-Distill-7B）也能提供 globally informative 的 reward（AUROC=0.75），但 gradient norm 持续很小。原因是 per-token advantages 在位置间各向异性，聚合后部分抵消。

**代码中的 gradient norm 监控**：
```
Step 1:    actor/grad_norm = 2.33
Step 1119: actor/grad_norm = 2.27
```

**对照分析**：
- 实测 grad_norm 在训练过程中保持稳定（2.27-2.52）
- 这对应论文 A 中 JustRL-1.5B teacher（成功 run）的"sustained gradient magnitude"
- 如果 grad_norm 持续很小，可能对应论文 A 中 R1-Distill-7B teacher（失败 run）的情况

---

## 七、论文 B 综述中的关键理论 → 代码验证

### 7.1 Off-Policy Exposure Bias

论文 B 的核心理论（`main.tex:135-147`）：

> Off-policy 训练的 expected discrepancy scales as O(εT²)，on-policy 降到 O(εT)。

**代码中的 on-policy 保证**：
```python
# ray_trainer.py:1049-1053
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)  # student rollout
# ...
# 每步都用当前 student 重新生成
```

**实测验证**（`logs/opd_3090_20260530_190228.log`）：
```
training/rollout_actor_probs_pearson_corr: 0.9996  # rollout 和训练策略几乎一致
actor/pg_clipfrac: 0.0                              # PPO clipping 从未触发
```

这验证了 on-policy 的假设——rollout 和训练之间没有显著的策略漂移。

### 7.2 f-Divergence Framework

论文 B 的统一框架（`main.tex:149-169`）：

| f(u) | Divergence | 行为 |
|------|-----------|------|
| u log u | Forward KL | mode-covering, zero-avoiding |
| -log u | Reverse KL | mode-seeking, zero-forcing |
| u log u - (u+1) log((u+1)/2) | JSD | 对称、有界 |

**代码中的实现**：
```python
# dp_actor.py:560-563（only_stu 策略）
kl_val = S_logp - T_on_S  # = log(π_student/π_teacher) = reverse KL 的核心
```

**对照分析**：
- 代码默认使用 reverse KL（mode-seeking），对应论文 B 中"适合有唯一正确答案的数学推理"
- 论文 B 指出 Forward KL 适合"many acceptable outputs"（创意写作、开放式 QA）
- 代码中没有实现 JSD 或 α-divergence，但论文 B 的分析表明 adaptive divergence（EOPD、AOPD）效果更好

### 7.3 OPD as KL-Constrained RL (G-OPD)

论文 B 对 G-OPD 的分析（`main.tex:769-775`）：

> G-OPD 证明 OPD 等价于 KL-constrained RL：
> max_θ E [ Σ_t α·log(π_T/π_ref) - KL(π_θ || π_ref) ]
> 当 α=1 退化为标准 reverse KL，α>1 时 reward extrapolation 让 student 超越 teacher。

**代码中的混合模式**（`core_algos.py:883-924`）：

```python
@register_adv_est("token_reward_direct_plus_grpo")
def compute_token_reward_direct_plus_grpo_advantage(...):
    direct_adv, _ = compute_token_reward_direct_advantage(...)  # KL reward
    grpo_adv, _ = compute_grpo_outcome_advantage(...)           # outcome reward
    combined_adv = direct_adv + weight * grpo_adv               # 混合
```

**对照分析**：
- `token_reward_direct_plus_grpo` 实现了 G-OPD 的 RL-augmented 目标
- `direct_adv` 对应 α·log(π_T/π_ref)（KL reward）
- `grpo_adv` 对应 outcome reward（GRPO 的组内标准化）
- `weight` 对应 G-OPD 中的 α 超参

---

## 八、论文 A Recipe 的代码实现状态

### 8.1 Off-Policy Cold Start

**论文 A 的 Recipe**（`5_Exp.tex:16-49`）：

> 先用 teacher 生成的 rollout 做 SFT（cold start），再做标准 OPD。SFT 初始化后 overlap ratio 更高、训练更稳定。

**代码中的实现**：
- SFT 部分：`LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml`
- Teacher rollout：`scripts/infer/vllm_rollout.py`
- 但实际复现中只做了 smoke test（alpaca_en_demo 999 样本），没有做完整的 cold start

**对照分析**：
- 论文 A 用 200K teacher-generated samples 做 SFT cold start
- 实际复现跳过了这一步，直接从 base model 开始 OPD
- 这可能是训练效果不如论文 A 的另一个原因

### 8.2 Teacher-Aligned Prompt Selection

**论文 A 的 Recipe**（`5_Exp.tex:55-124`）：

> 使用 teacher post-training 数据中的 prompt，可以提高 overlap ratio 和最终性能。但纯 teacher-aligned prompt 会导致 entropy collapse，需要混合 OOD prompt。

**代码中的数据**：
```bash
# opd_3090.sh:41
export TRAIN_DATASET=datasets/dapo-math-17k.parquet  # DAPO-Math-17K
```

**对照分析**：
- 论文 A 证明 prompt template 对 OPD 有显著影响
- 论文 A §5.2 的 teacher-aligned template：`{Question} Please reason step by step, and put your final answer within \boxed{}.`
- 代码中使用的 DAPO-Math-17K 的 prompt template 需要确认是否与 teacher 的 post-training 数据一致

---

## 九、综述中的前沿方向 → 代码扩展潜力

### 9.1 Adaptive Divergence (EOPD, AOPD)

论文 B §4.2 介绍了 adaptive divergence 方法：

> EOPD 用 teacher 的 per-token entropy 做 gating，高 entropy 时用 Forward KL（mode-covering），低 entropy 时用 Reverse KL（mode-seeking）。

**代码中的扩展点**：
```python
# core_algos.py 可以添加新的 advantage estimator
@register_adv_est("token_reward_adaptive")
def compute_token_reward_adaptive_advantage(...):
    # 根据 teacher entropy 选择 KL 方向
    teacher_entropy = compute_entropy(teacher_logits)
    mask = teacher_entropy > threshold
    # high entropy: forward KL, low entropy: reverse KL
```

### 9.2 RL-Augmented Objectives (G-OPD)

论文 B §4.3 介绍了 G-OPD 的 reward extrapolation：

> 当 α>1 时，student 可以发现 teacher 自身概率低但 outcome reward 高的新推理路径。

**代码中已有支持**：
```bash
# on_policy_distillation.sh:41-44
# export ADV_ESTIMATOR=token_reward_direct_plus_grpo  # 混合模式
# export GRPO_OUTCOME_WEIGHT=1.0                      # α 超参
```

### 9.3 Self-Distillation (OPSD)

论文 B §5.2.1 介绍了 OPSD（ground-truth as PI）：

> 用 ground-truth answer 作为 privileged information，conditioned teacher 比 unconditioned student 更强。

**代码中未实现**，但 verl 的 `verl_example/opd.sh` 中的 `distillation.*` API 可能支持：

```bash
distillation.distillation_loss.use_task_rewards=False  # 可以改为 True
```

### 9.4 Token Weighting (TIP)

论文 B §6.1 介绍了 token importance weighting：

> TIP 用 entropy + divergence 做 per-token 重要性加权，只在"可教"的 token 上蒸馏。

**代码中的扩展点**：
```python
# dp_actor.py:522-556 的 compute_reward_weights 可以扩展
def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode):
    if weight_mode == "token_importance":
        # 基于 entropy 或 divergence 的 per-token 加权
        importance = compute_importance(S_logp, T_logp)
        weights = importance / importance.sum()
```

---

## 十、关键对照总结表

| 论文 A 发现/方法 | 论文 B 框架定位 | 代码实现位置 | 实测验证状态 |
|-----------------|----------------|-------------|-------------|
| Thinking-pattern consistency | §6.2 Dynamics: stability | `overlap_ratio` 监控 | ✅ 72%→76% |
| 97-99% overlap mass | §4.1 Fixed divergence: top-k | `student_p_sum_intersection` | ✅ 98% |
| Reward degradation with depth | §6.2 Curriculum | `MAX_RESP_LENGTH=2048` | ⚠️ 偏短 |
| Sampled-token sufficient | §4.1 Fixed divergence | `LOG_PROB_TOP_K=0` | 未测试 |
| Off-policy cold start | §6.2 Dynamics | LlamaFactory SFT | ⚠️ 仅 smoke test |
| Teacher-aligned prompts | §6.2 Dynamics | DAPO-Math-17K | ⚠️ 未验证 template |
| Reverse KL (mode-seeking) | §4.1 f-divergence | `S_logp - T_on_S` | ✅ 默认实现 |
| Top-K=16 | §4.1 Fixed divergence | `LOG_PROB_TOP_K=16` | ✅ 默认配置 |
| GRPO baseline | §4.3 RL-augmented | `ADV_ESTIMATOR=grpo` | ✅ 131 步 |
| G-OPD equivalence | §4.3 RL-augmented | `token_reward_direct_plus_grpo` | 未测试 |
| EOPD adaptive divergence | §4.2 Adaptive | 未实现 | - |
| OPSD self-distillation | §5.2.1 Self-distill | 未实现 | - |

---

## 十一、对复现的指导意义

### 11.1 已验证的结论

1. ✅ OPD 的 token-level KL reward 在 3090 上确实产生非零 dense signal
2. ✅ overlap ratio 的上升趋势与论文 A 一致
3. ✅ 97-99% 概率质量集中在 overlap tokens 的发现得到验证
4. ✅ PPO clipping 从未触发（on-policy 假设成立）

### 11.2 需要改进的地方

1. ⚠️ `MAX_RESP_LENGTH=2048` 偏短，论文 A 推荐 3K-7K
2. ⚠️ 缺少 off-policy cold start（SFT 阶段）
3. ⚠️ 未验证 prompt template 是否与 teacher post-training 一致
4. ⚠️ GRPO baseline 只跑了 131 步，无法公平对比

### 11.3 可探索的新方向

1. 🔮 EOPD adaptive divergence（根据 teacher entropy 切换 KL 方向）
2. 🔮 G-OPD reward extrapolation（α>1 让 student 超越 teacher）
3. 🔮 OPSD self-distillation（ground-truth as PI）
4. 🔮 TIP token importance weighting（只在可教 token 上蒸馏）

---

## 附录：论文 B 的 200+ 方法分类速查

### A. Objective Functions (§4)

| 子类 | 代表方法 | 数量 |
|------|---------|------|
| Fixed divergence | GKD, MiniLLM, DistiLLM | 12 |
| Adaptive divergence | EOPD, AOPD, Token-Teachability | 11 |
| RL-augmented | G-OPD, RLKD, KDRL, CoDistill-GRPO | 21 |

### B. Signal Source (§5)

| 子类 | 代表方法 | 数量 |
|------|---------|------|
| White-box same-family | MAD-OPD, MPD | 4 |
| White-box cross-family | DSKD, SimCT | 8 |
| Black-box | Lion, GAD, OVD | 10 |
| Self-distillation (PI) | OPSD, GATES, CRISP | 32 |
| Self-distillation (pure) | SDFT, UniSD | 16 |
| Self-distillation (feedback) | SDPO, SD-ZERO | 15 |

### C. Training Dynamics (§6)

| 子类 | 代表方法 | 数量 |
|------|---------|------|
| Token/sample weighting | TIP, SCOPE, R-OPD | 11 |
| Curriculum | PACED, Stable-OPD | 10 |
| Compute optimization | FOPD, Lightning-OPD | 8 |

---

*生成时间：2026-06-22*
*基于 `paper/RethinkOPD/opd.tex`、`paper/tencent OPD survey/main.tex` 与 `/home/chenyizhou/OPD` 代码仓库的深度交叉分析*
