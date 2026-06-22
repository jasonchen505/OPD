# Rethink OPD 技术面试五类能力应对手册

面向：LLM 算法实习技术面试  
项目：Rethinking OPD 复现与 8x3090 后训练适配  
远程项目路径：`/home/chenyizhou/OPD`  
配套文档：`OPD_INTERVIEW_PREP.md`、`REPRODUCTION_REVIEW.md`、`reproduction_learning_notes/`

这份文档专门对应技术面试常见的五类考察能力：

1. 底层原理与方法设计理解
2. 实验和方案验证能力
3. 问题定位与排查能力
4. 工程落地能力
5. 业务与实际场景理解

核心目标不是“背概念”，而是能把项目讲成一条完整链路：

```text
为什么需要这个方法
  -> 方法怎么设计
  -> 我如何在受限硬件上跑起来
  -> 我怎么证明有效
  -> 遇到问题如何定位
  -> 如果进生产要怎么落地、监控、回滚和创造业务价值
```

---

## 0. 面试回答总原则

### 0.1 每个回答都用四段式

面对任何深挖题，建议用这个结构：

1. **问题背景**：这个方法或工程设计解决什么真实问题。
2. **核心机制**：它具体怎么做，最好能落到公式、代码路径、配置或日志。
3. **局限与风险**：不要只讲优点，要主动讲不适用条件。
4. **改进与验证**：如果继续做，如何证明或修复。

示例：

> OPD 解决的是 GRPO 等 outcome-level RL 在数学推理早期 reward 稀疏、以及传统 off-policy distillation 存在 distribution mismatch 的问题。它让 student 自己 rollout，再用 teacher 在 student 访问到的 token 状态上给 dense KL reward。我的复现里通过 `token_reward_direct` 直接把 token reward 当 advantage，top-K=16，默认 only_stu 策略。但它也有局限，比如 teacher/student thinking pattern 不兼容时会错对齐，teacher 成本高，且我的训练 response length 是 2048，离线评估是 16384，口径要说明。继续改进我会做 top-K、teacher temperature 和 OPD+GRPO hybrid ablation。

### 0.2 面试中要主动区分三类话

| 类型 | 能不能说 | 示例 |
|---|---|---|
| 已验证事实 | 可以直接说 | “我完成了 1119 step OPD 训练，日志里 step 1/2 `actor/pg_loss` 约 0.26/0.27。” |
| 合理分析 | 要说明是分析 | “我认为 2048 评估差主要来自截断，因为 format error 非常高。” |
| 未做实验 | 不能说成事实 | “我还没有做完整 top-K ablation，如果继续做我会这样验证。” |

### 0.3 这个项目最好的能力定位

不是“我跑出了一个分数”，而是：

> 我围绕 OPD 这个 LLM 后训练方法，理解了 token-level dense reward、top-K KL、PPO update 和 verl hybrid engine；在 8x3090 24GB 受限环境下完成 1.5B 可行性复现；并通过日志、评测和代码审查定位了工程复现中的关键问题。

---

## 1. 能力一：底层原理与方法设计理解

### 1.1 面试官真正想看什么

面试官不是只想听：

- OPD 是 On-Policy Distillation。
- GRPO 是 Group Relative Policy Optimization。
- KL 是分布距离。

他更想看：

- 这个方法解决什么后训练痛点。
- 为什么 reward 要这样设计。
- 为什么用 top-K 近似。
- 为什么这个方法可能失败。
- 如果要改进，你知道哪些方向。

### 1.2 你的核心回答框架

可以先用一句话总述：

> OPD 试图解决两个问题：一是传统 off-policy distillation 只在 teacher 或固定数据分布上训练，student 部署时会走到自己的状态分布；二是 GRPO 这类 outcome-level RL 在数学推理早期 reward 太稀疏。OPD 让 student 自己 rollout，再让 teacher 在这些 student-visited states 上提供 token-level KL reward。

然后展开：

1. **为什么 on-policy**：student 学自己会走到的状态，而不是只模仿 teacher demonstrations。
2. **为什么 token-level**：数学推理的最终对错很稀疏，但中间每个 token 都能通过 teacher 分布给信号。
3. **为什么 KL reward**：让 student 的下一个 token 分布向 teacher 对齐，而不是只学单条 teacher sample。
4. **为什么 top-K**：full vocabulary KL 成本太高，用高概率 token 集近似。
5. **为什么 still PPO**：虽然 reward 来自 distillation，但 policy update 仍需要限制策略变化，PPO clipping 提供稳定性。

### 1.3 标准回答：OPD 解决什么问题

面试官问：

> 你说 OPD 有用，它到底解决什么问题？

回答模板：

> 我理解 OPD 主要解决两个问题。第一个是传统蒸馏的 distribution mismatch。普通 KD 或 SFT 常常让 student 拟合 teacher 生成好的固定答案，但 student 真正上线时是沿着自己的 policy 生成，它会进入 teacher demonstration 没覆盖的状态。OPD 让 student 当前策略先 rollout，然后在这些 student 自己访问到的状态上，让 teacher 给 token-level 分布反馈，这样训练分布更接近部署分布。  
> 第二个是 RL 中 outcome reward 稀疏的问题。比如 GRPO 在数学题里只有最后答案对错，1.5B 模型早期大部分 batch 全错，policy gradient 信号很弱。OPD 用 teacher log-prob 做每个 token 位置的 dense reward，即使最后答案错，中间 token 也能收到 teacher 分布信号。

如果追问：

> 那是不是 OPD 一定比 GRPO 好？

回答：

> 不能这么说。OPD 的优势是 dense reward，但它依赖 teacher 的质量和 teacher/student thinking pattern 兼容。如果 teacher 不可靠或格式不兼容，token-level KL 会把 student 拉向错误方向。GRPO 的 outcome reward 虽稀疏，但目标更直接。因此更合理的方向是根据任务混合 token-level teacher reward 和 outcome reward，而不是绝对说谁更好。

### 1.4 标准回答：为什么 reward 设计成负 KL

面试官问：

> 为什么用 `-KL(student || teacher)` 做 reward？

回答模板：

> 因为我们想让 student 在它自己访问到的状态上，把下一个 token 分布对齐 teacher。KL 越小代表 student 和 teacher 越一致，所以取负 KL 作为 reward，越接近 0 越好。这里用的是 reverse KL 形式 `KL(pi_student || pi_teacher)`，更偏 mode-seeking，会惩罚 student 把概率放在 teacher 低概率 token 上。  
> 代码实现里不会对全词表算 KL，而是取 top-K token set 近似。我的复现配置是 `LOG_PROB_TOP_K=16`、`TOP_K_STRATEGY=only_stu`、`REWARD_WEIGHT_MODE=student_p`。也就是说先取 student 当前最相信的 K 个 token，再看 teacher 对这些 token 的 log-prob，用 student probability 做权重。

如果追问：

> 为什么不是正向 KL？

回答：

> 正向 KL `KL(teacher || student)` 更鼓励 student 覆盖 teacher 的所有高概率模式，mode-covering 更强；reverse KL 更关注 student 当前放概率的位置是否被 teacher 认可，mode-seeking 更强。OPD 的 on-policy 设定里，我们重点纠正 student 自己会选择的分布，所以 reverse KL 更自然。但局限是如果 teacher 的关键 token 不在 student top-K，`only_stu` 策略可能看不到，这就是 top-K strategy 的 trade-off。

### 1.5 标准回答：为什么用 top-K 近似

面试官问：

> 你说 top-K 是核心，那为什么要 top-K？K 怎么选？

回答模板：

> full-vocab KL 对每个 response token 都要在整个词表上算 log-softmax 和聚合，Qwen2 vocab 有十几万，长响应下计算和显存都很贵。top-K 的思想是高概率 token 集通常承载主要概率质量，所以只在 student 或 teacher 的 top-K 上近似 KL。  
> 我的复现用 K=16，这是原项目比较核心的配置之一。K 太小，估计偏差大，可能漏掉 teacher 认为重要但 student 当前没关注的 token；K 太大，更接近 full KL，但 teacher forward 后 gather/log-softmax 的显存和算力压力上升。真正严谨应该做 K=0/8/16/32 的 ablation，看 reward 分布、训练稳定性、最终评估。

### 1.6 标准回答：为什么 `only_stu`

面试官问：

> 为什么默认用 `only_stu`，不用 teacher top-K？

回答模板：

> `only_stu` 的逻辑是：先看 student 当前会把概率放在哪些 token 上，再用 teacher 判断这些 token 是否合理。它非常适合 on-policy 修正 student 当前错误分布，计算也更省，因为 teacher 只需要对 student top-K token 给 log-prob。  
> 局限是如果 teacher 很强，关键 token 可能完全不在 student top-K 中，`only_stu` 就看不到 teacher 想推荐的新 token。这种情况下可以考虑 `only_tch` 或 `union`，但代价是计算更重，也可能让 student 被拉向它当前完全不理解的 token 区域，训练更不稳定。

### 1.7 标准回答：OPD 的局限性

面试官问：

> OPD 有什么局限？什么情况下不适合？

回答模板：

> 我觉得主要有五个局限。  
> 第一，teacher 成本高。每个 student rollout 都要 teacher forward，token-level reward 比 outcome reward 贵。  
> 第二，teacher/student thinking pattern 要兼容。如果一个是 thinking 模型，一个是 non-thinking 模型，chat template 或推理格式不一致，token-level KL 会错对齐。  
> 第三，teacher 不一定正确。OPD 会蒸馏 teacher 的 bias 和错误。  
> 第四，token-level KL 不等价于 task correctness，可能学到 teacher 风格而不是最终答案正确。  
> 第五，训练和评估都对 max response length、format、N rollouts 等很敏感，我复现里就看到 2048 max tokens 会导致大量 format error。

改进方法：

- 加 outcome reward，做 OPD + GRPO hybrid。
- 做 teacher-aligned prompt selection。
- 使用多个 teacher 或 verifier 过滤 teacher 错误。
- 做 top-K strategy 和 teacher temperature ablation。
- 修复评测完整性校验，避免不完整结果误导判断。

### 1.8 这一类能力的主动讲法

如果面试官问得比较泛，你可以主动说：

> 我理解 OPD 的核心不是“又一个蒸馏名字”，而是把蒸馏放到了 student 的 on-policy 状态分布上，同时把 reward 从 response-level 稀疏信号变成 token-level dense distribution signal。它解决了 GRPO 早期 reward 稀疏和普通 KD distribution mismatch 的问题，但代价是 teacher 计算成本高、依赖格式兼容，而且 token KL 不一定等于最终业务目标，所以我会倾向于把它和 outcome reward、verifier、安全规则结合。

---

## 2. 能力二：实验和方案验证能力

### 2.1 面试官真正想看什么

面试官会从“你做了什么”追问到：

- 你怎么证明训练真的有效？
- 你怎么证明不是评估脚本 bug？
- 你怎么比较 baseline？
- 你知道哪些 confounder？
- 你是否能设计 ablation？
- 你是否能从日志和指标判断训练是否健康？

### 2.2 你的验证证据链

这个项目里已有证据可以组织成四层：

1. **训练确实跑完**：checkpoint tracker 到 step 1119。
2. **训练信号存在**：OPD step 1/2/10 `actor/pg_loss` 非零，`critic/score/mean` 是负 KL，`topk/overlap_ratio` 约 0.72。
3. **分布对齐趋势**：step 1119 `critic/score/mean` 从约 -0.26 到 -0.204，`topk/overlap_ratio` 到约 0.762。
4. **离线评估结果**：AIME24/AIME25/AMC23 的 N=16 评测，且区分 2048 和 16384 max tokens。

### 2.3 标准回答：你怎么证明 OPD 有效

面试官问：

> 你怎么证明 OPD 训练是有效的，不只是跑完了？

回答模板：

> 我会分训练过程和最终评估两层证明。训练过程中，我看 OPD 是否真的有 token-level 蒸馏信号：日志里 step 1/2/10 的 `actor/pg_loss` 分别大约 0.26/0.27/0.27，不像 GRPO 早期很多 step 是 0；`critic/score/mean` 是负 KL，从约 -0.26 到 step 1119 的 -0.204，说明 student-teacher KL 在变小；`topk/overlap_ratio` 从约 0.72 到 0.76，说明 student 和 teacher 的高概率 token 集更接近。  
> 最终评估上，我用 AIME24/AIME25/AMC23，N=16 rollouts，规则评分。这里我还发现一个关键 confounder：MAX_TOKENS=2048 时 format error 很高，AIME 指标非常低；把离线生成上限提高到 16384 后，AIME24 mean_score 到 28.1%，AIME25 到 24.0%，AMC23 到 72.2%。  
> 但我不会说这已经严格证明 OPD 全面优于 GRPO，因为 GRPO 没有完整同口径训练到 1119 步。更严谨的验证需要同模型、同训练步数、同 max tokens、同 N rollout 的 baseline 对比。

### 2.4 标准回答：你会怎么设计更严谨的 ablation

面试官问：

> 如果让你继续做实验，你会怎么证明方案有效？

回答模板：

我会按优先级做这些实验：

1. **Baseline 对比**
   - Base model 直接评估。
   - GRPO 同步数训练。
   - OPD 同步数训练。
   - OPD+GRPO hybrid。
   - 全部用同一套 eval max tokens、N、temperature、top_p 和评分器。

2. **Top-K ablation**
   - K=0、8、16、32。
   - 观察 reward 方差、`topk/overlap_ratio`、训练吞吐、最终 benchmark。

3. **Top-K strategy ablation**
   - `only_stu`、`only_tch`、`union`、`intersection`。
   - 验证 teacher 强弱差异下哪种更稳。

4. **Teacher temperature ablation**
   - T=0.7、1.0、1.3。
   - 看 teacher 分布锐化或软化对训练稳定性的影响。

5. **训练/评估长度口径实验**
   - train `MAX_RESP_LENGTH=2048/4096`。
   - eval `MAX_TOKENS=2048/8192/16384`。
   - 区分模型能力和生成截断。

### 2.5 标准回答：如何判断评估可靠

面试官问：

> 你的评估脚本可靠吗？怎么避免指标是假的？

回答模板：

> 我会检查三件事。第一是输出完整性，AIME24/AIME25/AMC23 应分别有 30x16、30x16、40x16 条输出，也就是 480、480、640 行。第二是评分逻辑，`grade.py` 先抽取最后一个 `\boxed{}`，再做归一化和 sympy 等价检查；我会抽样人工检查正确和错误案例。第三是 confounder，比如 max tokens 太短导致 format error，或者 `best_score` 被误当成单次 accuracy。  
> 我在审查报告里也指出当前 `gen_vllm.py` 有静默失败风险：worker 异常只打印不退出，保存前没有断言 `len(results) == samples * N`。所以如果要把评估作为正式实验，我会先修复完整性校验。

### 2.6 标准回答：mean_score、best_score 怎么解读

面试官问：

> 你说 AIME24 best_score 63.3%，这是不是 accuracy？

回答：

> 不是。`mean_score` 是所有 rollout 的平均正确率，更接近单次采样准确率。`best_score` 是每道题 N=16 个 rollout 中至少一个答对的比例，更接近 pass@16，反映采样分布中有没有正确解。面试和简历里我更强调 mean_score，同时解释 best_score 只是多采样潜力，不能直接当单次 accuracy。

### 2.7 标准回答：实验结果和预期不一致怎么办

面试官问：

> 如果你预期 OPD 有效，但评估很差，你怎么排查？

回答模板：

我会从四层排查：

1. **评估层**
   - 是否 max tokens 太短。
   - 是否 format error 高。
   - 是否 `\boxed{}` 抽取失败。
   - 是否 JSONL 行数不完整。

2. **训练信号层**
   - `actor/pg_loss` 是否长期为 0。
   - `critic/score/mean` 是否没有向 0 靠近。
   - `topk/overlap_ratio` 是否异常低。
   - reward 是否 NaN/inf 或方差过大。

3. **数据层**
   - chat template 是否正确。
   - `enable_thinking` 是否和模型匹配。
   - prompt 是否被截断。
   - 训练/测试是否有数据泄漏或不匹配。

4. **系统层**
   - teacher 是否加载正确。
   - bf16/sdpa/fallback 是否改变了路径。
   - vLLM 和 FSDP 权重同步是否正常。

项目中的实际案例：

> 我第一次看到 2048 评估下 AIME 很差，没有直接判断模型不行，而是检查输出发现很多没有最终 boxed 答案，format error 接近 98%，说明主要是截断问题。提高离线评估 max tokens 后指标明显恢复。

### 2.8 这一类能力的主动讲法

> 我不会只用最终分数证明有效。我会同时看训练中间指标、分布对齐指标、评估完整性、format error 和 baseline。这个项目让我意识到 LLM 后训练实验里最危险的是 confounder，例如 max tokens、chat template、rollout 数、best_score 口径；这些如果不控制，分数很容易被误读。

---

## 3. 能力三：问题定位与排查能力

### 3.1 面试官真正想看什么

面试官会给你一些故障场景：

- 模型上线后能力突然下降。
- 系统上线后延迟突然升高。
- 训练 loss 变成 0 或 NaN。
- 评估结果和预期不一致。
- 同样代码昨天能跑，今天不能跑。

他想看你是否能：

- 分层定位。
- 先验证假设再动手改。
- 有日志、指标和最小复现意识。
- 能区分算法问题、数据问题、评估问题、系统问题。

### 3.2 你的通用排查框架

遇到任何问题都按五层排：

```text
指标是否真实
  -> 数据是否一致
  -> 模型/算法信号是否异常
  -> 系统资源是否异常
  -> 代码版本/环境是否变化
```

每层对应检查：

| 层 | 检查项 |
|---|---|
| 指标层 | 评估脚本、样本数、format error、mean vs best、人工抽样 |
| 数据层 | prompt 分布、chat template、token length、去重、train/test mismatch |
| 算法层 | reward、advantage、loss、entropy、KL、top-K overlap |
| 系统层 | GPU 显存、CPU 内存、KV cache、Ray、I/O、worker 异常 |
| 环境层 | Python、CUDA、vLLM、transformers、flash-attn、git diff |

### 3.3 故障场景：模型能力突然下降

面试官问：

> 如果模型上线后能力突然下降，你怎么排查？

回答模板：

> 我会先确认下降是真实能力下降还是评估/流量变化。第一步看线上请求分布是否变了，比如 prompt 长度、任务类型、语言、是否更多长推理题。第二步看输出是否被截断、format 是否变化、拒答率是否上升。第三步抽样人工看 case，区分 reasoning 错、格式错、知识错还是安全策略误伤。  
> 如果是模型版本变更后下降，我会对比上线前后的 checkpoint、tokenizer、chat template、generation config。OPD 项目里我特别会检查 `enable_thinking`、max tokens、temperature、top_p，因为这些会显著改变数学输出。  
> 如果是训练后模型下降，我会回看训练日志：KL reward 是否异常、top-K overlap 是否下降、entropy 是否塌缩、response_length/clip_ratio 是否变高。最后用上一个稳定版本灰度回滚，保证线上可用性。

结合项目例子：

> 我复现里遇到过类似“评估能力看起来很差”的情况，AIME 在 2048 max tokens 下非常低。排查后发现不是模型完全不会，而是长推理被截断导致大量 format error。这个经验让我不会第一时间改训练，而是先检查评估和输出完整性。

### 3.4 故障场景：系统上线后突然变慢

面试官问：

> 如果系统上线后突然十分缓慢，你怎么排查？

回答模板：

> 我会先把耗时拆到 generation、teacher forward、log_prob、actor update、I/O 和调度。OPD/verl 里每步有 vLLM rollout、FSDP actor forward/backward、teacher reward model forward、Ray worker 调度和 checkpoint 保存。  
> 如果是推理慢，先看 vLLM KV cache 是否不足、batching 是否下降、max_model_len 是否过大、GPU 利用率是否低。  
> 如果是训练慢，看 FSDP all-gather/reduce-scatter、CPU offload 是否造成 PCIe 瓶颈、Ray 临时目录是否 I/O 压力、checkpoint 是否过频。  
> 我项目里为了 3090 跑通打开了 param/optimizer offload，这会省显存但增加 CPU/GPU 数据搬运，所以如果系统变慢，不能只看算法，要看 offload、KV cache、Ray tmp、CUDA_LAUNCH_BLOCKING 这类工程配置。

可引用项目：

- vLLM `gpu_memory_utilization=0.4/0.5` 是为了显存稳定，但可能牺牲吞吐。
- `save_freq` 从 20 调到 200 是为了减少 checkpoint 磁盘压力。
- Ray `/tmp` 空间不足会影响稳定性。
- `CUDA_LAUNCH_BLOCKING=1` 适合 debug，不适合生产性能。

### 3.5 故障场景：训练 loss 长期为 0

面试官问：

> 训练中 `actor/pg_loss` 一直是 0，你怎么判断问题？

回答模板：

> 先区分算法。GRPO 早期 loss 为 0 可能是正常的，因为 outcome reward 稀疏，如果一组 responses 全错，advantage 可能没有有效信号。我的 GRPO 日志里很多 step 就是 `pg_loss=0`。  
> 但 OPD 不应该长期为 0，因为 token-level KL reward 是 dense 的。若 OPD 的 `actor/pg_loss` 长期为 0，我会检查：teacher reward model 是否启用，`LOG_PROB_TOP_K` 是否大于 0，student_top_k_ids 是否传到 teacher，reward 是否被 mask 全部清掉，response_mask 是否全 0，以及 top-K strategy 是否产生空 valid token。  
> 还会检查 mixed precision、NaN/inf、teacher logits 是否正常。

### 3.6 故障场景：评估结果突然变好

面试官问：

> 如果结果突然大幅提升，你会相信吗？

回答：

> 不会立刻相信。我会先查是否数据泄漏、评估 max tokens 是否变了、N rollouts 是否变了、best_score 是否被当成 mean_score、是否启用了 model verifier、是否保存了不完整 JSONL。  
> 在我的项目里，2048 到 16384 max tokens 导致结果大幅变化，这是合理的，因为 format error 大幅下降。但如果没有记录评估配置，别人可能误以为模型训练本身大幅提升。所以我会把每次评估的 model、task、N、temperature、top_p、max_tokens、评分器版本都写入结果元数据。

### 3.7 故障场景：同样代码换机器跑不起来

面试官问：

> 你的复现换机器跑不起来，你怎么定位？

回答模板：

> 先固定环境版本：Python、CUDA、torch、transformers、vLLM、flash-attn、verl commit。我的项目里裸 `python3` 是 3.5.4，而 `.venv/bin/python` 是 3.12.3，如果脚本直接写 `python3`，换 shell 就会失败。  
> 然后查 GPU 显存和 CUDA capability。比如 flash-attn wheel 和 CUDA 13.1 不兼容，原始 A800 80GB 配置放到 3090 24GB 肯定 OOM。  
> 最后查路径硬编码，例如 `RAY_TMPDIR=/mnt/sdb2/...`、模型路径、数据路径、checkpoint 路径。生产或复现脚本应参数化，而不是靠我当前机器状态。

### 3.8 这一类能力的主动讲法

> 我在这个项目里最大的收获之一是，LLM 后训练的问题不能只从 loss 看。一次指标异常可能来自训练信号，也可能来自 max tokens、chat template、评测脚本完整性、Ray 临时目录、Python 环境或 flash-attn 兼容性。我的排查习惯是先分层定位，再用最小证据验证，而不是直接调超参。

---

## 4. 能力四：工程落地能力

### 4.1 面试官真正想看什么

面试官想看你是否知道：

- 理论上可行的训练方法，生产中为什么可能不可行。
- 如何部署和服务模型。
- 如何保证系统稳定。
- 如何做灰度、回滚、监控。
- 如何控制成本和资源。
- 如何把实验脚本变成可维护 pipeline。

### 4.2 从实验到生产的差距

OPD 在论文和实验中可行，但生产落地有明显挑战：

| 挑战 | 为什么难 |
|---|---|
| teacher 成本 | token-level teacher forward 很贵 |
| 训练资源 | vLLM rollout + FSDP training + teacher model 同时抢资源 |
| 数据闭环 | 需要真实业务 prompt、脱敏、采样、过滤 |
| 评估可靠 | 不能只看 benchmark，要看业务指标 |
| 部署稳定 | 需要版本管理、灰度、回滚、监控 |
| 安全合规 | teacher KL 不等于安全合规 |

### 4.3 标准回答：如果要把 OPD 用到生产

面试官问：

> 你这个 OPD 方案如果要落地到生产，你会怎么做？

回答模板：

> 我不会直接把论文训练脚本搬到生产。生产里我会拆成离线训练 pipeline 和在线服务两部分。  
> 离线部分：收集真实业务 prompt，做脱敏、去重和分桶；用当前 student 对业务 prompt rollout；用强 teacher 或 ensemble 给 token-level 或 step-level feedback；同时加入业务 outcome reward、安全规则和格式规则；训练候选 student。  
> 评估部分：离线 benchmark 只是第一层，还要做业务集评测、人工抽样、安全红队、延迟和成本评估。  
> 上线部分：用灰度发布，小流量 A/B，对比旧模型和新模型；监控成功率、拒答率、投诉率、延迟、成本、format compliance；出现异常时按模型版本和数据版本回滚。

### 4.4 标准回答：模型怎么部署

面试官问：

> 训练好的模型怎么部署？

回答模板：

> 训练完成后需要先把 FSDP checkpoint merge 成 HuggingFace 格式，确认 tokenizer、generation_config 和 chat template 一致。部署时可以用 vLLM/TGI/SGLang 之类的推理框架，根据业务吞吐和 latency 选择 tensor parallel、batching、KV cache 预算。  
> 如果是线上服务，我会用固定模型版本和配置版本，比如 model hash、tokenizer hash、prompt template version、decoding config version。不能只部署权重，不记录 generation config，因为 temperature、top_p、max_tokens、stop tokens 都会影响行为。

结合项目：

> 我的评估就是用 merged HF checkpoint `DeepSeek-R1-Distill-Qwen-1.5B-OPD-final`，vLLM 生成 AIME/AMC 输出。这个链路和生产推理相似，但生产还需要服务化、监控、灰度和回滚。

### 4.5 标准回答：如何保证稳定

面试官问：

> 上线后怎么保证系统稳定？

回答：

我会做四类保障：

1. **版本稳定**
   - 模型权重版本。
   - tokenizer 版本。
   - prompt template 版本。
   - decoding config 版本。
   - reward/eval 版本。

2. **服务稳定**
   - GPU 利用率、显存、KV cache hit/eviction。
   - QPS、P50/P95/P99 latency。
   - error rate、timeout、OOM。
   - fallback model 或降级策略。

3. **效果稳定**
   - 在线业务指标。
   - 人工抽检。
   - 安全/合规触发率。
   - prompt drift 监控。

4. **回滚稳定**
   - 保留上一稳定模型。
   - 配置可回滚。
   - 数据和训练版本可追溯。
   - 灰度失败自动切回。

### 4.6 标准回答：资源有限先优化哪里

面试官问：

> 如果资源有限，你会优先优化哪部分？

回答模板：

> 我会先判断瓶颈在训练还是推理。如果是训练 OPD，最大成本通常是 rollout KV cache、teacher forward 和 FSDP training state。资源有限时我会优先：  
> 1. 降低 response length 或做分阶段 curriculum。  
> 2. 减少 N_RESPONSES。  
> 3. 降低 top-K 或只在关键 token/step 上蒸馏。  
> 4. 使用 bf16、offload、gradient checkpointing。  
> 5. 选更小但质量足够的 teacher 或缓存 teacher logits。  
> 如果是线上推理，我会优先优化 max_tokens、batching、KV cache、量化、小模型蒸馏和路由策略。

结合你的项目：

> 我实际就是按这个思路把原 A800 配置缩到 3090：bf16、`MAX_RESP_LENGTH=2048`、`N_RESPONSES=2`、`MINI_BATCH_SIZE=16`、vLLM `gpu_memory_utilization=0.4`、FSDP offload。

### 4.7 标准回答：理论方案为什么可能落不了地

面试官问：

> OPD 理论上不错，为什么实际生产可能不可行？

回答：

> 最大问题是成本和闭环。OPD 训练要 student rollout，再 teacher token-level forward。业务数据大时，teacher 计算成本很高。其次，业务目标不一定能用 teacher likelihood 表示，比如客服满意度、安全合规、工具调用成功都需要额外 evaluator。第三，线上系统要稳定，不能因为长 max_tokens 或 teacher 调用导致延迟不可控。  
> 所以生产里不一定全量 OPD，可以先在高价值场景或错误高发 query 上做 targeted OPD，也可以离线周期性训练，而不是在线实时蒸馏。

### 4.8 标准回答：如何做回滚

面试官问：

> 上线后发现模型能力下降，怎么回滚？

回答：

> 我会把模型版本、tokenizer、prompt template、generation config 和安全策略作为一个 release bundle。灰度上线时保留上一稳定 bundle。如果线上关键指标下降，比如解决率、投诉率、P95 延迟或安全违规率超过阈值，就自动切回上一 bundle。  
> 数据上也要保留训练数据快照和 filtering 版本，否则之后无法追溯是哪批数据或哪次 reward 配置导致退化。

### 4.9 这一类能力的主动讲法

> 我做这个项目时，一个很强的体感是：后训练不是只有算法目标，系统资源决定了方案能不能落地。比如 OPD 的 dense reward 很漂亮，但在 3090 上必须处理 vLLM KV cache、FSDP offload、Ray tmp、checkpoint 频率和 Python 环境。生产里还要加监控、灰度和回滚，否则实验有效也不等于业务可用。

---

## 5. 能力五：业务与实际场景理解

### 5.1 面试官真正想看什么

面试官会问：

- 这个项目有什么业务价值？
- 用户真正关心什么？
- 什么场景适合 OPD？
- 什么场景不适合？
- 上线成本多高？
- 资源有限先优化哪部分？
- 这个方案如何为公司带来利润或效率提升？

### 5.2 OPD 适合的场景

适合：

1. **有强 teacher、需要部署小 student**
   - 大模型 API 很贵，线上要小模型低成本服务。
   - OPD 可以把 teacher 在真实 student 状态上的行为迁移给 student。

2. **长推理任务**
   - 数学、代码、复杂工具调用。
   - outcome reward 稀疏，token-level teacher signal 有价值。

3. **有明确业务 prompt 分布**
   - 客服、教育、代码助手、企业知识库问答。
   - 可以让 student 在真实业务 prompt 上 rollout。

4. **需要降低人工标注**
   - teacher 或 verifier 能提供较便宜的自动反馈。

不适合：

1. teacher 不可靠或成本极高。
2. 业务目标不是 teacher likelihood 能表达的。
3. 安全合规要求非常强但没有额外 safety evaluator。
4. 线上延迟极其敏感，不能承受长推理。
5. 数据分布变化太快，离线训练跟不上。

### 5.3 标准回答：这个项目有什么业务价值

面试官问：

> 你这个 OPD 项目有什么实际业务价值？

回答模板：

> 业务价值可以从“降本”和“提质”两方面看。很多公司有强 teacher，比如大模型 API 或内部大模型，但线上全量调用成本高、延迟高。OPD 可以用强 teacher 指导小 student，让小模型在真实业务 prompt 分布上学习 teacher 的行为，从而降低部署成本。  
> 另一方面，长推理任务中只靠最终对错奖励很稀疏，OPD 的 token-level dense signal 可以提高训练效率，尤其适合数学、代码、工具调用这类中间过程重要的场景。  
> 但我也会强调，OPD 不直接等于业务价值。它需要和业务 evaluator 结合，比如客服满意度、工具调用成功率、安全规则、格式要求，否则可能只学到 teacher 风格，不一定提升业务目标。

### 5.4 标准回答：用户更关心什么

面试官问：

> 用户更关心模型的什么？OPD 优化的东西和用户目标一致吗？

回答：

> 用户通常不关心模型是否和 teacher 分布接近，而关心任务是否完成、回答是否可靠、速度是否快、成本是否低、安全是否合规。OPD 优化的是 distribution alignment，它和用户目标相关但不完全一致。  
> 所以生产里我不会只看 KL reward 或 teacher likelihood，而会用业务指标闭环，比如问题解决率、正确率、人工转接率、工具调用成功率、用户满意度、延迟和成本。OPD 是训练手段，不是最终 KPI。

### 5.5 标准回答：成本怎么算

面试官问：

> 上线这个方案成本多高？

回答模板：

> 成本分训练成本和服务成本。训练成本包括 student rollout、teacher forward、FSDP 训练、评估和 checkpoint。OPD 比 GRPO 多 teacher token-level forward，比 SFT 多 on-policy rollout，所以训练成本更高。  
> 服务成本取决于最终部署的 student。如果 OPD 能把强 teacher 的能力迁移到小 student，线上服务成本可能下降；但如果训练成本太高、业务量太小，ROI 不一定划算。  
> 所以我会优先在高价值、高频、teacher 调用成本高的业务场景做 targeted distillation，而不是全场景盲目 OPD。

### 5.6 标准回答：如果资源有限，先优化哪里

面试官问：

> 公司资源有限，你会先优化哪些部分？

回答：

我会按 ROI 排：

1. **先优化评估和数据**
   - 找高频失败 case。
   - 修复 format/template/max_tokens 这种低成本问题。
   - 避免训练方向错。

2. **再优化 decoding 和服务配置**
   - max_tokens、temperature、stop tokens、batching。
   - 很多线上问题不需要重训。

3. **再做 targeted fine-tuning**
   - 只针对高价值业务分布做 SFT/OPD。
   - 不全量大规模训练。

4. **最后扩大模型或训练规模**
   - 更大 teacher、更长 response、更多 rollouts。
   - 这是最贵的。

项目例子：

> 我评估中把 max tokens 从 2048 调到 16384 后，指标变化很大。这说明资源有限时，先排查评估和生成配置可能比盲目加训练更有价值。

### 5.7 标准回答：业务中如何证明产生利润

面试官问：

> 怎么证明这个方案真的给公司带来价值？

回答：

> 我会设定业务指标，而不是只看 benchmark。比如客服场景可以看人工转接率下降、首次解决率提升、平均处理时长下降、用户满意度提升、每千次请求成本下降。代码助手可以看编译通过率、单测通过率、开发者接受率。教育数学场景可以看解题正确率、讲解可读性和延迟。  
> 然后用 A/B test 或灰度实验比较旧模型和 OPD student。如果收益覆盖训练和推理成本，才算有业务价值。

### 5.8 场景化回答：客服

面试官问：

> 如果是客服场景，OPD 怎么用？

回答：

> 我会先收集脱敏后的真实客服 query，按问题类型分桶。让当前小模型生成回答，再让强 teacher 或人工规则对回答进行指导。OPD 可以在小模型真实会犯错的状态上做蒸馏，同时我会加入业务规则 reward，比如是否回答完整、是否引用正确知识库、是否触碰合规风险。  
> 评估不只看 benchmark，而看人工转接率、用户满意度、合规违规率和延迟成本。上线时灰度发布，并保留旧模型回滚。

### 5.9 场景化回答：代码助手

面试官问：

> 如果是代码模型，OPD 有什么意义？

回答：

> 代码任务里 teacher 可以是更强的代码模型，student rollout 生成代码，teacher 提供 token-level 分布指导。但代码 correctness 不能只靠 teacher likelihood，所以还要结合编译结果、单元测试、静态检查作为 outcome reward。OPD 提供 dense signal，单测提供最终正确性约束。  
> 生产指标可以是 test pass rate、开发者接受率、生成延迟和补全采纳率。

### 5.10 场景化回答：教育数学

面试官问：

> 数学教育场景怎么用？

回答：

> 数学场景和我复现的 AIME/AMC 比较接近。OPD 可以把强 teacher 的推理分布迁移给小 student，降低服务成本。但教育产品不只要答案正确，还要步骤可读、符合教学风格、不能跳步太多。所以需要额外的 step-level verifier 或 rubric reward，而不是只看最终 boxed answer。

### 5.11 这一类能力的主动讲法

> 我会把 OPD 看成一种后训练工具，而不是业务目标本身。它适合有强 teacher、小 student、长推理和明确业务 prompt 分布的场景。真正上线时要用业务 KPI 证明，比如成本、延迟、解决率、安全，而不是只说 KL 变小或 benchmark 变高。

---

## 6. 五类能力综合模拟问答

### 6.1 问题：你这个方法为什么这样设计？

回答：

> 设计来自两个痛点：普通蒸馏有 distribution mismatch，GRPO 早期 reward 稀疏。OPD 让 student on-policy rollout，再用 teacher 在 student-visited states 上提供 token-level KL reward。top-K 是为了降低 full-vocab KL 成本，PPO clipping 是为了稳定 policy update。局限是 teacher 成本高、格式要兼容、KL 不等于最终业务目标，所以可改进方向是 OPD+outcome reward、teacher selection、top-K strategy ablation。

### 6.2 问题：怎么证明有效？

回答：

> 我会看训练和评估两层。训练上，OPD 的 `actor/pg_loss` early steps 非零，`critic/score/mean` 作为负 KL 从约 -0.26 到 -0.204，`topk/overlap_ratio` 从约 0.72 到 0.76。评估上，AIME/AMC N=16 规则评分显示长生成下指标提升。但我也会控制 confounder，比如 max tokens、format error、JSONL 完整性、mean_score vs best_score。更严谨还要做同口径 baseline 和 ablation。

### 6.3 问题：模型突然变差怎么办？

回答：

> 我会先判断是真能力下降还是评估/线上分布变化。检查 prompt 分布、max_tokens、chat template、format error、generation config；再看训练信号如 reward、KL、entropy、top-K overlap；再看系统环境和版本，比如 tokenizer、teacher checkpoint、Python/CUDA/vLLM。项目里我就遇到过 2048 评估让 AIME 看起来很差，排查发现是截断和 format error，而不是先去改模型。

### 6.4 问题：生产怎么落地？

回答：

> 我会离线收集真实业务 prompt，让 student rollout，用 teacher 和业务 evaluator 提供反馈，训练候选 student；再用业务指标、安全指标、人工抽检和延迟成本做离线评估。上线采用灰度和 A/B test，监控成功率、拒答率、合规、P95 延迟和成本，保留旧模型和配置 bundle 回滚。

### 6.5 问题：业务价值是什么？

回答：

> 业务价值不是 OPD 本身，而是用强 teacher 提升小 student，降低线上成本或提升任务成功率。它适合高频、长推理、teacher 调用贵、又有明确业务 prompt 分布的场景。用户关心的是正确、快、安全、便宜；所以 OPD 必须和业务 KPI 对齐，不能只看 teacher KL。

---

## 7. 可主动讲的项目经历

### 7.1 “我不是只跑脚本”

> 我追了 `ray_trainer.fit()` 的训练链路，知道 rollout、log_prob、teacher reward、distillation reward、advantage 和 actor update 是怎么串起来的；也追了 `token_reward_direct`、top-K teacher log-prob 和 PPO loss 的实现。

### 7.2 “我做过资源受限适配”

> 原配置是 8xA800 80GB，我这边是 8x3090 24GB，所以我系统地调了 bf16、FSDP offload、activation offload、gradient checkpointing、vLLM `gpu_memory_utilization`、response length、N responses 和 mini batch。

### 7.3 “我能从日志判断训练信号”

> 我不是只看最终 benchmark，而是看 OPD early steps 的 `actor/pg_loss`、`critic/score/mean`、`topk/overlap_ratio`，并和 GRPO 早期稀疏 reward 做对比。

### 7.4 “我发现了评估 confounder”

> AIME 2048 评估很差时，我没有直接说模型训练失败，而是检查 format error 和输出截断，发现 max tokens 是关键因素。

### 7.5 “我做过代码审查”

> 我发现 flash-attn fallback 只是当前 FSDP 路径 workaround，不完整兼容 flash-attn API；也发现评测生成脚本有 worker 失败后静默保存不完整结果的风险。

---

## 8. 面试中要避免的回答

### 8.1 避免只讲概念

弱回答：

> OPD 就是 on-policy distillation，用 KL 蒸馏 teacher。

强回答：

> OPD 解决 off-policy KD 的 distribution mismatch 和 GRPO outcome reward 稀疏问题；它在 student rollout 的状态上用 teacher token distribution 做 dense reward，但依赖 teacher 质量和格式兼容，成本也更高。

### 8.2 避免只讲结果

弱回答：

> 我最后 AIME24 到 28.1%。

强回答：

> AIME24 28.1% 是在 N=16、MAX_TOKENS=16384、规则评分下的 mean_score；2048 时因为截断 format error 很高，只有 1.9%。所以这个结果要和评估长度一起解释。

### 8.3 避免把问题说成已经彻底解决

弱回答：

> 我解决了 flash-attn 问题。

强回答：

> 我做了一个让当前 FSDP+sdpa 路径跑通的 workaround，但审查发现 fallback API 不完整，Megatron 或 sequence parallel 路径会失败，所以正式修复应对齐 transformers/flash-attn 的返回值。

### 8.4 避免把实验说成生产价值

弱回答：

> 实验有效，所以上线一定有收益。

强回答：

> 实验有效只说明在 benchmark 上有潜力。生产要看业务 KPI、延迟、成本、安全和用户满意度，还需要灰度和 A/B test。

---

## 9. 五类能力速查表

| 能力 | 面试官问题 | 你的核心抓手 |
|---|---|---|
| 底层原理 | 为什么这么设计？ | OPD 解决 distribution mismatch + sparse reward；top-K 降成本；负 KL 做 dense reward |
| 实验验证 | 怎么证明有效？ | 训练信号、top-K overlap、评估结果、baseline、ablation、confounder 控制 |
| 问题定位 | 结果不符合预期怎么办？ | 指标层、数据层、算法层、系统层、环境层分层排查 |
| 工程落地 | 怎么上线？ | 训练 pipeline、部署服务、监控、灰度、回滚、成本控制 |
| 业务理解 | 有什么价值？ | 强 teacher -> 小 student 降本提质；业务 KPI 而非 KL 才是最终目标 |

---

## 10. 最后背诵版

如果面试前只剩 5 分钟，背这一段：

> 我这个项目复现的是 Rethinking OPD。它解决的核心问题是：传统蒸馏在固定 teacher 数据上训练，和 student 部署时自己的分布不一致；而 GRPO 这类 outcome-level RL 在数学推理早期 reward 很稀疏。OPD 让 student 自己 rollout，然后在 student 访问到的 token 状态上，用 teacher 分布计算 top-K 近似 KL，把负 KL 当作 dense token reward，再用 PPO-style update 训练 student。  
> 我在 8x3090 24GB 上把原 A800 配置缩小，使用 bf16、FSDP offload、gradient checkpointing、activation offload，降低 response length、rollout 数和 vLLM KV cache 显存，完成 1119 step 1.5B OPD 训练。日志里 OPD early steps `actor/pg_loss` 非零，`topk/overlap_ratio` 约 0.72 到 0.76，说明蒸馏信号确实存在。  
> 评估上我发现 max tokens 是关键 confounder，2048 会严重截断数学推理导致 format error，16384 离线评测下 AIME24 mean 28.1%、AIME25 24.0%、AMC23 72.2%。但我会诚实说明这不是论文完全同口径复现，因为训练 response length 是 2048。  
> 从工程和业务角度，OPD 的价值是用强 teacher 指导小 student，在真实业务 prompt 分布上降低部署成本或提升长推理能力；但生产里还要考虑 teacher 成本、监控、灰度、回滚、安全和业务 KPI，不能只看 KL 或 benchmark。

