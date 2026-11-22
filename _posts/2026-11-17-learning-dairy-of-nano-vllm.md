---
layout: post
title: Learning Diary of Nano vLLM
date: 2026-11-17
description: Nano vLLM，一个 AI infra 入门级推理框架
tags: [AI infra]
---

在学习 AI infra 的过程中接触到了 Nano vLLM 这个相对简单的入门框架，感觉非常值得深入研究，以扫除 AI infra 学习过程中遇到的诸多困惑，并借此对整个推理框架建立一个系统的认知。

> 本文先沿一次请求的执行路径，梳理 **调度 → 前向推理 → 后处理**，再深入 **Embedding、Decoder 与 LMHead** 的模型实现及并行优化。

## 1. Nano vLLM 主体框架

一个 example 如下，上层接口十分简单。我们深入 `llm.generate()`，看看底层具体做了什么。

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # LLMEngine
...
outputs = llm.generate(prompts, sampling_params)  # 提示词、输出采样参数
```

`generate()` 首先做了一些预处理，然后通过 `add_request()` 方法把成对的 `prompts` 与 `sampling_params` 打包为一个 `Sequence` 对象，添加到 Scheduler 的 `waiting` 队列中等待处理。

随后进入循环，不断调用 `step()` 方法，直到这一批 `prompts` 全部处理完毕。因此，核心在于 `step()` 方法每轮循环如何安排 token 的计算。

`step()` 只有五行：

```python
seqs, is_prefill = self.scheduler.schedule()  # 1
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
token_ids = self.model_runner.call("run", seqs, is_prefill)  # 2
self.scheduler.postprocess(seqs, token_ids, is_prefill)  # 3
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

其中，`num_tokens` 统计本次 step 具体处理多少 token，用于进度显示；`outputs` 收集本轮算完的 `Sequence`。核心调用只涉及三行，已用 `# 1`～`# 3` 标出：

1. **调度**：每一个 step 要从队列中取出 `Sequence` 进行计算，`scheduler.schedule()` 决定取谁、一次可以取多少。
2. **前向推理**：`model_runner.call()` 负责实际的推理部分，将取出的 `Sequence` 送入模型。
3. **后处理**：`scheduler.postprocess()` 更新 `Sequence` 的一些必要属性，并判断一个 `Sequence` 是否已经计算完成。

> **核心流程：调度 → 前向推理 → 后处理。**

### 1.1 调度

在正式进入调度逻辑之前，我们需要先研究一下构成调度操作基础的三个类。

#### Block

推理过程中，为了减少显存浪费，对其进行更加精细灵活的管理，Nano vLLM 借鉴了操作系统中的分页思想。`Block` 类便是一个可分配的最小逻辑内存单位。

它并不实际管理内存，但记录一块内存的索引号（用于寻址）、引用数（判断是否可以释放）、哈希值（前缀特征，用于 KV 复用）和 token id 列表（与本块相关的 token）。`Block` 有两个方法：`update()` 与 `reset()`，前者更新哈希值与 token id 列表，后者重置本块所有属性。

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

#### Block Manager

`BlockManager` 统一管理所有的 `Block`，所有跟 `Block` 有关的操作都由它完成，譬如前缀哈希值更新、块分配、块回收以及 `Sequence` 的块表更新等等。

要完成块的管理，`BlockManager` 至少要维护以下信息：

- `block_size`：一个块能容纳的 token 数；
- `blocks`：所有逻辑块的信息；
- `free_block_ids`：哪些块是空闲的；
- `used_block_ids`：哪些块已经使用过了。

除此以外，为了快速命中已经缓存过的 KV，使用 `hash_to_block_id` 进行查找。

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

#### Sequence

`Sequence` 专门管理一个请求的信息，比如总 token 数、采样参数、占据的所有 `Block` 等。一个 `Sequence` 就对应一条用户的请求或者一条询问。

```python
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

大致了解核心类后，再来看 `schedule()` 函数，它只有两种返回：

1. (scheduled_seqs, True)
2. (scheduled_seqs, False)

为了使调度逻辑更简单，每次调度中如果存在 prefill，就只计算 prefill，否则计算 decode：前者取自 `waiting` 队列，后者取自 `running` 队列，后续 decode 调度会使用 CUDA Graph 进行加速。

下面三张图阐释了 `schedule()` 的逻辑，比较特殊的地方在于 `running` 队列调度时存在抢占逻辑。

<figure class="scheduler-diagram scheduler-diagram-overview">
  <img src="{{ '/assets/images/nano-vllm/nano-vllm-scheduler-complete.png' | relative_url }}" alt="Nano vLLM 调度总流程" width="1908" height="543">
  <figcaption>图 1：Nano vLLM 调度总流程</figcaption>
</figure>

<figure class="scheduler-diagram scheduler-diagram-detail scheduler-diagram-prefill">
  <img src="{{ '/assets/images/nano-vllm/nano-vllm-scheduler-prefill.png' | relative_url }}" alt="Prefill 调度流程" width="2385" height="3735" loading="lazy">
  <figcaption>图 2：Prefill 调度流程</figcaption>
</figure>

<figure class="scheduler-diagram scheduler-diagram-detail scheduler-diagram-decode">
  <img src="{{ '/assets/images/nano-vllm/nano-vllm-scheduler-decode.png' | relative_url }}" alt="Decode 调度流程" width="1719" height="1764" loading="lazy">
  <figcaption>图 3：Decode 调度流程</figcaption>
</figure>

### 1.2 前向推理

前向推理由 `ModelRunner` 负责，将调度阶段给出的 `seqs` 交给 `run()` 方法进行处理。首先需要搞明白以下两个问题：

1. 单独一张卡上的推理是怎么进行的？
2. 多张卡是如何进行协作的？

#### 单卡推理

单卡推理逻辑可以去看被调用的函数干了什么，五行尽收眼底：

```python
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
temperatures = self.prepare_sample(seqs) if self.rank == 0 else None  # sample on rank 0 only
logits = self.run_model(input_ids, positions, is_prefill)
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
reset_context()
```

> **单卡推理：变量准备 → 模型推理 → 采样 → 重置上下文。**

首先，针对 prefill / decode，事先计算所用变量的索引：

- **Decode**：逻辑比较简单，因为每次只处理最后一个 token。它的 `context` 语义包括最后一个 token 的 KV cache 写入位置、上下文长度、上下文 KV cache 的位置，返回当前所有序列的 token id 与 position。
- **Prefill**：处理稍显复杂，因为一个 seq 处理的是多个 token，且可能存在 chunked prefill。当前要算的 query 长度和需要的 KV 长度可能不同，新的 KV cache 要写入的位置也是一段内存，最后返回的也是当前所有序列的 token id 与 position（拍平）。

还要准备采样参数。`temperature` 控制模型输出的随机性，实际采样只在 rank 为 0 的 GPU 上运行，因此只保存一份。

```python
# 只展示比较简单的 decode prepare 逻辑
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = [] # ids of cur token to compute
    positions = [] # pos of cur token to compute
    slot_mapping = [] # pos of new token kv cache
    context_lens = [] # num of context token
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

然后是真正的模型推理。这里在 not prefill、token 处理数小且非测试场景下使用了 CUDA Graph 进行优化，因为该场景下性能瓶颈在于 CPU 对 CUDA Kernel 的调度而非 GPU。CUDA Graph 把多个算子打包录制为一张计算图，CPU 只需要发一次命令就可以跑完多个 CUDA Kernel。

录制好的 Graph 张量形状是固定的，每次启动前把变量填充进去即可；非此类场景仍然走正常推理路径，CPU 一次调度一个 CUDA Kernel。

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        ...
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

接着，借助 `Sampler` 类，从模型推理返回的 `logits` 数值中采样获得具体的 token ids：

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

最后清除本次推理的 context，为下一次推理做准备。

#### 多卡协作

推理过程可以使用多张卡进行，把 Tensor 切分到不同的卡上运算可以提高计算效率，也就是常说的分布式计算。

Nano vLLM 框架也支持多 GPU 推理，其采用主从式架构：每张 GPU 单独占有一个进程，rank 为 0 的 GPU 通过 `SharedMemory` 与 `event` 向其他 GPU 发送运行指令；所有 GPU 同时对负责的数据进行推理，需要汇总结果时使用 `all_reduce()` 等指令通信。

GPU 进程的初始化在 `LLMEngine` 类的 `__init__()` 方法中：

```python
self.ps = []
self.events = []
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)
```

这里使用 `spawn` 创建一个全新的进程，每个进程中都存有一个 `ModelRunner` 对象，每个进程的控制变量 `event` 都交给当前 `model_runner` 管理（rank = 0）。

再来看 `ModelRunner` 中相关的代码：

```python
import torch.distributed as dist
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()
```

rank = 0 的 GPU 会建立一块进程间的共享内存，之后所有指令的发布都通过该内存；其他 GPU 获取共享内存后进入 `loop` 状态，等待 `event` 被触发，然后从共享内存中加载指令运行。

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break

def read_shm(self):
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

只有 rank = 0 的 GPU 可以向共享内存写入。每次在 `step()` 时调用的 `call()` 方法就是一个简单的写入指令，然后触发其他 GPU 读取指令执行。

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    return method(*args)

def write_shm(self, method_name, *args):
    assert self.world_size > 1 and self.rank == 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()
```

### 1.3 后处理

后处理部分比较简单，主要是根据前向推理返回的结果，更新 `Sequence` 的一些属性：

- 新增 `Block` 的哈希值；
- 已缓存 token 数；
- 清空当前轮次调度的 token 数。

如果是 prefill 计算且缓存 token 数小于序列 token 数，说明还没到达 token 生成部分，不需要追加本轮生成的 token id，否则需要追加。

这里还会判断 `Sequence` 是否已经完成本次请求。如果完成，会标记为 `FINISHED`，并清空资源占用。

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

## 2. Nano vLLM 模型实现

上一部分主要关注整个框架是如何进行一次前向计算的，这里我们深入研究一下模型的实现以及优化。

Nano vLLM 使用 Qwen3 模型架构，针对 Embedding、Decoder 以及 LMHead 层进行并行优化，这也是目前所有 LLM 中最常见的三部分。其中，Embedding 与 Decoder 被封装为 `Qwen3Model` 作为 backbone，加上 LMHead 实现完整模型 `Qwen3ForCausalLM`。

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

```

### 2.1 Embedding

该层负责将输入的 token id 转化为对应的编码向量，在推理过程中相当于一个查表操作。但在实现过程中要考虑 Tensor Parallel：每张 GPU 只负责输入数据中一部分 token id 的嵌入，最后整合所有结果来加快推理速度。

在实现过程中主要考虑三个问题，这也是之后所有涉及 Tensor Parallel 操作的算子都要考虑的问题：

1. 权重如何加载？
2. 计算如何拆分？
3. 输出如何整合？

#### 权重加载

由于每张 GPU 仅仅负责一个区间的嵌入，它只需要加载该区间对应的权重即可。假设 Embedding 层完整权重尺寸为 `(num_embeddings, embedding_dim)`，则仅需沿第一个维度，按照本 GPU 负责的区间 `(start_idx, end_idx)` 加载权重即可。多卡协作时，会给予每个 GPU 一个独特的 rank 索引号，因此可以轻易计算出对应的区间。

```python
class VocabParallelEmbedding(nn.Module):
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # slice up
        param_data.copy_(loaded_weight)
```

#### 计算拆分

每张 GPU 都会收到完整的输入，它只能嵌入落在当前负责区间的 token id，其他超出部分都应被 mask 掉，视为无效输入，实际上还是进行了完整的嵌入计算，只是最后输出全部清 0，不影响 reduce 结果。

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
      ...
      self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
      self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
      ...

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # mask token id not used by this GPU
            x = mask * (x - self.vocab_start_idx) # offset + mask
        y = F.embedding(x, self.weight)
        ...
```

#### 输出整合

计算拆分以后，要思考每张 GPU 的输出结果是否需要整合以及如何整合：

- **是否需要整合**：看后继层需要什么，这里 Decoder 显然需要完整输入。
- **如何整合**：看你是如何进行计算拆分的。这里每张 GPU 输出的是形状相同的张量，只有负责区间的嵌入有意义，其他部分都被置为 0，因此 `all_reduce()` 操作便可以使所有 GPU 拿到完整结果。

```python
    def forward(self, x: torch.Tensor):
        ...
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # unsequeeze: (num_tokens) -> (num_tokens, 1)
            dist.all_reduce(y) # add all results in GPUs
        return y
```

### 2.2 Decoder

Decoder 占据 LLM 的大部分参数与计算量，可以说是 LLM 推理优化的重中之重。一个 Decoder 的主要结构包括 Attention Layer、MLP Layer、Norm Layer 以及残差连接。Nano vLLM 将残差连接与 Norm Layer 一并实现，只需要在返回列表中增加一个初始输入即可。

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

#### Attention

Attention 模块自顶向下依次为 **QKV 投影 → ROPE 位置编码 → 自注意力计算 → 输出投影**。

多头注意力的独立结构为并行性提供了天然的优势。在 QKV 投影中，可以把权重按输出维度拆分，从而把不同 head 放到不同 GPU 上并行计算，最后得到按列拆分的部分输出；当进行输出投影时，输入相当于在权重的输入维度上拆分，因此可以把权重也按输入维度拆分，最后通过 `all_reduce()` 获得最终结果。

这里涉及到两次权重的拆分：

| 并行方式 | 权重拆分维度 | 部分输出 | 整合方式 |
| --- | --- | --- | --- |
| **ColumnParallel** | 输出维度 | 某一维度部分连续区间的结果 | 拼接 |
| **RowParallel** | 输入维度 | 输出某一位置的部分结果 | 加和 |

```python
class Qwen3Attention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

##### QKV 投影

Nano vLLM 把 QKV 拼接在一起，一次计算搞定，参数的加载方式略有不同，需要依据 loaded id 分别进行 QKV 权重的载入，`chunk()` 方法把完整的权重拆为多份，当前 GPU 根据 rank 取对应的部分。

```python
class QKVParallelLinear(ColumnParallelLinear):
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class ColumnParallelLinear(LinearBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) # x @ weight.T
```

##### ROPE 位置编码

注意力本身无法区分序列中 token 的顺序。对于一个新 token，它的前文顺序随意交换，在没有位置信息的情况下，计算结果相同；对于文本这类明显有先后顺序的序列，显然是不合理的，因此需要进行位置编码。

ROPE 是目前最主流的位置编码方法，它在每一层注意力计算中都会注入相对位置信息。位置编码中没有任何需要学习的参数，因此可以只算一次，缓存为一个表格，后续直接查表。

这里沿隐藏维度从中间分开，前后两半按序组合注入位置信息。

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin # just a rotation
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

@lru_cache(1) # similar to "static"
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

##### 自注意力计算

自注意力使用 `flash_attn()` 中已有的接口，KV cache 也是在此缓存的：启动 token 数个 `store_kvcache_kernel()` 完成，`context` 中的 `slot_mapping` 用于锁定要写入的物理块位置。

```python
@triton.jit
def store_kvcache_kernel(
  ...
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx) # position of pysical block
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

由于是在自注意力计算前就进行了缓存，因此所有 KV 值完全可以从 cache 中读取（decode 阶段和 chunked prefill 阶段），可以直接传入 K cache 和 V cache。同时需要 `context` 中记录的一些信息，包括上下文长度、`block_table`；prefill 还需要拼接序列中每段 q 与 k 的长度。

```python
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ...
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v, ...)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, ...)
        return o
```

KV cache 的物理内存分配在 `ModelRunner` 中进行，了解 KV cache 的形状更有利于理解上面 `store_kvcache_kernel()` 的索引计算。这里 KV cache 的数量并不是用户指定的，而是在初始化时计算的。

可以看到，模型的每层都存储了 KV cache 对应的切片，每个 `Block` 里可以容纳 block size 个 token。

```python
class ModelRunner:
    def allocate_kv_cache(self):
        ...
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

##### 输出投影

输出投影是一个先升维再降维的操作。由于 QKV 投影时把权重按列切分，每个 GPU 只能拿到一部分输出，因此这一层的输入也是沿着隐藏层维度切分的一部分。

很自然地，可以将权重也按隐藏维度（输入维度）切分，最后进行 `all_reduce()` 便可拿到完整的结果，这是一个典型的 RowParallel。

```python
class RowParallelLinear(LinearBase):
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

#### MLP

MLP 整体的并行逻辑与 Attention 相似，也是先 ColumnParallel，然后 RowParallel。

不过，MLP 中存在门控机制：对形状相同的输入变量做不同的操作，得到形状相同的输出，再逐元素乘积，即存在 $W_1x$ 与 $W_2x$。考虑到 $W_1$ 与 $W_2$ 形状相同，可以拼接在一起加速计算。

这里使用了 `MergedColumnParallelLinear`，主要是参数的加载方式改变了：每张 GPU 负责 $W_1$ 与 $W_2$ 匹配的固定区间，得到按列拼接在一起的部分输出 $[x_1, x_2]$，进入门控时直接用 `chunk()` 劈开走不同路径即可。

```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

#### RMSNorm

RMS 正则化沿隐藏维度进行，可学习参数即为隐藏维度大小，只有 scale 操作，这里把残差连接也融合进去。

```python
class RMSNorm(nn.Module):
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x
```

### 2.3 LMHead

LMHead 用于将 hidden 映射为词表的 logits 分数。仍然是一张 GPU 负责一部分权重，按输出维度切分，最后通过 `gather()` 整合到 rank = 0 的 GPU 中。

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1 # last indice of every seq
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight ) # (token num, num_embeddings)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0) # ordered by rank
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

## 总结

至此，整个 Nano vLLM 的框架与模型实现基本理清，后续考虑使用该框架跑一些实验对比，学习一下推理性能分析工具。