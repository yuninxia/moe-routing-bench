You: 

example.pdf

PDF

Categories\_of\_Layouts.md

File

Topic 2: Mixture-of-Experts (MoEs) and Learning to Specialize MoE models activate only a fraction of their parameters for any given input, enabling massive model scale with constant computational cost. This topic explores the optimization and design of these architectures. Project Focus: A blend of literature review and implementation. Review the evolution of routing algorithms in MoEs. The implementation could involve building a simple MoE layer on top of a standard transformer block and experimenting with different routing strategies. The report should discuss the trend towards parameter-efficient and dynamic routing. Core Papers: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer Learning to Specialize: Joint Gating-Expert Training for Adaptive MoEs in Decentralized Settings Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models PERFT: Parameter-Efficient Routed Fine-Tuning for Mixture-of-Expert Model From Sparse to Soft Mixtures of Experts 对这个MOE的这个routine algorithms的这个东西其实是一个这个课程的project: https://akyrillidis.github.io/comp414-514/project/ The course project can be categorized as a literature review, original research, or a literature review that leads to original research. Literature Review: An in-depth review and analysis of a collection of papers around a topic. The review must provide a critical summary, exposition, and discussion of the topic, which will often require reading related work for context. This option will be sparsely selected by the tutor; it is easy to throw a bunch of papers into ChatGPT/Gemini and get your work done. The literature review should be on an interesting collection of papers that they do not seem connected but they are. Original Research: You are strongly encouraged to align the project with your current research interests. Otherwise, the instructor can provide ideas. Projects can be theoretical (e.g., proving a convergence guarantee) or experimental (e.g., implementing and benchmarking an algorithm). This will be the default option for projects. You are encouraged to team up. Optimal size for a team is 3 students: deviations from this number will be considered given the circumstances (either there is no one to team up, or You have your own research You want to push forward, or the project is difficult and allows more team-members). Milestones Project Selection: Choose a project topic ASAP. Deadline: October 3rd. Project Proposal: Submit a one-page description of your project. Outline the core problem, why it is interesting, and a list of key papers to read. For research projects, provide a clear plan of the steps You will take. Deadline: October 10th. In-Class Presentation: Towards the end of the semester, You will give a short spotlight talk (5-10 minutes) on your project. Focus on high-level ideas and key results. Final Report: A written report of at least six pages (excluding references) in a standard conference format (e.g., ICML/NeurIPS LaTeX template). Deadline: End of the semester. Note that exceptional projects can be continued beyond the semester with the goal of publication. 其实我在想是我是否可以呃因为你可以发现在cute DSL 当中这部分当中其实有非常多的它的layout的数据表达之类的我们在这里做moe的这个东西我们能不能使用cute DSL去实现所有的代码然后进行 performance 的比较 从而实现这个 project或者完成这个课程的项目然后再进行 performance 的比较然后进行 performance 的比较然后进行 performance 的比较再进行 performance 你觉得这是一个好思路吗？我唯一的担心就是，有些论文可能不太好用 cute-dsl 来实现 结论：这是个\*\*很好的课题方向\*\*，而且完全可以把“MoE 路由/调度 → 低层数据布局与核函数”这条线用 \*\*CuTe DSL（CUTLASS 4.x 的 Python DSL）\*\*串起来做成“文献综述 + 原型实现 + 对比评测”的项目。CuTe DSL 原生支持用 Python JIT 写 GPU 内核、表达与 C++ CuTe 一致的\*\*布局代数\*\*（composition / logical division / logical product / complement / coalesce 等），并能和框架通过 DLPack 互通，这正适合你把 MoE 的\*\*dispatch（打包）/ combine（回收）/ 分组 GEMM\*\*这几步做成可复用内核原语，再对不同路由算法做核级性能对比。(\[NVIDIA Docs\]\[1\]) 另外，课堂项目类型也鼓励“综述引出可运行原型”，而近年的研究也表明\*\*不同 token-mixer 并非唯一关键\*\*、通用结构同样重要，做“结构 + 实现 + 评测”的结合有学术价值。 --- # 你担心的点 & 可行性判断 \* \*\*“论文难以用 CuTe DSL 复现”\*\*：MoE 的重计算主要在 \*\*dispatch/pack → per\-expert FFN（两次 GEMM）→ combine/unpack\*\*。其中打包/回收恰好是\*\*布局变换 + gather/scatter\*\*问题，CuTe 的\*\*coalesce / composition / logical division / complement\*\*把这些操作形式化并可直接编码；GEMM 可以直接用 CUTLASS/DSL 的模板或调用现成 GEMM。风险更大的是\*\*多机/多卡的 all-to-all\*\*，但课程项目完全可以先\*\*单卡\*\*做核级对比。 \* \*\*CuTe DSL 的成熟度\*\*：官方文档明确提供 JIT、布局参数、线程值布局、与 PyTorch/JAX 的 DLPack 接口等，并作为 CUTLASS 4.0+ 的核心 Python 前端；2025 LLVM Dev Meeting 也列了相关 talk，社区在活跃演进。(\[NVIDIA Docs\]\[1\]) \* \*\*路线选择\*\*：选 2–3 种具代表性的路由/设计，核级对比就能做得很扎实，不需要把所有论文都等量实现。 --- # 一页式 Project Proposal（可直接交） \*\*题目\*\*：Routing Algorithms in MoEs from a Layout/Kernels View — A CuTe-DSL Implementation & Benchmark \*\*问题与动机（Why）\*\* MoE 通过稀疏激活实现“巨大参数量 + 常数计算”，但真实性能瓶颈往往在\*\*路由→打包→分组 GEMM→回收\*\*的内存访问与布局变换；不同路由（Top-k/Top-1/Soft/Sequential）对内核形状与数据摆放的要求不同。我们希望用 \*\*CuTe DSL\*\* 把这些步骤抽象成通用内核原语，量化不同路由在\*\*单卡\*\*条件下的核级吞吐、带宽效率和算/存重用，并复核文献报告的系统收益。参考：Sparsely-Gated MoE、Switch Transformer、Soft-MoE、PERFT、Chain-of-Experts 等。(\[arXiv\]\[2\]) \*\*核心文献（必读）\*\* Shazeer17（MoE/Top-k + 负载均衡损失）、Switch Transformer（Top-1 简化与稳定性）、Learning to Specialize（门与专家\*\*联合学习\*\*）、Soft-MoE（全可微稀疏/soft 路由）、PERFT（路由化 PEFT 微调）、Chain-of-Experts（层内串行专家通信）；系统侧：\*\*MegaBlocks\*\*（把 MoE 重写成 block-sparse 内核）。(\[U of T Computer Science\]\[3\]) \*\*实现范围（What）\*\* 1. \*\*CuTe-DSL 内核原语\*\* \* \*\*Pack/Dispatch\*\*：按 expert id \*\*分桶 + 前缀和\*\*计算写入偏移，生成\*\*紧凑专家批\*\*（避免 padding）。布局层面使用 \*\*composition / logical division / complement / coalesce\*\* 构造“(token→专家) 重排 + 切 tile + 连接”流水。 \* \*\*Grouped GEMM\*\*：调用 CUTLASS/DSL GEMM（或按需要自定义小批-多形状 gemm；先统一形状再对比）。 \* \*\*Combine/Unpack\*\*：按 gate 权重（Top-k 或 Soft）把专家输出写回原序，并支持加权合成。\*\*coalesce\*\*/relative-coalesce 可用来把多级嵌套 shape 收敛为紧凑访问。 2. \*\*路由策略最小可用集（至少 3 个）\*\* \* \*\*Top-k（Shazeer17）\*\*：k=2，含 load-balancing loss、capacity factor； \* \*\*Top-1（Switch）\*\*：更简单稳定； \* \*\*Soft-MoE\*\*：权重化合成，验证 soft 混合对核形状/写回路径的影响； （可选）\*\*Chain-of-Experts\*\*：做“2 次迭代”的微型原型，观察串行专家对访存与缓存复用的影响。(\[arXiv\]\[2\]) \*\*评测方法（How）\*\* \* \*\*微基准\*\*：只测 MoE 层（不含注意力），维度覆盖 \*tokens × hidden × experts × capacity\*；记录 \*\*kernel 时间/GB/s/inst-per\-cycle\*\*。 \* \*\*端到端小模型\*\*：把 MoE-FFN 插入到标准 Transformer block（例如 tiny LLM / TinyStories），比较相同验证集下吞吐与收敛趋势。 \* \*\*对照\*\*：PyTorch 朴素实现、Tutel/MegaBlocks 的公开实现（如有可复用内核），对照\*\*block-sparse\*\*思路。(\[arXiv\]\[4\]) \*\*时间线\*\* \* 选题已定；\*\*10/10 提案\*\*（本页即可）。 \* 10 月中：完成 Top-1/Top-k 打包/回收内核 + 基准脚本； \* 11 月：加入 Soft-MoE 路由、完善 grouped GEMM/融合算子； \* 期末：报告 ≥6 页 + 5–10 分钟课堂短讲。 \*\*可交付\*\* \* CuTe-DSL 内核原语（pack/combine）； \* 3 种路由的核级性能曲线与 roofline； \* 一份“路由 → 布局/内核形状”的方法学总结。 --- # 具体落地建议（含 CuTe-DSL 技术要点） 1. \*\*先做单卡核级\*\*。避免多机 all-to-all，专注\*\*布局 → 访存 → 分组 GEMM\*\*。 2. \*\*把“打包/回收”写成纯布局算子\*\*： \* 用 \*\*composition\*\* 把“原始 (B,T,H)”坐标映射到“(expert\_bucket, local\_idx, H)”；再用 \*\*logical division\*\* 按 tile 抽取/排列，组合出每个专家的\*\*致密批\*\*；必要时用 \*\*complement\*\* 快速求 tiler 的补布局用于索引反变换；输出前对嵌套 shape \*\*coalesce\*\* 成紧凑模式，便于后续 GEMM 读写。 3. \*\*CuTe-DSL 编码提示\*\* \* `cute.make_layout(shape=..., stride=...)` 定义自定义布局；`cute.composition(B, A)` 生成组合布局；`cute.coalesce(A)` 做（相对）合并；这些在 CuTe DSL 示例里都有 API。 \* 官方文档提供了 \*\*JIT/调试/自动调参\*\* 等实用模块，适合快速原型与内核参数扫描。(\[NVIDIA Docs\]\[5\]) 4. \*\*选择三条“路径”对比\*\* \* \*\*Padding 路径\*\*：按 capacity 对齐 → 分组 GEMM； \* \*\*紧凑打包路径\*\*：prefix-sum + scatter，完全无 padding； \* \*\*Block-sparse 路径\*\*（借鉴 MegaBlocks 思路）：固定 block 尺寸，走 block-sparse GEMM。(\[arXiv\]\[6\]) > 小示意（伪代码，展示思路） > >

```
python
> # inputs: X[B, T, H], gate_idx[B,T] ∈ [0..E), gate_w[B,T,k]
> # 1) 统计各 expert 负载并prefix-sum求offset
> load = bincount(gate_idx)                # CPU/PyTorch
> offsets = exclusive_scan(load)
>
> # 2) 用 CuTe DSL 构造“原→目标”的布局映射
> A = cute.make_layout(shape=(B,T,H), stride=(T*H, H, 1))
> # 目标布局：(E, cap, H) 其中 cap == max load 或使用紧凑长度
> B = cute.make_layout(shape=(E, cap, H), stride=(cap*H, H, 1))
> # composition/logic_divide/complement 组合出重排映射
> B_o_A = cute.composition(B, A)           # 实际需按桶重排后再coalesce
>
> # 3) 调用 grouped GEMM: Y_packed = FFN(W1,W2) on (E, cap, H)
>
> # 4) Combine: 依据 gate_w 反映射写回到 (B,T,H)，支持 Top-k/Soft
>
```

\> > 其中 \*\*composition / coalesce\*\* 的等价代码路径与示例见 CuTe 文档与 layout-categories 说明。 --- # 建议实现与评测集 \* \*\*路由/论文到实现映射\*\* \* \*\*Sparsely-Gated MoE（Top-k）\*\*：实现分桶 + 容量 + 负载均衡损失；核上对比 padding vs 紧凑打包。(\[arXiv\]\[2\]) \* \*\*Switch（Top-1）\*\*：最小代价路由，与 Top-k 对比核效率。(\[arXiv\]\[7\]) \* \*\*Soft-MoE\*\*：combine 阶段做加权回写，对比写回访存与算/存比变化。(\[arXiv\]\[8\]) \* （可选）\*\*Chain-of-Experts\*\*：做 2-step 串行专家微原型，量化缓存/共享内存复用的利弊。(\[arXiv\]\[9\]) \* （系统参照）\*\*MegaBlocks\*\*：做一个 block-sparse 版本的 pack/combine 以对照紧凑打包。(\[arXiv\]\[6\]) \* \*\*评测指标\*\*：单层 kernel latency、GB/s、Achieved Occupancy、GEMM TFLOP/s；端到端测 tokens/s。 \* \*\*数据/规模\*\*：B×T ∈ {32×128, 64×256, 128×512}，E ∈ {4,8,16}，H ∈ {1024, 2048}，capacity ∈ {1.0, 1.25}。 \* \*\*成功标准\*\*：给出“路由→布局/核形状→性能”的清晰映射与结论；在 ≥2 组规模上，CuTe-DSL 原语的紧凑打包显著优于 padding 路径；Soft-MoE 的写回成本差异被定量化。 --- # 风险与备选 \* \*\*多卡通信\*\* 不在本次范围；若时间富余，可把打包产物对齐到 NCCL A2A 所需的“等长分片”。Tutel 的流水与并行策略可做文字对照。(\[arXiv\]\[4\]) \* \*\*论文复杂度\*\*：Chain-of-Experts 更偏模型设计，不必完整收敛，只要给出核级“串行 vs 并行”代价对比即可。(\[arXiv\]\[9\]) \* \*\*验证基线\*\*：若 block-sparse 路径实现成本偏高，可先只做紧凑打包与 padding 两个路径；把 MegaBlocks 作为文献对照。(\[arXiv\]\[6\]) --- # 小结 \* 这条路线\*\*学术上站得住\*\*（路由/结构综述 + 把 MoE 的数据移动问题“用布局代数说清楚”），\*\*工程上可交付\*\*（CuTe-DSL 原语 + 基准 + 图表）。 \* CuTe 的 \*\*composition / logical division / coalesce\*\* 正是把“分桶-切 tile-重排-合并”形式化并可直接编码的工具；示例与 API 文档都已给出。 (\[NVIDIA Docs\]\[10\]) \* 如果你希望，我可以把上面的“一页式 Proposal”转成 \*\*ICML/NeurIPS LaTeX 模板\*\*版本并附带一个 \*\*CuTe-DSL 微基准脚本\*\*骨架，直接 push 到你的仓库里。 \*\*补充参考\*\*（便于查阅） CuTe-DSL 总览/API 与布局代数文档。(\[NVIDIA Docs\]\[1\]) “通用结构同样重要”的实验观察（Metaformer 小结）。 \*\*核心论文\*\*：Shazeer17；Switch Transformer；Learning to Specialize（联合门/专家训练）；Soft-MoE；PERFT；Chain-of-Experts；系统侧 MegaBlocks/Tutel/DeepSpeed-MoE 教程。(\[arXiv\]\[2\]) \[1\]: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute\_dsl\_general/dsl\_introduction.html?utm\_source=chatgpt.com "Introduction — NVIDIA CUTLASS Documentation" \[2\]: https://arxiv.org/abs/1701.06538?utm\_source=chatgpt.com "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" \[3\]: https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf?utm\_source=chatgpt.com "THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER" \[4\]: https://arxiv.org/abs/2206.03382?utm\_source=chatgpt.com "\[2206.03382\] Tutel: Adaptive Mixture-of-Experts at Scale" \[5\]: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute\_dsl.html?utm\_source=chatgpt.com "CuTe DSL — NVIDIA CUTLASS Documentation" \[6\]: https://arxiv.org/abs/2211.15841?utm\_source=chatgpt.com "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts" \[7\]: https://arxiv.org/abs/2101.03961?utm\_source=chatgpt.com "Switch Transformers: Scaling to Trillion Parameter Models ..." \[8\]: https://arxiv.org/abs/2308.00951?utm\_source=chatgpt.com "From Sparse to Soft Mixtures of Experts" \[9\]: https://arxiv.org/pdf/2506.18945?utm\_source=chatgpt.com "Chain-of-Experts: Unlocking the Communication Power of ..." \[10\]: https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02\_layout\_algebra.html?utm\_source=chatgpt.com "CuTe Layout Algebra — NVIDIA CUTLASS Documentation" 可以，\*\*A100 非常合适\*\*做这个项目。CuTe-DSL 官方说清楚：\*\*从 Ampere（SM80）起的 NVIDIA GPU 都支持\*\*，A100 正是 Ampere 架构。(\[NVIDIA Docs\]\[1\]) # 建议配置（按“必需 → 推荐 → 进阶”分层） \*\*最低可用（做单卡内核原型与微基准就够）\*\* \* 1 张 \*\*Ampere 及以上\*\*的 NVIDIA GPU（例：A100/H100/B200；显存 ≥16–24 GB 更舒服）。CuTe-DSL 提供 `@kernel/@jit` 的 JIT 能直接在这类卡上生成核函数。(\[NVIDIA Docs\]\[1\]) \* CUDA/驱动与 CuTe-DSL 匹配：\*\*CUDA Toolkit 12.9\*\* 对应的驱动版本（官方 Quick Start/issue 都要求与 CTK 12.9 匹配，driver ≥ 525.60.13），`pip install nvidia-cutlass-dsl` 即可装 DSL。(\[NVIDIA Docs\]\[2\]) \* 基本软件：Linux（Ubuntu 常见）、Python 3.10–3.12、PyTorch（可用 \*\*DLPack\*\* 与 DSL 零拷数据互通）。(\[NVIDIA Docs\]\[3\]) \* 性能分析：\*\*Nsight Compute\*\*/CLI（`ncu`）做 SM/内存访存剖析。(\[NVIDIA Docs\]\[4\]) \*\*推荐（更顺手）\*\* \* \*\*A100 80 GB\*\*（或 40 GB 也行）：显存与带宽更充裕，做不同路由/容量因子下的 pack/combine/GEMM 更不易 OOM；A100 80 GB 带宽可达 \*\*~2 TB/s\*\*；SXM 机型还带 \*\*NVLink\*\*。(\[NVIDIA\]\[5\]) \*\*进阶（如要评估跨卡 all-to-all 路由）\*\* \* \*\*2 张以上 GPU\*\* + \*\*NVLink/高速网络\*\*，用 \*\*NCCL\*\* 的 \*\*AllToAll\*\*/P2P 做专家分发与回收的通信实验（课程项目并不强制，多卡仅用于研究扩展）。(\[NVIDIA Docs\]\[6\]) # 结论 \* 做你计划的 \*\*CuTe-DSL 实现 + 单卡核级对比\*\*：\*\*一张 A100 就能很好完成\*\*。(\[NVIDIA Docs\]\[1\]) \* 若要进一步量化 \*\*多卡路由通信\*\* 的影响，再考虑 NVLink + NCCL 的两卡/多卡平台。(\[NVIDIA Docs\]\[6\]) > 备忘：CUTLASS/CuTe 在 CUDA 11.4 起兼容，但官方推荐用 \*\*CUDA 12.x（近来 12.8/12.9）\*\*以获得最佳支持与工具链；确保驱动与 CTK 对应即可。(\[NVIDIA Docs\]\[3\]) \[1\]: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/faqs.html?utm\_source=chatgpt.com "FAQs — NVIDIA CUTLASS Documentation" \[2\]: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick\_start.html?utm\_source=chatgpt.com "Quick Start Guide — NVIDIA CUTLASS Documentation" \[3\]: https://docs.nvidia.com/cutlass/overview.html?utm\_source=chatgpt.com "Overview — NVIDIA CUTLASS Documentation" \[4\]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?utm\_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.0 documentation" \[5\]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf?utm\_source=chatgpt.com "NVIDIA A100 | Tensor Core GPU" \[6\]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html?utm\_source=chatgpt.com "Overview of NCCL — NCCL 2.28.3 documentation" 我现在的想法是使用cute-dsl完成这个项目（Topic 2: Mixture-of-Experts (MoEs) and Learning to Specialize）,其中比较难实现的代码可以直接从https://github.com/Dao-AILab/quack查看或使用相关cute-dsl实现的源码，你觉得可行吗

ChatGPT: 

Request for GPT-5 Pro

Reasoned for 7m 14s

结论先说在前面：**可行，而且是个很聪明的组合拳。**  
用 **CuTe‑DSL** 把 MoE 的 *dispatch/pack → grouped GEMM → combine/unpack* 做成可复用内核原语；对难点参考/复用 **Dao‑AILab 的 QuACK**（一套基于 CuTe‑DSL 的高性能内核）能显著降低工程门槛，同时仍保留足够的“你自己的贡献面”（MoE 路由与数据重排的设计与评测）。课程项目对“综述 + 可运行原型”的定位也完全匹配这一思路与里程碑安排。[akyrillidis.github.io+1](https://akyrillidis.github.io/comp414-514/project)

下面我把**可行性、限制与落地计划**说清楚：

---

## 1) 为什么可行

-   **QuACK 已经把“记忆/带宽瓶颈”的内核写法跑通**（RMSNorm / Softmax / Cross‑Entropy / Hopper/Blackwell GEMM+epilogue 等），并且是 **Apache‑2.0** 许可；可以按规范引用或改写，用作模板（尤其是归约、向量化、访存合并、epilogue 融合等套路）。[GitHub](https://github.com/Dao-AILab/quack)
    
-   **CuTe‑DSL 本身就是为“在 Python 里写高性能 CUDA 内核”设计**，有 JIT、与框架的 DLPack 互通、以及“布局代数”（composition / divide / coalesce …），非常适合把 MoE 的 pack/dispatch/回写抽象成**纯布局变换 + gather/scatter**。[NVIDIA Docs+2NVIDIA Docs+2](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html?utm_source=chatgpt.com)
    
-   **Grouped/多问题 GEMM 的体系是现成的**（CUTLASS 的 grouped kernel/persistent scheduler 思路），用来承接“每个专家一个小批”的 FFN 两次 GEMM 很自然。[NVIDIA Docs](https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html?utm_source=chatgpt.com)
    
-   课程项目的 **Topic 2** 明确鼓励“路由算法演进的综述 + 简单实现 + 实证对比”，你的 MoE‑in‑CuTe‑DSL 正好满足，并且可把“路由→布局/内核形状→性能”的连接讲透。[akyrillidis.github.io](https://akyrillidis.github.io/comp414-514/project)
    

---

## 2) 需要注意的限制/风险

-   **硬件要求差异**：QuACK 的 README 写得很直白——**要求 H100 或 B200（CUDA 12.9+，Python 3.12）**。如果你只有 **A100（Ampere）**，QuACK 里的 Hopper/Blackwell 专用 GEMM/Epilogue 可能就不适用，需要用 **CuTe‑DSL 自己实例化 Ampere 友好的 GEMM** 或退回到 cuBLAS/torch 的 GEMM 做对照。[GitHub](https://github.com/Dao-AILab/quack)
    
-   **MoE 专用的 pack/combine 多半还是要自己写**：QuACK 侧重通用核（归约/归一化/softmax/GEMM），**没有**现成的“按 expert 分桶打包/回写”的 MoE 内核；不过它的访存与归约写法可以直接拿来“套模板”。[GitHub](https://github.com/Dao-AILab/quack)
    
-   **学术规范**：课程页面强调项目要有“自己的问题与实现/分析”。复用开源代码时请在报告与仓库里明确**出处与改动**，并清楚标注你们的**原创模块与实验设计**。[akyrillidis.github.io](https://akyrillidis.github.io/comp414-514/project)
    

---

## 3) 推荐的项目切分：哪些“复用”，哪些“自研”

**可复用（参考/移植 QuACK 写法）：**

-   访存与归约内核的**向量化/协同归约模式**（RMSNorm/Softmax 的 tile+warp 模式）；
    
-   **GEMM+epilogue** 的组织方式（如需在 Hopper/BW 上做端到端参考）；
    
-   基准脚手架（benchmarks/test 组织结构）。  
    （以上来自 QuACK README 的内核列表与用法描述。）[GitHub](https://github.com/Dao-AILab/quack)
    

**需自研（你们的“贡献面”）：**

-   **Pack/Dispatch 内核**：按路由 top‑1/top‑k/soft 生成 `expert_id` 与 `offset`，两阶段（计数→前缀和→重排）或单阶段（分桶+局部前缀）把 `X[B,T,H]` 打成 `(E, cap, H)` 的致密批；
    
    -   这里把 **布局代数**用起来：`composition` 把原坐标映射到“(expert\_bucket, local\_idx, H)”，`divide/coalesce` 用于 tile 化与回收；
        
-   **Combine/Unpack 内核**：按 gate 权重把专家输出写回原序（top‑k 做稀疏加权、soft‑moe 做稠密加权）；
    
-   **Grouped GEMM 驱动**：基于 CUTLASS/CuTe‑DSL 的 grouped/persistent 思路跑一组“专家小 GEMM”，并与“padding 对齐 + batched GEMM”的简化路径对比。[NVIDIA Docs](https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html?utm_source=chatgpt.com)
    
-   **路由策略实验**：Top‑1（Switch）、Top‑2（Shazeer）、Soft‑MoE、（可选）Chain‑of‑Experts 的两步串行原型；报告中讨论“参数高效/动态路由趋势”。（课题页列出的代表论文就是评分参考。）[akyrillidis.github.io](https://akyrillidis.github.io/comp414-514/project)
    

---

## 4) 评测与报告建议（对齐课程要求）

**微基准（Kernel‑level）**

-   只测 MoE‑FFN 层：`pack → (GEMM1→act→GEMM2) → combine` 的各段时间、DRAM GB/s、Achieved Occupancy、GEMM TFLOP/s；
    
-   规模网格：`B×T ∈ {32×128, 64×256, 128×512}`, `E ∈ {4,8,16}`, `H ∈ {1024, 2048}`, capacity factor ∈ {1.0, 1.25}。
    
-   路径对照：**padding 路径 vs 紧凑打包路径 vs（可选）block‑sparse/MegaBlocks 思路**。
    

**端到端小模型**

-   在标准 Transformer block 上替换 FFN→MoE‑FFN（不改注意力），比较不同路由的吞吐与验证集 loss/accuracy 曲线（以“趋势”而非 SOTA 为主）。
    

**报告要点（≥6页）**

-   综述：从 “Sparsely‑Gated → Switch → Soft/Adaptive/Chain‑of‑Experts/PERFT” 的**路由/训练策略演进**；
    
-   方法：把每种路由落到**布局与内核形状**（例如 top‑k 导致的不均衡与 capacity/padding 的取舍）；
    
-   实验：图表化“路由→布局→性能”的映射与权衡；
    
-   代码可复用表（Provenance）：列出复用自 QuACK 的模块与**你们新增/改写**的内核与实验框架。  
    （以上组织与里程碑对齐课程 Project 页的要求。）[akyrillidis.github.io](https://akyrillidis.github.io/comp414-514/project)
    

---

## 5) 环境与兼容性清单

-   **仅用 CuTe‑DSL 自研（Ampere/Hopper/Blackwell 通吃）**：按官方文档安装 `nvidia-cutlass-dsl`，CUDA 12.x，PyTorch 通过 DLPack 互通；A100 也可跑（但没有 QuACK 的 Hopper/Blackwell 专用 GEMM）。[NVIDIA Docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html?utm_source=chatgpt.com)
    
-   **若要直接用 QuACK 内核**：准备 **H100/B200 + CUDA 12.9+ + Python 3.12**，`pip install quack-kernels`。[GitHub](https://github.com/Dao-AILab/quack)
    

---

## 6) 一条“最小可交付”路线图

1.  **第 1 周**：完成 Top‑1 路由 + **pack/combine** 的 CuTe‑DSL 原型（先不融合 GEMM）；微基准记录 `pack/combine` 的 GB/s。
    
2.  **第 2–3 周**：接入 **grouped GEMM**（CuTe/CUTLASS DSL 或 torch/cublas 对照），补上 Top‑2；做 *padding vs 紧凑* 的对照。[NVIDIA Docs](https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html?utm_source=chatgpt.com)
    
3.  **第 4 周**：加入 **soft‑MoE** 回写（加权合成路径），完成端到端 tiny 模型实验；撰写方法与实验部分。
    
4.  **收尾**：补充 Chain‑of‑Experts 的微原型（如时间允许），整理“路由→布局”图谱与负载均衡讨论，完成报告与 slides。
    

---

## 7) 合规复用与署名建议

-   在 README/报告中加 **“Code provenance”** 表：列出 *QuACK* 中直接引用/改写的文件与函数（含版本与链接），以及你们新增的 `moe_pack.cu`, `moe_combine.cu`, `grouped_gemm_driver.py` 等；
    
-   报告正文与参考文献同时**引用 QuACK 仓库与 CuTe‑DSL 文档**；仓库 LICENSE 保留 Apache‑2.0 头。[GitHub+1](https://github.com/Dao-AILab/quack)
    

---

### 小结

-   **可做而且值得做**：用 CuTe‑DSL 写 MoE 的 *pack/combine + grouped GEMM*，把不同路由策略的系统开销“落到布局/核”层解释清楚；
    
-   **QuACK 能帮你少踩很多坑**（内存/访存优化套路与性能基准），但要注意它目前**偏 Hopper/Blackwell**；A100 用户建议以 **CuTe‑DSL 自研内核**为主，把 QuACK 作为参考与（可选的）对照实现。[GitHub+1](https://github.com/Dao-AILab/quack)
    

如果你愿意，我可以**按上面路线直接给你一份“一页式 Proposal（中文/英文）+ 评测脚手架结构清单”**，把 QuACK 复用边界也写进提案里，便于 10/10 的项目申报。

You: 

那么我们关注的更多的是推理还是训练？每次实验花费的时间长吗？如果代码写好的话。目前听你的分析好像是比较快的，不像训练要花费很长时间

ChatGPT: 

Request for GPT-5 Pro

Reasoned for 4m 38s

结论先行：**我们把精力放在“推理（inference）与内核级微基准”为主**，只在需要验证“学习到专长（learning to specialize）”这一点时，**做很轻量的训练（如只训 gate 或做小规模微调）**。  
这样一来，**单次实验通常是“分钟级”**；只有在做成套大规模评测或参数网格搜索时才会到“小时级”。你的理解是对的：这类项目的主要成本在**把内核与评测脚手架写好**，而不是像大模型训练那样长时间占 GPU。

---

## 我们具体关注什么？

-   **主线（建议）**：以 **CuTe‑DSL** 实现 MoE 的 **pack/dispatch → grouped GEMM → combine/unpack** 核心原语，比较 **不同路由策略（Top‑1/Top‑k/Soft）** 在**单卡**上的吞吐与带宽利用率。
    
-   **可选拓展**：
    
    -   **学习到专长**：冻结专家 FFN，只**微调 gating（或门+很薄的一层）**，用小数据集验证“是否出现专家分工”，这通常是**小时级**而不是天级。
        
    -   **多卡通信**：如果时间允许，再看 all‑to‑all 带来的系统差异；不是必需。
        

---

## 单次实验需要多久？（可用来预估你自己的集群时间）

推理与评测的时间主要由**总 token 量**与\*\*系统吞吐（tokens/s）\*\*决定。一个简单的估算公式：

$$
\text{总时长} \;\approx\; \frac{N \times (L_\text{prompt} + L_\text{gen})}{\text{吞吐 (tokens/s)} \times \text{并行度}} \;+\; \text{固定开销}
$$

-   $N$：样本数；$L_\text{prompt}$/$L_\text{gen}$：平均提示/生成 token 数。
    
-   “固定开销”包括数据加载、首次 JIT/预热、日志等，一般是分钟级。
    

**两个参考情景**（只是示例，方便你代入自己的吞吐）：

1.  $N=2{,}000$，$L_\text{prompt}=300$，$L_\text{gen}=150$，整机有效吞吐 $=3{,}000$ tok/s：  
     总 token = $2{,}000 \times 450 = 900{,}000$；用时 $= 900{,}000/3{,}000 = 300$ 秒 ≈ **5 分钟**。
    
2.  $N=10{,}000$，$L_\text{prompt}=500$，$L_\text{gen}=300$，整机吞吐 $=1{,}000$ tok/s：  
     总 token = $10{,}000 \times 800 = 8{,}000{,}000$；用时 $= 8{,}000$ 秒 ≈ **2.22 小时**。
    

> 关键点：**一旦内核与评测脚手架稳定**，你会发现大多数“对比实验”都是**分钟级**；只有当你扩大数据量、做多组超参/多路由/多模型的**批量评测**时，才会叠加成小时级。

---

## 为什么训练要慢得多？

-   **计算路径**：训练 = 前向 + 反向 + 优化器更新，**每步 FLOPs ~ 2–4× 推理**；再乘以**遍历多轮数据**（epochs）。
    
-   **系统负载**：反向传播对显存、通信更敏感，调参也更频繁。
    
-   **结论**：即便是很小的 MoE 训练，也常从**小时**起步；而你的项目重点（内核与路由对比）**完全不需要长时间训练**。
    

---

## 怎么把推理实验做得“又快又稳”？

1.  **两阶段评测**：
    
    -   **微基准**（单层 MoE）：对不同路由/容量因子/打包策略，测 kernel 时间、GB/s、TFLOP/s（分钟级）。
        
    -   **端到端小模型**：在固定验证集上测 tokens/s 与准确率/困惑度（10–60 分钟级）。
        
2.  **批处理与长度控制**：按序列长度分桶；把 `max_new_tokens` 设到“够用但不浪费”；避免极端长尾样本拖慢整体。
    
3.  **高效后端**：使用 **CuTe‑DSL** 融合的 pack/combine 与 **grouped GEMM**；推理用 **fp16/bf16**，开启张量核心路径。
    
4.  **避免 Python 热点**：不要在 Python 里循环 scatter/gather；把重排/写回尽量进 DSL 内核（这一步对速度最关键）。
    
5.  **预热与固定开销**：先跑一小批次预热（JIT/内存池/内核选择），再计时。
    
6.  **并行与流水**：数据准备与推理分进程/线程，开启 pinned memory、异步拷贝；评测时避免逐 token 打印。
    
7.  **科学抽样**：先在 50–200 条上做“筛查”，只对明显有希望的设定上全量集。用简单的置信区间/自举评估差异显著性。
    

---

## 如果要验证“Learning to Specialize”的训练部分，怎么控制时间？

-   **只训 gate（或 gate+轻薄层），冻结专家**：小数据上几万～几十万 token 就能看到路由熵下降/专家分工迹象，通常**1–3 小时**可完成一次实验。
    
-   **PEFT/LoRA 路由化微调**：在 MoE 上只路由需要的子模块，同样可以小时级完成一次小实验。
    
-   **端到端大规模 MoE 预训练**：不在本项目范围（天级+多卡），没必要。
    

---

### 小结

-   我们**以推理和内核微基准为主**，必要时做**极轻量**的 gate 训练验证“专长”。
    
-   **写好 DSL 内核与评测脚手架后**，一次实验一般就是**几分钟**；只有当你做大规模成套评测时才会到**小时级**。
    
-   真正影响时长的是**数据规模与并行度**，不是“是否是 MoE”。
    

如果你愿意，我可以把上面“估时公式 + 预热 & 计时脚本骨架 + 实验清单（Top‑1 / Top‑k / Soft 路由）”整理成一个最小可运行的 **benchmark harness**（含命令行参数），你只需把你的模型与数据路径填进去就能开始测。

You: 

我现在已经创建了一个private repo叫做https://github.com/yuninxia/moe-routing-bench，现在的文件结构是\[yx87@ufront moe-routing-bench\]$ pwd /home/yx87/pg/moe-routing-bench \[yx87@ufront moe-routing-bench\]$ tree . ├── 3rdparty │ └── quack │ ├── CLAUDE.md │ ├── LICENSE │ ├── README.md │ ├── benchmarks │ │ ├── benchmark\_cross\_entropy.py │ │ ├── benchmark\_gemm\_sm90.py │ │ ├── benchmark\_layernorm.py │ │ ├── benchmark\_rmsnorm.py │ │ ├── benchmark\_softmax.py │ │ ├── benchmark\_topk.py │ │ ├── pytorch\_benchmark\_rmsnorm.py │ │ ├── pytorch\_benchmark\_rmsnorm\_backward.py │ │ └── visual\_outputs │ │ ├── quack\_vs\_pytorch\_backward\_speedup.png │ │ ├── quack\_vs\_pytorch\_speedup.png │ │ ├── quack\_vs\_torchcompile\_backward\_speedup.png │ │ ├── quack\_vs\_torchcompile\_speedup.png │ │ ├── rmsnorm\_backward\_speedup\_comparison.png │ │ └── rmsnorm\_speedup\_comparison.png │ ├── docs │ │ ├── dsl\_control\_flow.rst │ │ └── limitations.rst │ ├── media │ │ ├── 2025-07-10-membound-sol.md │ │ ├── QuACK.png │ │ ├── bf16\_kernel\_benchmarks\_single\_row.svg │ │ ├── block\_reduction.png │ │ ├── cluster\_reduction.png │ │ ├── combined\_kernel\_benchmarks\_final.png │ │ ├── dsmem.png │ │ ├── gpu-memory-hierarchy.png │ │ ├── liger-16k-65k-ncu.png │ │ ├── liger-16k-65k.png │ │ ├── max\_reduction.png │ │ ├── memory-access-hierarchy.png │ │ ├── our-16k-131k-arithmetic-intensity-white.png │ │ ├── our-16k-131k-sol.png │ │ ├── our-16k-131k.png │ │ ├── productivity-performance.png │ │ ├── pytorch-16k-131k.png │ │ ├── thread\_reduction.png │ │ ├── tv\_layout.png │ │ ├── warp\_reduction.png │ │ └── warp\_storage.png │ ├── pyproject.toml │ ├── quack │ │ ├── \_\_init\_\_.py │ │ ├── activation.py │ │ ├── autotuner.py │ │ ├── copy\_utils.py │ │ ├── cross\_entropy.py │ │ ├── cute\_dsl\_utils.py │ │ ├── fast\_math.py │ │ ├── gemm.py │ │ ├── gemm\_act.py │ │ ├── gemm\_config.py │ │ ├── gemm\_dact.py │ │ ├── gemm\_default\_epi.py │ │ ├── gemm\_interface.py │ │ ├── gemm\_sm100.py │ │ ├── gemm\_sm90.py │ │ ├── gemm\_symmetric.py │ │ ├── gemm\_wrapper\_utils.py │ │ ├── layernorm.py │ │ ├── linear.py │ │ ├── linear\_cross\_entropy.py │ │ ├── mlp.py │ │ ├── pipeline.py │ │ ├── reduce.py │ │ ├── reduction\_base.py │ │ ├── rmsnorm.py │ │ ├── sm90\_utils.py │ │ ├── softmax.py │ │ ├── sort │ │ │ ├── bitonic\_sort.py │ │ │ ├── generate\_sorting\_networks.py │ │ │ ├── sorting\_networks.py │ │ │ └── utils.py │ │ ├── tensormap\_manager.py │ │ ├── tile\_scheduler.py │ │ ├── topk.py │ │ ├── utils.py │ │ └── varlen\_utils.py │ └── tests │ ├── test\_cross\_entropy.py │ ├── test\_layernorm.py │ ├── test\_linear.py │ ├── test\_linear\_cross\_entropy.py │ ├── test\_linear\_varlen\_k.py │ ├── test\_linear\_varlen\_m.py │ ├── test\_rmsnorm.py │ ├── test\_softmax.py │ ├── test\_symmetric\_gemm.py │ └── test\_topk.py ├── LICENSE └── README.md 9 directories, 89 files \[yx87@ufront moe-routing-bench\]$ 你可以看到我目前只是把quark当做一个第三方库放到3rdparty当中，现在我希望继续在这个项目中开始实现我们的project，下一步你有什么建议吗？我已经安装了这两个package，这体现在了我现在的readme中： # moe-routing-bench ## Env Setup

```
bash
pip3 install nvidia-cutlass-dsl
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
```

我的偏好是你可以给我具体的plan，并给我可以work on并迭代的代码，这样是最好的。当然，你可以用你觉得最合适的方式给我建议

ChatGPT: 

Request for GPT-5 Pro

Finished reasoning

太好了，你已经把 **QuACK** 作为第三方子模块 vendoring 进来了（`3rdparty/quack`）。接下来我建议用“**能跑起来的最小可用版本 (MVP)** → 持续加速 → 加入更多实现”的路线推进，并且把结构一次性搭好，后面迭代只需要在既有骨架里加代码/替换实现即可。

下面给出一个**两周内就能形成可复现基线**的计划 + 可直接粘贴到仓库里的代码骨架（含 CLI、基准框架、路由 reference 实现、与 QuACK 的可选集成钩子）。你可以先把这些文件按路径新建，`pip install -e .` 之后就能跑起来，后续我们在这些入口里不断替换/优化为 QuACK Kernel、CUTLASS DSL 或 Triton 等实现。

---

## 目标与阶段计划

**总体目标**：做一个针对 MoE Routing 的可复现基准套件，覆盖：

-   **Gating + Top‑k 选择**（logits → top‑k indices & combine weights）
    
-   **Token Dispatch（按专家重排/打包）**
    
-   **Combine（专家输出回写并按权重聚合）**
    
-   （可选）**专家计算占位**（先用 identity / shared linear；后续换成 per‑expert GEMM）
    

**第 1 阶段（今天就能完成）**

1.  搭建包结构与可运行 CLI（`moe-routing-bench` 可 `pip install -e .`）。
    
2.  提供 **纯 PyTorch baseline**：top‑k、dispatch（排序重排法）、combine（逆重排+按权重求和）。
    
3.  单测验证：路由 + identity 专家 ≡ 直接对输入按 top‑k 权重求和的结果。
    
4.  基准脚本：统计 `latency / tokens/s`，保存 JSON/CSV 结果。
    

**第 2 阶段（接下来 1–3 天）**

1.  集成 **QuACK Top‑k**（已在骨架预留 wrapper，try/except 自动回退到 torch）。
    
2.  增加 gating 线性层（可切换 PyTorch / QuACK Linear）。
    
3.  跑一组典型形状：`tokens ∈ {8K, 16K, 32K}`，`experts ∈ {64, 128, 256}`，`k=2`，`hidden=4096`。
    

**第 3 阶段（1 周内）**

1.  加入 **分阶段剖析**（单独计时 gating、top‑k、dispatch、combine），并输出到 CSV。
    
2.  增加 per‑expert GEMM（起步用共享 Linear 或 identity；后续替换为 QuACK GEMM）。
    
3.  Nsight / NVTX 标记（可视化热点）；容量限制（capacity factor）与丢弃策略基线。
    

---

## 目录建议

```bash
moe-routing-bench/
├── pyproject.toml               # 顶层打包文件（让 3rdparty/quack 被发现）
├── moe_routing_bench/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── topk_impls.py            # torch / quack topk 统一接口
│   ├── routing.py               # gating/topk/dispatch/combine 参考实现
│   ├── bench.py                 # 基准测量函数
│   └── cli.py                   # 可运行 CLI
├── tests/
│   └── test_identity_routing.py # 单测：identity 专家等价性
└── README.md                    # 使用说明（在你的现有 README 基础上追加）
```

> 说明：我们用 `pyproject.toml` 让 `3rdparty/quack/quack` 作为包 `quack` 被 setuptools 发现，这样 `import quack` 就能工作（无需改 `PYTHONPATH`）。

---

## 1) 顶层 `pyproject.toml`

把下面文件放到仓库根目录（与 `README.md` 同级）：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "moe-routing-bench"
version = "0.0.1"
description = "Micro/macro benchmarks for Mixture-of-Experts routing components"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.3",           # 你已手动装了 cu128 版，这里只做版本下限
  "nvidia-cutlass-dsl",   # 你已手动安装
  "numpy>=1.24",
]

[tool.setuptools.packages.find]
where = [".", "3rdparty/quack"]
include = ["moe_routing_bench*", "quack*"]

[tool.pytest.ini_options]
addopts = "-q"
```

安装（一次）：

```bash
pip install -e .
```

---

## 2) `moe_routing_bench/__init__.py`

```python
__all__ = ["config", "utils", "topk_impls", "routing", "bench"]
```

---

## 3) `moe_routing_bench/config.py`

```python
from dataclasses import dataclass

@dataclass
class BenchConfig:
    num_tokens: int = 16384     # B*S
    hidden_dim: int = 4096
    num_experts: int = 128
    top_k: int = 2
    dtype: str = "float16"      # "float16" | "bfloat16" | "float32"
    device: str = "cuda"        # "cuda" | "cpu"
    seed: int = 17
    warmup: int = 10
    iters: int = 100
```

---

## 4) `moe_routing_bench/utils.py`

```python
import contextlib
import time
import torch
from typing import Tuple

def get_dtype(name: str):
    name = name.lower()
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    return torch.float32

@contextlib.contextmanager
def cuda_timer(sync: bool = True):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[timer] { (t1 - t0) * 1e3:.3f} ms")

def measure_latency_ms(fn, warmup=10, iters=100, sync=True) -> Tuple[float, float]:
    # returns (avg_ms, std_ms)
    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    import numpy as np
    return float(np.mean(times)), float(np.std(times))
```

---

## 5) `moe_routing_bench/topk_impls.py`

```python
from typing import Tuple
import torch

def torch_topk(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns (values, indices)
    vals, idx = torch.topk(logits, k, dim=-1)
    return vals, idx

def quack_topk(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Try common signatures from 3rdparty/quack. Falls back to torch.topk on mismatch.
    """
    try:
        from quack.topk import topk as quack_topk_fn  # type: ignore
    except Exception:
        return torch_topk(logits, k)

    try:
        out = quack_topk_fn(logits, k)
    except Exception:
        return torch_topk(logits, k)

    # Normalize to (values, indices)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        v, i = out
        return v, i
    # some implementations might return only indices
    if torch.is_tensor(out) and out.dtype in (torch.int32, torch.int64):
        idx = out
        vals = torch.gather(logits, dim=-1, index=idx)
        return vals, idx

    # unknown shape/signature -> safe fallback
    return torch_topk(logits, k)

def run_topk(logits: torch.Tensor, k: int, impl: str):
    impl = impl.lower()
    if impl == "quack":
        return quack_topk(logits, k)
    return torch_topk(logits, k)
```

---

## 6) `moe_routing_bench/routing.py`

> 这部分给出**路由参考实现**：
> 
> -   gating（线性层可选）
>     
> -   top‑k（可选用 torch / quack）
>     
> -   dispatch（排序重排法：向量化，无循环）
>     
> -   combine（逆重排 + 权重聚合）
>     
> -   identity 专家端到端（做基线验证与计时）
>     

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

from .topk_impls import run_topk
from .utils import get_dtype

# ---------- Gating ----------

class GatingLinear(nn.Module):
    """
    Simple gating: logits = x @ W_g + b
    """
    def __init__(self, hidden_dim: int, num_experts: int, dtype=torch.float16, device="cuda"):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts, bias=True, device=device, dtype=dtype)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # [T, E]

def topk_logits_then_softmax_over_k(
    logits: torch.Tensor,
    k: int,
    impl: str = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    先在完整专家维度上做 top-k，得到 (values, indices)；
    再对每个 token 的 k 个 logits 做 softmax，得到 combine weights。
    返回: (indices:[T,k], weights:[T,k])
    """
    vals, idx = run_topk(logits, k, impl=impl)
    weights = torch.softmax(vals, dim=-1).to(vals.dtype)
    return idx, weights

# ---------- Dispatch / Combine ----------

@torch.no_grad()
def pack_by_expert(
    x: torch.Tensor,              # [T, H]
    topk_idx: torch.Tensor,       # [T, k]
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    向量化的“排序重排法”打包：
      - 展平 topk_idx -> [M=T*k]
      - 对 expert id 排序，得到排序索引 order
      - 重复 x (repeat_interleave k 次) 并按 order 重排 -> packed: [M, H]
      - 返回 packed、order、counts（每个专家的条目数）、offsets（每个专家起始位置的前缀和）
    """
    T, H = x.shape
    k = topk_idx.shape[1]
    flat_eids = topk_idx.reshape(-1)                       # [M]
    order = torch.argsort(flat_eids)                       # [M]
    x_rep = x.repeat_interleave(k, dim=0)                  # [M, H]
    packed = x_rep.index_select(0, order)                  # [M, H]
    counts = torch.bincount(flat_eids, minlength=num_experts)  # [E]
    offsets = torch.cumsum(counts, dim=0)                  # [E]
    return packed, order, counts, offsets

@torch.no_grad()
def unpack_to_tokens_and_combine(
    expert_outputs_sorted: torch.Tensor,  # [M, H]，与 pack_by_expert 的 packed 同序
    order: torch.Tensor,                  # [M]
    num_tokens: int,
    k: int,
    combine_weights: torch.Tensor,        # [T, k]
) -> torch.Tensor:
    """
    将按专家连续的输出还原回 (token, k) 的原顺序，再按 combine weights 在 k 维做加权求和。
    """
    M, H = expert_outputs_sorted.shape
    assert M == num_tokens * k
    # 逆排序：把 [sorted] -> [entry-order]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(M, device=order.device)
    unsorted = expert_outputs_sorted.index_select(0, inv)  # [M,H] (entry-order)
    # entry-order 是 token 0 的 k 个条目、token 1 的 k 个条目、...
    unsorted = unsorted.view(num_tokens, k, H)             # [T,k,H]
    y = (unsorted * combine_weights.unsqueeze(-1)).sum(dim=1)  # [T,H]
    return y

# ---------- End-to-end (identity experts) ----------

def route_identity_pipeline(
    x: torch.Tensor,                         # [T,H]
    num_experts: int,
    topk_idx: torch.Tensor,                  # [T,k]
    combine_weights: torch.Tensor,           # [T,k]
) -> Tuple[torch.Tensor, dict]:
    """
    用 identity 专家（不做计算）走完整路由路径，返回输出与一些中间统计信息。
    """
    T, H = x.shape
    k = topk_idx.shape[1]
    packed, order, counts, offsets = pack_by_expert(x, topk_idx, num_experts)
    # identity专家：输出=输入
    expert_outputs_sorted = packed
    y = unpack_to_tokens_and_combine(expert_outputs_sorted, order, T, k, combine_weights)
    stats = {
        "counts": counts,
        "offsets": offsets,
    }
    return y, stats

# ---------- 便捷入口：从 logits 直接跑完 ----------

def end2end_identity_from_logits(
    x: torch.Tensor,                         # [T,H]
    logits: torch.Tensor,                    # [T,E]
    k: int,
    impl_topk: str = "torch",
) -> torch.Tensor:
    idx, w = topk_logits_then_softmax_over_k(logits, k=k, impl=impl_topk)
    y, _ = route_identity_pipeline(x, logits.shape[1], idx, w)
    return y

# ---------- 辅助：生成伪数据 ----------

def make_fake_batch(num_tokens: int, hidden: int, dtype="float16", device="cuda") -> torch.Tensor:
    dt = get_dtype(dtype)
    g = torch.Generator(device=device).manual_seed(42)
    return torch.randn(num_tokens, hidden, device=device, dtype=dt, generator=g)
```

---

## 7) `moe_routing_bench/bench.py`

```python
import json
from dataclasses import asdict
from typing import Dict, Any
import torch
from .config import BenchConfig
from .routing import (
    GatingLinear,
    topk_logits_then_softmax_over_k,
    pack_by_expert,
    unpack_to_tokens_and_combine,
    make_fake_batch,
)
from .utils import measure_latency_ms, get_dtype

def bench_topk(cfg: BenchConfig, impl: str = "torch") -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dt = get_dtype(cfg.dtype)

    x = make_fake_batch(cfg.num_tokens, cfg.hidden_dim, cfg.dtype, device)
    gating = GatingLinear(cfg.hidden_dim, cfg.num_experts, dtype=dt, device=device)
    logits = gating(x)  # [T,E]

    def run_once():
        idx, w = topk_logits_then_softmax_over_k(logits, cfg.top_k, impl=impl)
        torch._assert(idx.shape == (cfg.num_tokens, cfg.top_k), "shape check")
        return idx, w

    avg, std = measure_latency_ms(run_once, warmup=cfg.warmup, iters=cfg.iters)
    toks_per_s = cfg.num_tokens / (avg / 1e3)

    return {
        "name": "topk",
        "impl": impl,
        "avg_ms": avg,
        "std_ms": std,
        "tokens_per_s": toks_per_s,
        **asdict(cfg),
    }

def bench_routing_identity(cfg: BenchConfig, impl_topk: str = "torch") -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dt = get_dtype(cfg.dtype)

    x = make_fake_batch(cfg.num_tokens, cfg.hidden_dim, cfg.dtype, device)
    gating = GatingLinear(cfg.hidden_dim, cfg.num_experts, dtype=dt, device=device)
    logits = gating(x)  # [T,E]

    idx, w = topk_logits_then_softmax_over_k(logits, cfg.top_k, impl=impl_topk)

    def run_once():
        packed, order, counts, offsets = pack_by_expert(x, idx, cfg.num_experts)
        # identity experts
        y = unpack_to_tokens_and_combine(packed, order, cfg.num_tokens, cfg.top_k, w)
        return y

    avg, std = measure_latency_ms(run_once, warmup=cfg.warmup, iters=cfg.iters)
    toks_per_s = cfg.num_tokens / (avg / 1e3)

    return {
        "name": "routing_identity",
        "impl_topk": impl_topk,
        "avg_ms": avg,
        "std_ms": std,
        "tokens_per_s": toks_per_s,
        **asdict(cfg),
    }

def save_jsonl(record: Dict[str, Any], path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

---

## 8) `moe_routing_bench/cli.py`

```python
import argparse
import json
from .config import BenchConfig
from .bench import bench_topk, bench_routing_identity, save_jsonl

def parse_args():
    p = argparse.ArgumentParser(prog="moe-routing-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--num-tokens", type=int, default=16384)
        sp.add_argument("--hidden-dim", type=int, default=4096)
        sp.add_argument("--num-experts", type=int, default=128)
        sp.add_argument("--top-k", type=int, default=2)
        sp.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
        sp.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
        sp.add_argument("--seed", type=int, default=17)
        sp.add_argument("--warmup", type=int, default=10)
        sp.add_argument("--iters", type=int, default=100)
        sp.add_argument("--out", type=str, default="")  # jsonl path

    sp1 = sub.add_parser("bench-topk", help="Benchmark top-k selection")
    add_common(sp1)
    sp1.add_argument("--impl", type=str, default="torch", choices=["torch","quack"])

    sp2 = sub.add_parser("bench-routing", help="Benchmark dispatch+combine with identity experts")
    add_common(sp2)
    sp2.add_argument("--impl-topk", type=str, default="torch", choices=["torch","quack"])

    return p.parse_args()

def main():
    args = parse_args()
    cfg = BenchConfig(
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
    )

    if args.cmd == "bench-topk":
        rec = bench_topk(cfg, impl=args.impl)
    elif args.cmd == "bench-routing":
        rec = bench_routing_identity(cfg, impl_topk=args.impl_topk)
    else:
        raise RuntimeError("unknown cmd")

    print(json.dumps(rec, indent=2))
    if args.out:
        save_jsonl(rec, args.out)

if __name__ == "__main__":
    main()
```

运行示例（GPU）：

```bash
python -m moe_routing_bench.cli bench-topk --impl torch --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 200
python -m moe_routing_bench.cli bench-topk --impl quack  # 若 quack.topk 可用会自动调用
python -m moe_routing_bench.cli bench-routing --impl-topk torch
```

---

## 9) `tests/test_identity_routing.py`

```python
import torch
from moe_routing_bench.routing import (
    GatingLinear,
    topk_logits_then_softmax_over_k,
    route_identity_pipeline,
    make_fake_batch,
)
from moe_routing_bench.utils import get_dtype

def test_identity_routing_equivalence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, H, E, K = 1024, 256, 32, 2
    dtype = get_dtype("float16" if device == "cuda" else "float32")

    x = make_fake_batch(T, H, "float16" if device=="cuda" else "float32", device)
    gating = GatingLinear(H, E, dtype=dtype, device=device)
    logits = gating(x)

    idx, w = topk_logits_then_softmax_over_k(logits, k=K, impl="torch")
    # 路由 + identity 专家
    y, _ = route_identity_pipeline(x, E, idx, w)

    # 直接对输入按 top-k 权重求和（因为专家是 identity）
    # 注意：直接加权的输入是 x（每个 token 的相同 x_i 重复 K 次）
    x_rep = x.unsqueeze(1).expand(T, K, H)
    y_ref = (x_rep * w.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
```

运行：

```bash
pytest -q
```

---

## 10) README 追加用法（示例）

在你的 `README.md` 末尾追加：

```markdown
## Usage

### Install (editable)
```bash
pip install -e .
```

### Quick benches

```bash
# Top-k (torch)
python -m moe_routing_bench.cli bench-topk --impl torch --num-tokens 16384 --num-experts 128 --top-k 2 --hidden-dim 4096 --iters 200

# Try QuACK Top-k (falls back to torch if not available)
python -m moe_routing_bench.cli bench-topk --impl quack

# Routing (dispatch + combine) with identity experts
python -m moe_routing_bench.cli bench-routing --impl-topk torch
```

### Tests

```bash
pytest -q
```

```markdown
---

## 接下来怎么迭代（优先级从高到低）

1. **实测 QuACK Top‑k**  
   - 在 `topk_impls.py` 里完善 `quack_topk` 签名适配（根据 `3rdparty/quack/quack/topk.py` 中函数真实返回值调整）。  
   - `bench-topk` 跑 torch vs quack（相同输入与 dtype），确立基础加速比。

2. **Gating Linear 用 QuACK**  
   - 在 `routing.py` 里为 `GatingLinear` 增加后端选择（torch / quack）。QuACK 已有 `linear.py` / `gemm.py`，可参考其 `quack/tests` 与 `benchmarks`。  
   - 若短期内不想碰复杂 API，先只替换 top‑k，就能看见 end2end 的一部分收益。

3. **Dispatch 优化**  
   - 当前参考实现用 “排序重排” 法（稳定且向量化）；后续可以引入 prefix‑sum / radix 的纯 GPU kernel，减少排序开销。  
   - 你也可以尝试用 QuACK 写一个简单的**分桶写入** kernel（atomic / scan 变体），形成 *dispatch_quack* 实现，与参考实现对比。

4. **统计与可视化**  
   - `bench.py` 里把阶段用 `NVTX` 标记（`torch.cuda.nvtx`），便于 Nsight Systems/Compute 分析。  
   - 输出 CSV / JSONL（已支持），用 notebook 画图（后续我可以给你一个小脚本）。

5. **容量管理**  
   - 增加 `capacity_factor` 与溢出丢弃策略（dummy 实现即可）。测试不同 `E, K, cap` 下的吞吐/丢弃率权衡。

6. **专家计算（可选）**  
   - 先用共享 Linear 验证 pipeline，然后切换到 per‑expert GEMM（QuACK/CUTLASS）。  
   - 注意 pack 后的按专家批形状不均衡，最好构造 *ragged* 批处理或分专家循环（再考虑 kernel 合并）。

---

## 小提示与坑位

- **包发现**：顶层 `pyproject.toml` 已把 `3rdparty/quack` 加入 `where`，`pip install -e .` 后 `import quack` 能找到 vendored 包。  
- **CUDA/Driver**：你已安装 `torch --index-url https://download.pytorch.org/whl/cu128`；确保驱动 ≥ CUDA 12.8 对应版本。首次运行 QuACK 内核会有编译/缓存开销，基准前多做若干 warmup。  
- **dtype**：优先 `float16` 或 `bfloat16`。如需验证数值稳定性，增加 `float32` 跑 correctness（非性能）。  
- **随机数与可复现**：已在 `BenchConfig.seed` 和数据生成中固定；QuACK 内核如涉及 autotune，要记录选择的配置。  

---

如果你把上述文件加进去并 `pip install -e .`，现在就能跑：
- `bench-topk`（torch/可选 quack）
- `bench-routing`（identity 专家）
- `pytest` 单测验证等价性

等你运行一遍（尤其是 `bench-topk --impl quack`），把输出/报错贴给我，我会基于真实输出帮你**完善 QuACK 接口适配**和**下一步的内核替换建议**。
::contentReference[oaicite:0]{index=0}
```

目前的运行结果如何？

结论先说在前面：

* 你现在这组 **micro (Top‑K)** 和 **routing-only (identity experts)** 的基准结果**是合理的**。
  *Top‑K* 微基准里，`quack` 比 `torch` 版本快约 **2.4%**（0.3157→0.3083 ms）；
  但在 **routing_identity**（包含 *top‑k + 打包/回收*）里，两者差异 **≈0.02 ms**，远小于抖动，**几乎不可见**——这很符合预期：路由阶段的*主要瓶颈不是 top‑k 自身*，而是 **pack/unpack（gather/scatter/重排）** 等内存带宽受限操作。

---

## 为什么说“符合预期”？（用你的数字快速校验）

以 `bench-routing` 为例（N=16,384 tokens，H=4,096，K=2，FP16）：

* 仅就 **数据搬运** 粗略估算：
  每个 token 在 *dispatch* 需要搬 `H × K × 2 bytes = 4096×2×2 = 16,384 bytes`；
  *combine* 另一次同量搬运；**往返共 ~32 KB/token**。
  总量 ≈ `16,384 tokens × 32 KB ≈ 512 MiB`。
* 你的计时 `~3.40 ms` ⇒ 实测带宽 ≈ `512 MiB / 3.40 ms ≈ 150–160 GB/s`。
  对单卡（特别是随机访问/重排型访问）来说，这是一个**非常合理**的“有效带宽”水平（远低于 HBM 峰值是正常的，重排访问+核启动/同步开销很难吃满带宽）。

而 *Top‑K 选择* 本身只做一次 `topk(logits)`，强度小得多，所以 **微基准能看出 2–3% 的实现差异**，但放到包含 **pack/unpack** 的 *routing* 总时间里，这点差异只占总时长 ~9–10% 的一小部分，**贡献到总时长的边际收益 <0.3%**，被噪声淹没——与你的测得现象吻合。

> 粗略算一下份额：`T_total ≈ 3.40 ms`，`T_topk ≈ 0.31 ms`，
> `quack` 在 Top‑K 上省 ~0.0075 ms ⇒ 对总时长的节省约 **0.22%**（很难测出来）。

---

## 动态路由算法要比较什么“性能”？

分成 **系统指标** 与 **模型质量** 两大块。课程项目的叙述也强调“参数高效 & 动态路由”，所以建议做**双指标帕累托**（系统速度/成本 vs. 质量）。

### 1) 系统侧（训练 & 推理通用）

**主指标（建议在每个实验图中都报）：**

* **吞吐（throughput）**：tokens/s（或 sequences/s）。
  训练用 *samples/s* + *tokens/s* 都要报；推理一般用 *tokens/s*。
* **延迟（latency）**：单步 forward/backward 的 wall‑time；推理看 P50/P95。
* **路由开销占比**：Top‑K、pack、all‑to‑all（若多卡）、combine 各自的时间份额。
* **显存占用**：峰值显存、激活占用；不同 `top_k` / 容量因子 CF 下的变化。
* **（如多卡）通信开销**：All‑to‑All 的时间/占比、传输字节数、有效带宽。

**辅助指标：**

* **专家负载均衡度**：tokens-per-expert 的 **CV 值** 或与均匀分布的 **KL**；
  **溢出/丢弃率**（capacity factor 导致的 drop）。
* **有效带宽**：以「搬运字节 / 时间」折算（就像上面的 150–160 GB/s）。
* **可扩展性**：弱/强扩展（Experts 数、`top_k`、序列长、batch 的变化曲线）。

### 2) 模型质量侧（训练为主，推理看保真/稳健性）

* **验证困惑度/损失**（LM）：如 ppl on held‑out set。
* **任务指标**：分类/问答/翻译等的 Accuracy/F1/ROUGE…（选 1–2 个小任务即可）。
* **收敛速度/样本效率**：达到相同 ppl/accuracy 所需的 **训练 tokens** 或 **时间**。
* **稳定性**：是否出现**专家塌缩**（负载严重不均）、是否需要 z‑loss/aux‑loss 才稳定。

> **如何读结果**：
> 动态路由 A 优于 B 的充分证据是：**在相同计算/时间预算下**，A 达到 **更低的 ppl 或更高的准确率**；或 **在相同质量水平下**，A 的 **tokens/s 更高 / 成本更低**。
> 这就是你报告里推荐的 **帕累托曲线**（质量 vs. 速度/成本）。

---

## 具体到你项目里的“下一步怎么量”

你现在已经有：

* `bench-topk`（Top‑K 微基准）
* `bench-routing`（路由打包/回收 + identity expert）

建议**马上补齐**下面 3 类基准（先单卡）：

1. **pack/unpack 微基准（最关键）**
   单独计时 *pack*（按 expert 分桶重排）与 *combine*（按 gate 写回/加权）。

   * 先给一个 **Torch baseline**（排序+索引：稳定但不快），跑通功能与校验；
   * 再在同一接口下挂上 **QuACK/CuTe‑DSL** 版本（收到正确性回归后再比速度）。
   * **输出**：avg ms / std / tokens/s / 有效带宽（用上面的 32KB/token 折算）。

2. **MoE‑FFN 层微基准（不含注意力）**
   *routing + grouped GEMM + combine*，expert 用两层 FFN（`H->4H->H`）。

   * 目的：看路由开销相对 GEMM 的占比、不同 `top_k`/CF 的平衡点。
   * **输出**：层时延、GEMM TFLOP/s、路由份额。

3. **端到端“小模型”**
   在一个 **小型 Transformer** 中把 FFN 换成 MoE‑FFN（Top‑1/Top‑2/Soft‑MoE 各 1 个），在一个小语料（如 1–5GB 文本或 TinyStories/WT2）上训练 **若干千步**。

   * **输出**：验证 ppl 曲线、每步 ms、tokens/s、显存、是否有丢弃/塌缩。
   * 这个环节回答“**动态路由有没有伤害/提升质量**”。

> 你问“到底比速度/吞吐/延迟还是 eval 百分数？”
> **都要比**，但分工明确：
>
> * **微基准**（Top‑K、pack/unpack、MoE‑FFN）→ *系统侧*（吞吐/延迟/带宽/占比）。
> * **小模型训练/eval** → *质量侧*（ppl/accuracy/收敛）。
>   这样在报告里能清楚说明：**哪个路由更“快”、哪个更“好”、以及为何**。

---

## 你这批结果的两个小建议

1. **把 micro 的迭代数对齐**（100 vs 200）并重复 3 次取均值/置信区间。
   以你现有数据粗算 95% 置信区间：

   * `torch topk`: ±0.00016 ms；`quack topk`: ±0.00255 ms；
   * 两者均值差 ~0.0075 ms，**在 micro 层面显著**；但在 routing 总时长上不可见，这是正常的。
2. **分解 routing 时长**（推荐用 NVTX / PyTorch profiler）：
   加上 `topk` / `make_indices` / `pack` / `combine` / `sync` 的区段标记，一眼看到瓶颈。
   经验上 **pack/combine** 会吃掉 >70% 的 routing 时间。

---

## 立即可用的一段基准代码（Torch baseline 的 pack/unpack）

> 目的：先把 **pack/unpack 的接口** 跑通与校验，形成 `bench-pack`。
> 说明：这是纯 Torch 的“排序 + 索引”参考实现，便于后续换成 QuACK/CuTe‑DSL 内核做对比。

```python
# 保存为 scripts/bench_pack.py
import argparse, json, time
import torch

def pack_by_expert_torch(x, assign, num_experts):
    # x: [N, H], assign: [N, K] int64 in [0, E)
    N, H = x.shape
    K = assign.shape[1]
    expert = assign.reshape(-1)                      # [N*K]
    x_rep = x.repeat_interleave(K, dim=0)           # [N*K, H]
    order = torch.argsort(expert, stable=True)      # group by expert
    x_packed = x_rep.index_select(0, order)         # [N*K, H]
    counts = torch.bincount(expert, minlength=num_experts)  # [E]
    starts = torch.cumsum(counts, dim=0) - counts            # [E]
    meta = (order, counts, starts, N, K, H, num_experts)
    return x_packed, meta

def unpack_by_expert_torch(y_packed, meta, gate_w=None):
    order, counts, starts, N, K, H, E = meta
    # invert 'order' so we can map back to [N*K, H]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    y_rep = y_packed.index_select(0, inv).reshape(N, K, H)   # [N, K, H]
    if gate_w is not None:
        y_rep = y_rep * gate_w.unsqueeze(-1)                 # 加权写回
    y = y_rep.sum(dim=1).contiguous()                        # [N, H]
    return y

def bench(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    N, H, E, K = args.num_tokens, args.hidden_dim, args.num_experts, args.top_k
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    x = torch.randn(N, H, device=device, dtype=dtype)
    logits = torch.randn(N, E, device=device, dtype=torch.float32)  # router logits
    gate_w, assign = torch.topk(logits, k=K, dim=-1)                 # [N, K] float32, int64
    assign = assign.to(torch.int64)
    gate_w = torch.softmax(gate_w, dim=-1).to(dtype)

    # warmup
    for _ in range(args.warmup):
        y_packed, meta = pack_by_expert_torch(x, assign, E)
        _ = unpack_by_expert_torch(y_packed, meta, gate_w)

    torch.cuda.synchronize()
    t0 = time.time()
    iters = args.iters
    for _ in range(iters):
        y_packed, meta = pack_by_expert_torch(x, assign, E)
        _ = unpack_by_expert_torch(y_packed, meta, gate_w)
    torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / iters

    # 计算等效带宽：2 次搬运（pack + combine）
    bytes_per_token = H * K * 2 * 2  # H * K * 2bytes * 2 passes
    total_bytes = N * bytes_per_token
    gbps = (total_bytes / (avg_ms / 1e3)) / 1e9
    result = {
        "name": "pack+combine_torch_baseline",
        "avg_ms": avg_ms,
        "tokens_per_s": N / (avg_ms / 1e3),
        "eff_bandwidth_GBps": gbps,
        "num_tokens": N,
        "hidden_dim": H,
        "num_experts": E,
        "top_k": K,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=int, default=16384)
    p.add_argument("--hidden-dim", type=int, default=4096)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    bench(args)
```

运行示例：

```bash
python scripts/bench_pack.py --num-tokens 16384 --hidden-dim 4096 --num-experts 128 --top-k 2 --iters 100
```

有了这个 **Torch 基线**，你就可以在相同接口下实现 `pack_by_expert_quack()`（CuTe‑DSL 版本），替换调用并直接得到 apples‑to‑apples 的速度/带宽对比。

---

## 报告里该怎么组织实验结果（建议结构）

1. **Top‑K 微基准**：不同 `K∈{1,2,4}`、`E∈{32,64,128}` 的曲线；`quack` 相对 `torch` 的速度提升。
2. **pack/unpack 微基准**：同样 sweep 参数 + **有效带宽**（GB/s）；展示瓶颈在 pack/combine。
3. **MoE‑FFN 层微基准**：显示“路由份额 vs GEMM”的占比（Stacked bar）。
4. **端到端质量**：小模型在固定训练 tokens 下的 ppl 曲线；不同路由（Top‑1/Top‑2/Soft‑MoE）的质量‑速度帕累托。
5. **负载均衡**：CV/溢出率随训练进程变化，是否需要/用了哪种 aux loss。

---

### 小结

* 你的结果**对**：Top‑K 的实现差异在 micro 层面可见，但在 routing 总时长上 **几乎被 pack/unpack 吞噬**——这正是 MoE 路由优化的核心难点与价值点。
* “动态路由算法的性能”需要 **系统指标 + 质量指标** 双轨评估；报告里建议用 **帕累托前沿** 来展示不同策略的权衡。
* 上面给了一个可直接跑的 **pack/unpack Torch 基线**；把它放进 repo（例如 `scripts/bench_pack.py`），你就能马上得到路由主瓶颈的数字基础；随后替换为 **CuTe‑DSL/QuACK** 版本进行对比，就是你项目的技术亮点。

结论先说在前面：
**结果看起来合理**，我们的 `pack+combine (torch baseline)` 在 `16384×4096、E=128、k=2、fp16` 下测到 **3.37 ms**、按你设定的 *32KB/token* 折算到 **159.5 GB/s** 的有效带宽，这与我们用“排序+索引”的基线实现的期望量级一致（明显低于 HBM 上限，说明还有优化空间）。这一步相当于把**路由阶段的内存瓶颈**跑通了。

> 小注：你当前用的**32KB/token**是假设（便于横向比较）。严格来说，对 *pack+combine*，`k=2、H=4096、fp16` 的**理论字节**约是 *48KB/token*（pack：读 1×H、写 2×H；combine：读 2×H、写 1×H），如果用 48KB 折算，这一条目会是 ~**239 GB/s**。**关键是统一口径**：报告里把口径写清楚，两种口径都给也行。

---

## 你问“比较动态路由算法到底比较什么？”

我们建议把指标分成四层，兼顾“系统性能”和“学习效果”：

1. **核/算子级**（pack、top‑k、combine、all‑to‑all）：

   * 平均/尾延迟（ms）、吞吐（tokens/s）、**有效带宽**（GB/s）、SM 占用、L2/L1 命中（如有 Nsight 数据）。
   * **负载统计**：每个 expert 的 token 数分布、CV（变异系数）、**overflow/drop rate**（当 CF<1 或路由不均衡时）。
2. **层级（MoE‑FFN）**：

   * 总时延拆解：`router(topk+pack) / GEMM / combine` 的**时间占比**；GEMM 的 **TFLOP/s** 与理想值的差距。
3. **模型级（训练/推理吞吐）**：

   * 训练：每步 ms / tokens/s / 显存 / 稳定性（是否塌缩到少数专家）。
   * 推理：prompt/生成吞吐、batch 扩展性。
4. **质量（仅训练实验）**：

   * 验证集 ppl/准确率随 tokens 的曲线；**速度‑质量帕累托**（例如 Top‑1 vs Top‑2 vs Soft‑MoE）。

> 课程要求里“参数高效/动态路由趋势”的讨论，重点就是把**路由策略**如何改变 (1)~(3)，以及是否在 (4) 上带来可测的差异。

---

## Roadmap（可直接当 Milestones 用）

把你刚跑通的 `pack+combine` 当 **M0**，在此基础上做 **M1→M3**。每个里程碑都有**可提交的脚本和图表**。

### ✅ M0（已完成） — Torch 基线的 pack+combine 微基准

* 目标：确认指标口径、把 JSON 输出持久化（`results/*.jsonl`）。
* 产出：当前的 1 组点位 + 代码。
* 立刻行动：做 **参数 sweep**，得到 3×3×3 的网格（`E∈{32,64,128}, k∈{1,2,4}, T×H 至少两档`）。

> 下面给你一个**可直接放进仓库运行的 sweep 脚本**。

**`scripts/sweep_pack.py`**

```python
#!/usr/bin/env python3
import argparse, json, subprocess, sys, os, itertools, datetime, pathlib

def run_once(py, num_tokens, hidden_dim, num_experts, top_k, iters, dtype):
    cmd = [
        sys.executable, "scripts/bench_pack.py",
        "--num-tokens", str(num_tokens),
        "--hidden-dim", str(hidden_dim),
        "--num-experts", str(num_experts),
        "--top-k", str(top_k),
        "--iters", str(iters),
        "--dtype", dtype,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=str, default="8192,16384,32768")
    p.add_argument("--hidden", type=str, default="2048,4096")
    p.add_argument("--experts", type=str, default="32,64,128")
    p.add_argument("--topk", type=str, default="1,2,4")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--out", type=str, default="results/pack.jsonl")
    args = p.parse_args()

    grids = {
        "num_tokens": [int(x) for x in args.tokens.split(",") if x],
        "hidden_dim": [int(x) for x in args.hidden.split(",") if x],
        "num_experts": [int(x) for x in args.experts.split(",") if x],
        "top_k": [int(x) for x in args.topk.split(",") if x],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for nt, hd, ne, k in itertools.product(
            grids["num_tokens"], grids["hidden_dim"], grids["num_experts"], grids["top_k"]
        ):
            res = run_once(sys.executable, nt, hd, ne, k, args.iters, args.dtype)
            res["ts"] = datetime.datetime.now().isoformat()
            print(json.dumps(res))
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
```

**运行示例**

```bash
python scripts/sweep_pack.py --tokens 8192,16384 --hidden 2048,4096 --experts 32,64,128 --topk 1,2,4 --iters 100
```

---

### 🧩 M1 — 拆解路由阶段：单独的 pack 与 combine（Torch vs QuACK）

**目标**：分开计时 *pack* 与 *combine*，并在相同接口下切换 **QuACK/CuTe‑DSL** 版本（正确性优先，再比速度）。

* 代码结构建议：

  ```
  moe_routing_bench/
    routers/
      pack_torch.py
      pack_quack.py           # 先返回 NotImplemented，再逐步填
      combine_torch.py
      combine_quack.py        # 同上
    metrics/
      routing_stats.py        # CV/overflow/entropy 统计
  scripts/
    bench_pack_only.py        # 只测 pack
    bench_combine_only.py     # 只测 combine
  ```
* 指标口径（报告里固定写清楚）：

  * **有效带宽**：`BW = (bytes_moved / time)`；建议**同时**给出

    * **BW‑32K**：沿用你现在的 *32KB/token* 口径（便于横向对比）；
    * **BW‑strict**：按 `(k+1)*H*2B` 估算 **pack** 和 **combine**，两者相加得到 `≈ 48KB/token`（对 `k=2, H=4096, fp16`）。
  * **负载统计**（用于分析路由算法的系统效应）：`counts[e]`、`CV = std/mean`、`overflow_rate`、`gating_entropy`。

> 正确性回归：**pack→combine 应为恒等映射**（忽略溢出时），误差容忍 `atol=1e-3`（fp16）。把这个做成 `pytest`。

---

### ⚙️ M2 — MoE‑FFN 层微基准（不含注意力）

**目标**：测 `router + grouped GEMM + combine` 的**时间分摊**，观察不同 `k、E、CF` 下路由占比何时成为瓶颈。

**实现权衡（避免显存爆炸）**

* 为了先跑通**性能侧**，可以**共享专家权重**（每个专家用同一对 `W1/W2`）——这样不会影响我们在乎的“路由开销 vs GEMM”的时间比例；报告里注明“microbench 权重共享，用于系统分析”。
* 后续如需“真专家权重”，把 `H=1024` 或 `E` 降低即可。

**给你一个可跑的基准脚本骨架（Torch baseline）**

> 这个脚本实现了：top‑k→pack（排序重排实现）→两层 FFN（共享权重）→combine（`scatter_add_`，带权重），并输出拆解后的耗时、GEMM FLOPs/TFLOP/s、负载统计与 overflow。

**`scripts/bench_moe_layer.py`**

```python
#!/usr/bin/env python3
import argparse, json, time
import torch
torch.backends.cuda.matmul.allow_tf32 = True

def topk_router(logits, k):
    # logits: [N, E]
    topk = torch.topk(logits, k=k, dim=-1)           # values, indices
    gate_idx = topk.indices                          # [N, k]
    # 归一化权重（softmax on selected experts）
    gate_w = torch.softmax(topk.values, dim=-1)      # [N, k]
    return gate_idx, gate_w

@torch.no_grad()
def pack_by_expert(x, gate_idx):
    # x: [N, H], gate_idx: [N, k] in [0..E)
    N, H = x.shape
    k = gate_idx.shape[1]
    flat_e = gate_idx.reshape(-1)                    # [N*k]
    flat_t = torch.arange(N, device=x.device).repeat_interleave(k)  # [N*k]

    # 排序重排：同 expert 的 token 紧邻
    order = torch.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]
    sorted_t = flat_t[order]

    # 统计每个 expert 的 load
    E = int(torch.max(flat_e).item()) + 1
    counts = torch.bincount(flat_e, minlength=E)     # [E]
    offsets = torch.zeros(E+1, device=x.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)

    packed = x[sorted_t]                             # [N*k, H]
    return packed, sorted_e, sorted_t, offsets, counts

@torch.no_grad()
def combine_to_tokens(y_packed, sorted_e, sorted_t, offsets, gate_idx, gate_w, H):
    # y_packed: [N*k, H]（与 pack 的排序一致）
    N = gate_idx.shape[0]; k = gate_idx.shape[1]
    out = torch.zeros((N, H), device=y_packed.device, dtype=y_packed.dtype)

    # 恢复：对每个条目，按 token 累加加权和
    # 这里直接用 sorted_t 做 scatter_add
    # 权重要与排序后的一一对应
    flat_w = gate_w.reshape(-1)
    order = torch.argsort(sorted_e, stable=True)  # same 'order' as pack
    w_sorted = flat_w[order]

    out.scatter_add_(0, sorted_t.unsqueeze(-1).expand_as(y_packed), y_packed * w_sorted.unsqueeze(-1))
    return out

def bench_once(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype=="float16" else torch.bfloat16

    N, H, E, k = args.num_tokens, args.hidden_dim, args.num_experts, args.top_k
    x = torch.randn(N, H, device=device, dtype=dtype)

    # 生成路由 logits（可换成 Zipf 分布做非均匀负载）
    logits = torch.randn(N, E, device=device, dtype=torch.float32)

    # 预热
    for _ in range(5):
        _ = x.norm(); torch.cuda.synchronize()

    # 计时：router
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    gate_idx, gate_w = topk_router(logits, k)
    t1.record(); torch.cuda.synchronize()
    router_ms = t0.elapsed_time(t1)

    # 计时：pack
    t2 = torch.cuda.Event(True); t3 = torch.cuda.Event(True)
    t2.record()
    packed, sorted_e, sorted_t, offsets, counts = pack_by_expert(x, gate_idx)
    t3.record(); torch.cuda.synchronize()
    pack_ms = t2.elapsed_time(t3)

    # 计时：FFN（共享权重）
    exp = args.expand
    W1 = torch.randn(H, exp*H, device=device, dtype=dtype) / (H**0.5)
    W2 = torch.randn(exp*H, H, device=device, dtype=dtype) / ((exp*H)**0.5)

    t4 = torch.cuda.Event(True); t5 = torch.cuda.Event(True)
    t4.record()
    y = torch.nn.functional.gelu(packed @ W1)
    y = y @ W2
    t5.record(); torch.cuda.synchronize()
    ffn_ms = t4.elapsed_time(t5)

    # 计时：combine
    t6 = torch.cuda.Event(True); t7 = torch.cuda.Event(True)
    t6.record()
    out = combine_to_tokens(y, sorted_e, sorted_t, offsets, gate_idx, gate_w, H)
    t7.record(); torch.cuda.synchronize()
    combine_ms = t6.elapsed_time(t7)

    total_ms = router_ms + pack_ms + ffn_ms + combine_ms

    # FLOPs（共享权重近似）：两层 Linear，m = N*k
    m = N * k
    flops = 2*m*H*(exp*H) + 2*m*(exp*H)*H   # = 4*m*exp*H^2 * 2 = 8*m*exp*H^2
    tflops = flops / (total_ms/1e3) / 1e12

    # 负载统计
    counts_cpu = counts.to(torch.float32).cpu()
    mean_load = counts_cpu.mean().item()
    std_load = counts_cpu.std(unbiased=False).item()
    cv = std_load / (mean_load + 1e-9)

    # 有效带宽估计（两种口径）
    bytes_per_token_32k = 32*1024
    bytes_per_token_strict = 48*1024  # 近似：k=2、H=4096、fp16 时，pack+combine 总读写
    bw32 = (N*bytes_per_token_32k) / ( (pack_ms+combine_ms)/1e3 ) / 1e9
    bwst = (N*bytes_per_token_strict) / ( (pack_ms+combine_ms)/1e3 ) / 1e9

    return {
        "name": "moe_ffn_layer_microbench",
        "avg_ms_total": total_ms,
        "router_ms": router_ms,
        "pack_ms": pack_ms,
        "ffn_ms": ffn_ms,
        "combine_ms": combine_ms,
        "tflops_total": tflops,
        "routing_share": (router_ms+pack_ms+combine_ms)/total_ms,
        "bw_GBps_32k": bw32,
        "bw_GBps_strict": bwst,
        "num_tokens": N, "hidden_dim": H, "num_experts": E, "top_k": k,
        "expand": exp, "dtype": args.dtype, "device": "cuda",
        "load_mean": mean_load, "load_cv": cv,
        "iters": 1, "seed": args.seed
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=16384)
    ap.add_argument("--hidden-dim", type=int, default=1024)   # 先用小 H 跑通
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expand", type=int, default=4)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    res = bench_once(args)
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

**运行示例**

```bash
python scripts/bench_moe_layer.py --num-tokens 8192 --hidden-dim 1024 --num-experts 64 --top-k 2 --expand 4
```

> 备注：`hidden-dim=4096, E=128` 时若做“真专家权重”，显存会非常大（单层就可能 >30GB）。建议先用共享权重或减小 H/E 跑通“时间分摊”，再逐步加大规模。

---

### 📈 M3 — 端到端（小模型训练 & 推理）

* **训练（质量‑速度帕累托）**：

  * 在一个小 Transformer（例如 6 层、`d_model=512`、`seq=512`）上把 FFN 替换为 MoE‑FFN，做 3 个变体：**Top‑1 / Top‑2 / Soft‑MoE**。
  * 数据：可用一个小语料（TinyStories/WikiText‑2 等）跑 **几千步**。
  * 输出：验证 ppl 曲线、每步 ms、tokens/s、显存、`CV/overflow` 随训练步的变化曲线。
* **推理**：固定检查点下，测 batch 扩展下的 tokens/s，并汇报单 step 延迟与峰值吞吐。

---

## 报告组织（建议目录 & 图表）

1. **背景 & 相关工作**（路由算法演化：Top‑k/Top‑1、Soft‑MoE、Learning‑to‑Specialize、PERFT、Chain‑of‑Experts 等；以及系统侧 MegaBlocks/Tutel 思路的要点）。
2. **方法学**：把路由阶段分解为 `topk→pack→(grouped)GEMM→combine`，用布局视角解释为什么是**访存瓶颈**；说明 CuTe‑DSL/QuACK 的设计点。
3. **实验设置**：硬件、软件、口径（**有效带宽口径写清楚**）、网格参数、负载分布。
4. **结果（系统侧）**：

   * 图 A：`Top‑K` 微基准（Torch vs QuACK）速度曲线；
   * 图 B：`pack/combine` 有效带宽 vs (`E`,`k`,`T×H`)，以及相对加速比；
   * 图 C：`MoE‑FFN` 时间分摊堆叠图（router vs GEMM）；
   * 表 D：`CV/overflow` 与性能的相关性。
5. **结果（质量侧）**：

   * 验证 ppl 曲线（Top‑1/Top‑2/Soft‑MoE）；
   * 帕累托图（质量 vs tokens/s）。
6. **讨论**：

   * 动态路由对系统与质量的权衡；
   * 参数高效微调（PERFT）与路由的结合；
   * 未来工作：多卡 All‑to‑All、block‑sparse 路径等。

---

## 你现在可以立刻做的 5 件事

1. **跑 sweep**：执行上面的 `scripts/sweep_pack.py`，把结果写到 `results/pack.jsonl`。
2. **拆 pack/combine**：复制你已有的 `scripts/bench_pack.py`，拆成 `bench_pack_only.py / bench_combine_only.py`，分别计时，并把**BW‑32K / BW‑strict**都打印。
3. **添加统计**：在基准里记录 `counts[e]、CV、overflow_rate`（如暂未实现容量丢弃，可先记 CV）。
4. **M2 脚本落地**：把 `scripts/bench_moe_layer.py` 放进仓库，用共享权重版本先出一张**时间分摊**图（哪怕先 `H=1024,E=32`）。
5. **README 更新**：补充“如何复现实验”的命令与**口径定义**（有效带宽、FLOPs 的计算方式、随机种子）。

---

## 常见坑位与建议

* **口径统一**：报告里统一“是否计 pack+combine 两者的读写”，否则带宽数字不可比。
* **随机负载 vs 真实负载**：微基准默认负载近似均匀；可以额外加一个 **Zipf 分布**路由来模拟偏斜（更接近真实训练早期）。
* **正确性回归**：`combine(pack(x))≈x`（忽略溢出），`max_abs_err < 1e-3`。
* **显存爆炸**：真专家权重时注意 `E*H*4H` 的量级，先用共享权重做系统侧，再做小尺度真权重验证趋势。
* **计时**：用 CUDA events 包括 `synchronize()`，避免 host 端计时误差。
* **结果固化**：所有 JSONL 加上 `ts` 与 `git SHA`（可用 `subprocess.check_output(['git','rev-parse','HEAD'])` 写入）。

---

如果你愿意，我可以在下一步把 **pack/combine 的 Torch vs QuACK 接口**（`moe_routing_bench/routers/{pack,combine}_{torch,quack}.py`）的骨架也写出来，并给出 `pytest` 的正确性用例。这样你就能只填内核、立即对齐到同一个基准框架。
