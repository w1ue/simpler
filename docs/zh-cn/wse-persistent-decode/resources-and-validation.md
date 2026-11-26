# 常驻 Decode 的内存、通路与验证

日期：2026-11-25。[完整请求组件推演](./request-service.md) · [仿真入口](../../../examples/wse/persistent_decode/request_service/README.md)。

## 4. 各级内存里放什么、谁负责

| 位置 | 本次内容和预算 | 可读写方 / 数据通路 | 生命周期与回收条件 |
| ---- | -------------- | ------------------- | ------------------ |
| Host 内存，作为 Host HBM 角色 | 请求、稳定 prefill 快照、grant/cancel/credit 源、D2H 控制和输出区；固定区合计 4480B | Host 写 H2D 源、读 D2H 结果；DMA 读写；Core 无直接解引用 | 固定区跨请求存在；DMA 持源期间不能改写。serving 自有副本另外计，不在此固定区预算中 |
| Bailu | 两芯片各 3 group×8 页×128B=3072B；页中为 history/context/key/group 标量包 | DirectDMA 导入；batcher 预取和写回；本路径不使用 Core 直访 | 物理池覆盖会话，请求租约按页增量持有；本请求 DMA 与设备引用排空后归还 |
| home local SRAM | 固定帧、授权表、fetch 区、dirty 两项×3 group、output 两项、两组通信 bank；Core 0/8 分别 7296/7168B | 本 Core 读写；batcher/CoreLink 在授权传输窗口读写；其他 Core 不按普通指针直接读写 | 布局覆盖会话；slot 状态绑定 generation；脏项所有组完成、输出收到 credit、通信最后使用后分别复用 |
| expert local SRAM | 固定输入/输出，8 个路由 Core 各 1024B，共享 Core 5/13 各 512B | 专家 Core 处理，通信单元搬运；home 经传输取回结果 | bank/task/credit 共同约束，不能以“消息已接收”替代最后使用 |
| `batcher.mem` 角色 | 每 endpoint 声明 8192B 控制额度；descriptor/队列/credit/进度在 Python 中有影子 | batcher 推进传输，Host/Core 只通过提交和完成协议改变责任 | 会话级固定预算；本次没有把全部影子字段编码成原生结构，尚未证明真实 SRAM 装得下 |

单个 cell 除 activation/grant/frame 为 512B，其余数据 cell 默认 128B。Python payload 未做原生字节序列化，以上是声明的功能资源预算；不是实际 DeepSeek 权重/KV 容量估算。大张量、attention 历史、权重流量及真实 Host HBM 管理尚未实现。

传输逐条经历 `Accepted → Issued → SourceReadDone → DataVisible`；这些事件不混为一种 done。目的数据只有 publish 后可读，源在读取责任结束前不能改写，业务/任务 credit 在最后使用后归还。

## 5. 硬件通路与预算如何约束推进

| 通路 | 本次使用方式 | 等待和保证 |
| ---- | ------------ | ---------- |
| Host → Bailu | 直接导入，不经过 batcher | 与其他访问共享 H2D ingress/Bailu bank；全部导入可见后才激活 |
| Host/Bailu → Core SRAM | 经 batcher | 外部非确定性域进入 SRAM 必须有搬运、ready 和 credit；不能靠 Core 忙等非确定性直访 |
| 同片 SRAM → SRAM | CorePush，预置 payload 区 | 先取得接收 credit 和未来写端口；注入到可见固定 2 tick，排队时间另计 |
| 跨芯片 SRAM → SRAM | 外部域的 batcher 服务模型 | 声明共享 cross 带宽和完成事件，未实现真实两端 batcher/链路协议细节 |
| Core SRAM → Bailu/Host | batcher 写回 / D2H 输出 | dirty/output 分环，分别保留稳定源和有界积压 |

**0.5 RTT 用在接收端看到 payload 即可推进的关键传递上。**credit 回传服务下一次复用，不能把它解释成 Low→Grant、远端专家计算或任意外部访问都只需半个往返。

仿真声明每个 batcher endpoint 每 tick 仅执行一次 issue 或 publish，共 4 个等待 context，数据最多使用 3 个；控制和数据交替选取。Bailu 两 bank 及 H2D/D2H/cross 域各自按每芯片每 tick 64B 服务，共享而非每请求独享；外部写 SRAM 遇到已预留的 CorePush 接收端口时延后。

后台写回与输出各限制为每 home endpoint 一个在途 job，避免将所有数据 context 堵在尚未就绪的后台流上。**给控制流留 context，还要给当前计算的数据流留资源。**本轮在声明的注入条件下跑通这一策略，没有证明所有拓扑和任意流量下均无死锁。

## 6. 正常场景的实际里程碑

以下由本次 trace 摘取；tick 是逻辑单位，不对应 ns，也不能据此推算 TPOT。完整事件含组件、request、slot、generation、token/layer、transfer、path 和 endpoint，见[事件摘录](../../../examples/wse/persistent_decode/request_service/evidence/milestones.json)。

| tick | 组件 | 事件与含义 |
| ---- | ---- | ---------- |
| 0 | simpler Host / Core | profile 绑定、布局封存、端点配置、启动一次常驻程序 |
| 0 / 1 / 2 | serving / simpler | A、B 被接受；C 因无 slot 排队 |
| 32 / 35 | Core | A/B 的初始页表和授权安装 |
| 40 / 43 | batcher / Core | A/B 的历史状态已在 SRAM，进入 active |
| 60 / 64 | Core | A/B 增量授权安装；B 三组页数变为 3/2/3 |
| 65 | Core / ExpertCore | A 第一个 token 的第 0 层合并为 10143 |
| 1093 / 1144 | Core | A/B 的第一个 token 61/0 本地提交 |
| 2087 / 2094 / 2097 | Core / serving | A 生成 EOS；结束状态到达；最后输出副本到达 |
| 2113 / 2114 | simpler / serving | A 排空回收；C 以 generation 2 接入；CloseInput |
| 2154 / 2190 | Core | C active；B 第二 token 59 提交，早于首轮写回源读取完成 |
| 3159 / 3237 | Core | C 第一 token 39；B 第三 token 47 |
| 4150 / 4179 | Core / simpler | C 长度结束；C 排空回收 |
| 4255 / 4283 | Core / simpler | B 长度结束；B 排空，形成 SessionDrained |
| 4289 | Core / simpler | PersistentKernelExit 后 SessionUnloaded |

正常场景有 4565 次传输；全部都检查四阶段事件顺序、路径、共享服务预算与最终责任归零。8 个 token 的 320 个逐层 hidden 以及最终 token 均与独立整数 oracle 一致。

## 7. 额外场景与已验证的性质

| 场景 | 注入 / 涉及组件 | 通过条件 |
| ---- | --------------- | -------- |
| normal | 三请求、两 slot、跨组页边界、受控延迟写回 | EOS/长度正确，C 复用 A，第二 token 使用 SRAM 最新状态，最终全部回收 |
| slow_output | simpler 暂不接收 B 输出，直到环满 | 第三 token 停在输出环边界，取得 Host 所有权并归还 credit 后恢复，无覆盖 |
| slow_writeback | batcher 延迟 B 写回，直到 dirty 环满 | 第三 token 等待，写回恢复后继续；在途源不被覆盖 |
| slow_client | serving 暂缓客户端发送 | Core 先取得输出 credit，随后客户端才发送 B token，设备 buffer 与网络寿命解耦 |
| cancel_import | serving 在 prefill DMA 在途时取消 A | 不 Activate、不生成 token；已发 DMA 完成后才能归还页和 slot |
| cancel_running | serving 在 B 首 token 后取消 | Cancel 经 batcher 到 Core，在 token 安全边界停；在途输出被消费/丢弃并归还责任 |
| deny_refill | simpler 对 Low 返回拒绝授权 | 不写未授权 context；以 capacity_denied 停止并排空，A/B/C 各生成 1 token |

另有 12 项拒绝/保护检查：不支持的 sampling、生成超限、缺失 group、CloseInput 后提交、launch 后分配、重复请求、DMA 在途归还租约、DMA 源覆盖、旧 generation 输出 credit、替换活跃页表前缀、活跃请求未排空即退出、分组原子预留不足。最后一项必须无部分分配副作用，返回等待而不是破坏已有账本。

本轮发现并消除了两类模拟捷径：Core 不再读取 Host 页账本来“安装”授权，Host 不再窥视 Core 环和状态来判断输出或排空。相关状态必须由已复制到目的内存的 H2D/D2H 包通知；源码里的回调只是模拟完成通知，实际仍需映射成驱动、doorbell、completion 或编译程序可见的事件。

## 8. 从本原型到实际落地还缺什么

| 目标组件 | 下一步需要接实的契约 |
| -------- | -------------------- |
| serving / 模型适配 | 完整请求授权、输出最终通知、取消边界；从实际 StepCommand 迁移时确定唯一 KV 账本与页池所有权，不能两个分配器同时管同一池 |
| PyPTO 编译器 / model lib | 生成固定设备控制流与资源清单；真实 attention、位置、权重、分组压缩尾部、采样及 DSpark 能力 |
| simpler WSE Host | 实际程序部署、地址绑定、完整请求 API、grant 页表 ABI、真实 allocator/租约、错误隔离与进度接口 |
| batcher / CoreLink / AICore | descriptor 编码、端口/credit/队列上限、跨片接收入口、SourceReadDone/DataVisible、取消后的 drain 可观测性 |
| 会话与多芯片控制 | 每参与端点的启动/退出确认、执行域 fence、超时隔离；不是单一 Python 对象的一次 ready/exit |

该实验没有实现跨请求 collective 的 active-mask 协议、抢占迁移、任意中途恢复和多 Host。它可用于检查“完整请求授权 + 固定 Core 推进 + batcher 交接”所需通路和资源责任；不能作为真实 DeepSeek 数值正确、原生内存布局可装入或确定时延硬件已经实现的证明。

本次复现、源码哈希、完整压缩 trace 和结果随[仿真目录](../../../examples/wse/persistent_decode/request_service/README.md)保存。原有 C++ 仿真及其 50/61 检查保持原证据口径。
