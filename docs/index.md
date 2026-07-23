# Agentic RL 训练服务 —— 部署后使用文档

> 适用服务：**Agentic RL（verl + Harbor + ACK）GPU 训练服务**
> 场景：在阿里云 ACK 上用 SWE-bench + SWE-agent 对 Qwen2.5 系列模型做 agentic 强化学习（GRPO）训练。
> 本文只讲**部署成功之后如何使用**；部署参数填写请见服务详情页说明。

---

## 1. 部署成功后你得到了什么

服务实例创建完成后，在 ComputeNest 控制台的**实例详情 → 输出（Outputs）**里能看到以下资源，后续操作都会用到：

| 输出项 | 示例值 | 用途 |
|---|---|---|
| `ClusterId` | `ca50109364...` | ACK 集群 ID，连接集群用 |
| `Namespace` | `agentic-rl` | 训练工作负载所在命名空间 |
| `RayClusterName` | `raycluster-verl` | verl 训练主进程所在的 RayCluster |
| `ModelPvcName` | `verl-models` | 模型持久卷（挂到 head pod `/var/model`）|
| `ImagePullSecret` | `acr-registry` | 拉取/推送镜像的密钥名 |
| `ServiceAccountName` | `rayclustertest` | 已配好 RBAC 的 ServiceAccount |
| `BuildkitAddress` | `tcp://buildkitd.agentic-rl.svc:1234` | 构建 sandbox 镜像的 BuildKit 地址 |
| `ImageRegistryPrefix` | `<ACR实例名>-registry-vpc.<region>.cr.aliyuncs.com/swe-bench` | sandbox 任务镜像仓库前缀 |
| `AcrRegistryEndpoint` | `<ACR实例名>-registry-vpc.<region>.cr.aliyuncs.com` | ACR 企业版 VPC 域名 |
 
集群里已默认装好：**KubeRay Operator**（`kuberay-system`）、**OpenKruise Agents Sandbox 控制器**（`sandbox-system`）、**BuildKit**、以及一个双卡 GPU 的 **verl 训练 head Pod**。开箱即可训练，无需再装组件。

---

## 2. 连接集群

推荐用阿里云控制台，最省事、不依赖公网：

1. 打开 **容器服务 ACK 控制台** → 找到输出里的 `ClusterId` 对应集群。
2. 集群页面右上角点 **通过 CloudShell 连接集群**（自动配好 `kubectl` 与 kubeconfig）。

验证连接：

```bash
kubectl -n agentic-rl get pods
# 期望看到：
# raycluster-verl-head-xxxxx   1/1   Running   ...
# buildkitd-xxxxx              1/1   Running   ...
```

> 如需在**本地机器**用 kubectl，集群默认只有内网 API 端点。可在 ACK 控制台「集群信息 → 集群端点」开启公网访问后下载公网 kubeconfig（仅调试期需要，正式使用建议用 CloudShell）。

> **⚠️ RAM 子用户必读：访问集群需先获得 RBAC 授权。**
> 部署服务的**主账号（集群创建者）默认就是集群管理员**，直接能用 CloudShell / kubectl。但**其他 RAM 子用户**即使能登录控制台，默认也**没有任何集群 RBAC 权限**，执行任何 `kubectl` 会报：
> ```
> Error from server (Forbidden): pods is forbidden: User "<uid>" cannot list resource "pods" ...
> ```
> 这是“认证通过但未授权”，不是集群故障。**解决**：由主账号在 **ACK 控制台 → 集群 → 授权管理（RBAC）**中，给该子用户授予本集群（或 `agentic-rl` 命名空间）的**管理员 / 运维**角色后，kubectl 才能正常使用。

---

## 3. 进入训练 Pod

所有训练操作都在 verl head Pod 内进行。先取 pod 名，再进入：

```bash
# 取 head pod 名
HEAD=$(kubectl -n agentic-rl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
echo $HEAD

# 进入交互 shell
kubectl -n agentic-rl exec -it $HEAD -- bash
```

进入后确认 GPU 就绪（应看到 2 张 L20）：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

---

## 4. 预置模型与数据集（首次必做）

> **⚠️ 重要：这是部署后第一步，不做无法训练。**
> 服务只负责把集群、GPU、RayCluster、BuildKit、RBAC 等**基础设施**拉起来。head Pod 上只挂了两个**初始为空**的持久卷：`/var/model`（模型）和 `/var/model-dataset`（数据/checkpoint）；**不预置任何模型和数据集，也不下发训练配置 ConfigMap**。
> 因此每次全新部署后，你必须**自己**：①下载模型到 `/var/model`；②下载数据集到 `/home/verl`；③按第 5 节把训练脚本 `run-2gpu.sh` 写进 head Pod（脚本里的参数就是「训练配置」，无需依赖任何 ConfigMap）。

模型 / 数据卷初始为空，第一次使用需要下载（下载到 PVC，Pod 重启不丢）。**在 head Pod 内执行**：

```bash
# 1) 下载模型到 /var/model（香港区直连 HuggingFace 很快）
# 注：镜像内已确认可用的是 huggingface-cli（新版 huggingface_hub 也可用简写 hf）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /var/model/Qwen2.5-3B-Instruct

# 2) 下载 SWE-bench 数据集到 /home/verl（注意名字带 swe-bench/ 前缀）
harbor dataset download swe-bench/swe-bench-verified -o /home/verl
# 产出：/home/verl/swe-bench-verified/<task>/{task.toml, instruction.md, environment, tests}
```

> 换模型时把 `/var/model/<你的模型目录>` 换掉，并同步修改训练脚本里的 `actor_rollout_ref.model.path`。

---

## 5. 启动训练

### 5.1 冒烟验证（1 个任务，约 5~7 分钟，强烈建议首跑先做）

本服务在部署时已把冒烟脚本 `run-2gpu.sh` **预置在 head Pod 内**（挂载为只读文件 `/opt/agentic/run-2gpu.sh`；其中 sandbox 镜像仓库地址 `REGISTRY` 已按你的 ACR 实例**自动注入**，无需手改）。脚本用**双卡 L20 训练 Qwen2.5-3B**、跑 1 个任务，已验证跑通。

> 前提：已完成第 4 节的模型 / 数据集下载（脚本读取 `/var/model/Qwen2.5-3B-Instruct` 与 `/home/verl/swe-bench-verified`）。

在 head Pod 内直接后台启动，日志写到 `/tmp/train.log`：

```bash
cd /home/verl && nohup bash /opt/agentic/run-2gpu.sh > /tmp/train.log 2>&1 &
```

想先看脚本内容：`cat /opt/agentic/run-2gpu.sh`。

### 5.2 正式训练

冒烟通过后，把预置脚本**拷出来改**（挂载点只读）再跑正式训练：`cp /opt/agentic/run-2gpu.sh /home/verl/run-formal.sh`，编辑后 `nohup bash /home/verl/run-formal.sh > /tmp/train.log 2>&1 &`。主要改这几处：

| 参数 | 冒烟值 | 正式建议 |
|---|---|---|
| `data.harbor_train_limit` / `harbor_val_limit` | `1` | **去掉这两行**（用全量任务）|
| `data.train_batch_size` | `1` | `≥ n_gpus`（如 8、16）|
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `1` | dp_size 的整数倍，且能整除过滤后的 batch |
| `trainer.total_epochs` | `1` | 按需要设 |
| `trainer.save_freq` | `1`（每步存）| 按需要，如 `10` |

> 双卡 L20 下，`param_offload`、`optimizer_offload`、`tensor_model_parallel_size=2` 是 3B 不 OOM 的关键，**同规格不要动**。换 GPU 规格 / 换更大模型见 5.3。

### 5.3 换 GPU 规格 / 换更大模型：改哪几个参数

本服务是**单机多卡**架构：RayCluster 只有一个 head Pod，训练全在这一个 Pod 上跑，Pod 申请 `GpuPerNode` 张卡。默认 `ecs.gn8is-2x.8xlarge` = 2×L20。

**一条铁律**——这三个数必须完全相等，否则 head Pod 调度不出来（Pending）或训练一启动就报错：

```
ECS 规格物理卡数  ==  GpuPerNode（模板参数）  ==  trainer.n_gpus_per_node（脚本）
```

#### 脚本里与规格/模型绑定的变量（改规格/换模型时对照下表改）

预置脚本 `/opt/agentic/run-2gpu.sh` 是 **2×L20 + Qwen2.5-3B** 版本、只读。要改规格或换模型，先 `cp /opt/agentic/run-2gpu.sh /home/verl/run-my.sh` 再改下面这些变量（其余不用动）：

| 脚本变量 | 默认（2×L20/3B）| 何时改 / 改成什么 |
|---|---|---|
| `actor_rollout_ref.model.path` | `/var/model/Qwen2.5-3B-Instruct` | 换模型：改成你下载到 `/var/model/` 的模型目录 |
| `trainer.n_gpus_per_node` | `2` | 换卡数：= 规格物理卡数（必须 = 模板 `GpuPerNode`）|
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | 换卡数：能整除卡数即可，一般直接 = 卡数 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.9` | 显存紧时调低（7B 在 2 卡需降到约 `0.5`）|
| `...fsdp_config.param_offload` / `optimizer_offload` | `true` | 卡多显存宽裕可设 `false` 提速；紧张保持 `true` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `12288` | OOM 时调小 |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | `24576` | OOM 时调小 |
| `data.max_prompt_length` / `max_response_length` | `4096` | OOM 时调小 |
| `trainer.experiment_name` 与 `trainer.default_local_dir` | `qwen2.5-3b` | 换模型时改个名，checkpoint 目录好区分 |
| `data.harbor_train_limit` / `harbor_val_limit` | `1` | 冒烟用 1；正式训练删掉这两行用全量 |

> 模板参数（部署前填）↔ 脚本变量的对应：模板 `GpuPerNode` ↔ 脚本 `trainer.n_gpus_per_node`（必须相等）；模板 `WorkerInstanceType` 决定物理卡数；模板 `RayHeadMemory`/`RayHeadCpu` 是 Pod 资源上限（见下）。

#### 场景 A：换卡数 / 卡型（如 2 卡 → 4 卡）

只改下面这几处，其余不动：

| 改哪里 | 参数 | 改成 |
|---|---|---|
| 模板（部署前填） | `WorkerInstanceType` | 目标 GPU 规格，如 `ecs.gn8is-4x.16xlarge`（4 卡）|
| 模板（部署前填） | `GpuPerNode` | 该规格的**物理卡数**，如 `4` |
| 脚本（head Pod 内） | `trainer.n_gpus_per_node` | 同 `GpuPerNode`，如 `4` |
| 脚本（head Pod 内） | `tensor_model_parallel_size` | 能**整除**卡数即可（一般直接 = 卡数）|

> 卡多、显存宽裕时，可把脚本里 `param_offload` / `optimizer_offload` 改 `false` 提速。checkpoint 分片名里的 `world_size` 会自动跟着卡数变，不用手改。

#### 场景 B：换更大模型（如 Qwen2.5-7B）

1. 先把模型下到 `/var/model/<目录>`，再改脚本 `actor_rollout_ref.model.path`。
2. 模型越大越吃显存。**2×L20 跑 7B 全参显存吃紧**：实测 `gpu_memory_utilization=0.9` 会在 `update_actor` 阶段 CUDA OOM。要在 2 卡上跑，需把它降到约 `0.5` 给训练让出显存，并保持 `param_offload=true`；**更稳妥是直接上 4 卡（4×L20 或 A100-80G）**，同时按场景 A 改规格。
3. 仍 OOM 就继续下调 `ppo_max_token_len_per_gpu`、`data.max_prompt_length` / `max_response_length`。

#### RayHeadMemory / RayHeadCpu 到底怎么填

- **`RayHeadMemory`（内存上限）**：主要给 **CPU offload**（把参数/优化器状态卸载到内存）用。3B **稳态训练**约 48G，但**保存 checkpoint、汇聚全量状态时峰值会高得多**（90Gi 限额实测被 `OOMKilled`），所以默认 `220Gi` 并非虚高。填写原则：**别明显低于默认，且绝不能超过所选规格的实际内存**（超了 head Pod 会一直 `Pending`）。换更大模型时内存占用上升，保持默认或适当调大。
- **`RayHeadCpu`（CPU 上限）**：默认 `16` 够用；**不能超过规格的 vCPU 数**。

#### 换规格前必查两条

- `tensor_model_parallel_size` 必须整除 `n_gpus_per_node`。
- 目标 `ZoneId` 要有该 GPU 规格库存（模板里 `WorkerInstanceType` 会按 `ZoneId` 联动过滤）。

#### 场景 C：多机多卡（当前不支持）

现在只有 `headGroupSpec`（`trainer.nnodes=1`），仅支持单机多卡。要跨多台 GPU 机器，需在模板 RayCluster 里新增 `workerGroupSpec` 并把脚本改 `trainer.nnodes>1`——属结构性改动，不是改数值能搞定。

---

## 6. 监控进度与验证成功

### 实时看日志

```bash
kubectl -n agentic-rl exec $HEAD -- tail -f /tmp/train.log
```

### 训练全流程（首个 step 约 5~7 分钟）

1. vLLM 双卡加载模型 → `Proxy server started at http://<head-ip>:xxxxx`
2. **首次**会用 BuildKit 构建 sandbox 镜像（`FROM swebench/...`）并推到 ACR，随后拉起一个 `<task>-xxxx` 的 sandbox pod（可 `kubectl -n agentic-rl get pods` 看到）
3. SWE-agent 在 sandbox 里多轮解题 → 采集轨迹算 reward → `update_actor` 更新 → 保存 checkpoint

### 验证成功的标志（对照日志）

- Pod 全程 `RESTARTS 0`
- 有 validation 指标：`val-core/harbor/reward/mean@1`（小样本下为 `0.0` 属正常）、`val-aux/num_turns/mean`（agent 真实多轮交互，>0）
- 优化器 step 越过：`timing_s/update_actor` 有值、`actor/pg_loss`、`actor/grad_norm` 有值
- checkpoint 落盘（见下节）
- 结束打印 `Training Progress: 100%`

### 实测基准（双卡 L20 / Qwen2.5-3B / 1 个任务冒烟）

已在双卡 L20（ecs.gn8is-2x.8xlarge）集群上实际跑通 1 step GRPO，供对照预期：

| 指标 | 实测值 |
|---|---|
| 单 step 耗时 | `timing_s/step` ≈ 171s（首次含 sandbox 镜像构建，全程约 5 分钟）|
| `update_actor` | ≈ 10.8s（优化器 step 确实执行）|
| `save_checkpoint` | ≈ 14.5s |
| 显存峰值 | `actor/perf/max_memory_allocated_gb` ≈ 28.8G（单卡，未 OOM）|
| CPU offload 内存 | `cpu_memory_used_gb` ≈ 47G |
| `num_turns` | 验证 4 / 训练 5~9（agent 真实多轮交互）|
| `val-core/harbor/reward/mean@1` | `0.0`（3B 小样本正常，advantage/grad_norm 也为 0）|
| 吞吐 | `perf/throughput` ≈ 32 tok/s |

> reward=0 不代表链路有问题：3B 模型小样本下解不出 SWE-bench 题属正常现象，优化器/检查点链路仍完整跑通。想看到非零 reward 需换更强模型或扩大样本。

---

## 7. 训练产物：checkpoint 与导出

checkpoint 保存在持久卷 `/var/model-dataset/checkpoint/qwen2.5-3b/`：

```bash
kubectl -n agentic-rl exec $HEAD -- ls -R /var/model-dataset/checkpoint/qwen2.5-3b/global_step_1
# actor/model_world_size_2_rank_0.pt / rank_1.pt   ← 分片权重（双卡各一份）
# actor/optim_world_size_2_rank_*.pt               ← 优化器状态
# actor/extra_state_world_size_2_rank_*.pt
# actor/huggingface/                               ← HF 配置
# actor/fsdp_config.json
# ../latest_checkpointed_iteration.txt             ← 最新已存步数
```

**合并分片导出为标准 HuggingFace 模型**（verl 自带工具）：

```bash
kubectl -n agentic-rl exec -it $HEAD -- bash -c '
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /var/model-dataset/checkpoint/qwen2.5-3b/global_step_1/actor \
  --target_dir /var/model-dataset/checkpoint/qwen2.5-3b/global_step_1/merged_hf
'
```

导出后 `merged_hf/` 即可用 `transformers` / vLLM 直接加载。要下载到本地可用 `kubectl cp`。

---

## 8. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| sandbox pod 拉镜像报 `not found` 秒失败 | 缺 `skip_image_check: false`，跳过了镜像构建 | 确认脚本 `environment_kwargs` 含 `skip_image_check: false` |
| 日志报 `secrets "acr-registry" is forbidden` | SA 无读 secret 权限 | 本服务模板已固化该 RBAC；若自建命名空间需给 SA 补 `secrets: [get,list]` |
| SWE-agent 反复 `Connection error` 且 base_url 是 `0.0.0.0` | `llm_proxy_ip` 没注入真实 IP | 用脚本里的 `PROXY_IP=$(hostname -i ...)` 动态注入，勿用 export |
| `update_actor` 报 `x % y != 0` | 小 batch 下 `ppo_mini_batch_size` 不整除 | 冒烟用 `1`；正式设为 dp_size 整数倍 |
| head Pod `OOMKilled` | 内存不足 | 保持 head 内存 limit ≥ 220Gi 与 CPU offload 配置 |
| `reward/mean@1: 0.0` | 小样本 + 3B 模型没解对 | 正常，不代表链路有问题；扩大样本/换更强模型再看 |
| `kubectl` 报 `Forbidden: User "..." cannot list resource` | RAM 子用户无集群 RBAC（认证通过但未授权）| 由主账号在 ACK 控制台「授权管理」给该子用户授予集群 RBAC 角色（详见第 2 节）|

> 首次拉取巨型 verl 镜像约 15~20 分钟属正常；`Model openai/qwen-max does not support function calling`、FSDP/torch 版本告警均可忽略。

---