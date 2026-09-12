# SoyDNGPNext（复现）

华中农业大学开源的深度基因组预测工具包的复现工作区。该工具包用模块化卷积网络把 SNP 基因型矩阵直接映射为数量或质量表型，支持自定义网络结构、ONNX 跨平台推理、动静态显存调度，参考 *Briefings in Bioinformatics* 的 SoyDNGP 算法。

本工作区以上游快照加复现分支的方式复现，不从零重写 SoyDNGPNext：

- `upstream/` 是独立 git 仓库，origin 指向上游 GitHub（https://github.com/IndigoFloyd/SoyDNGPNext）。
- `upstream-legacy` 标签：冻结的上游提交 ebda01d，未改动的上游基线。
- `repro` 分支：兼容性补丁、测试与复现流程。兼容性补丁记录在 `upstream/COMPATIBILITY_PATCHES.md`；算法与训练协议的修正在 G4 的修正版训练器中，用 T 编号。
- 外层 `data/10_test_examples.vcf` 是作者发布的官方全 SNP 推断夹具（10 样本 × 42195 SNP）。

下文先给设计规格（基因型编码、显存调度、组网、训练、推断），再给运行环境与复现进度。上游包的模块清单见 `upstream/README.md`。

## 设计规格

### 基因型编码

不是标准独热编码。每个位点先映射到一个内部编码，再展开为三通道激活：

| 原始标记 | 内部编码 | ch1 | ch2 | ch3 |
|---|---|---|---|---|
| 1/1 或 1\|1（纯合变异） | 1 | 1 | 1 | 0 |
| 0/1 或 0\|1（杂合变异） | 2 | 1 | 0 | 1 |
| 0/0、0\|0 或缺失 | 3 | 0 | 1 | 1 |

编码张量默认重组为 `(3, 206, 206)` 空间特征图，对应 42436 个变异位点（206×206），送入 2D CNN。

### 显存调度

- 推断用 `reader.py`（cuDF + CuPy），显存加载快，默认丢弃行索引。
- 训练用 `reader_cpu.py`（Pandas + NumPy，`reset=False`），避免 VCF 预处理和反向传播在 GPU 端争抢显存。

### 动态组网

`remodel.py` 读 `data/model.yaml` 逐行实例化积木，不改 PyTorch 代码即可拼网络。积木：`CNN_Block`（卷积 + BN + Dropout）、`Rediual_Block`（残差，stride=1 串联两层卷积）、`SqueezeExcite` / `CBAM` / `CoordAtt`（通道与空间注意力）、`Linear_`（回归取 1，分类取类别数）。

### 双任务训练

- 数量性状 `train_n`：MSE + Adam，按验证集 Pearson r 选点，只保存 r 最高的权重。
- 质量性状 `train_p`：多分类，记录 acc / recall / precision / F1 与混淆矩阵，支持早停。
- 输出由 `utils.outpath()` 自动归档到 `runs/train/train{N}`。

### 推断与部署

`forward.py` 用 ONNX Runtime 做动态 batch 推断。本地缺权重时 `utils.downloads()` 从华中农大服务器拉取；数量性状按 `n_trait.yaml` 的 min/max 逆归一化，质量性状按 `p_trait.yaml` 映射回原始级别。RAPIDS（cuDF / CuPy）对宿主驱动与 CUDA 版本依赖严格。上游文档建议的 Docker 镜像 `indigofloyd/soydngp:general` 已不可拉取（见「运行环境」），本机改用本地 Conda 环境 `soydngp312`。

## 运行环境

部署从 Docker 改为本地 Conda。Docker 镜像 `indigofloyd/soydngp:general` 拉取失败（作者给的是私有仓库地址，不是 Docker Hub 公开仓库；且镜像内置的旧版 CUDA 驱动与 RTX 5090 D 的 CUDA 13 栈不匹配）。项目本身提供标准 PyPI 包 `soydngpnext`，核心逻辑可在本地 Conda 环境中用匹配新显卡的 PyTorch 与 CUDA 运行，省去容器层的驱动适配。

当前可用环境为 `soydngp312`（Python 3.12）：

| 组件 | 版本 / 构建 | 说明 |
|---|---|---|
| Python | 3.12 | 突破 py3.10 的包版本上限，支持 ORT 1.30+ 与新版 RAPIDS |
| PyTorch | 2.14.0+cu130（TUNA 源） | 原生 CUDA 13 |
| ONNX | onnx 1.22 + onnxruntime-gpu 1.30 | 支持更高 IR；导出 opset 需钉在 26 以内 |
| 生信加速 | cupy-cuda13x + cudf-cu13 | 满足 GPU VCF Reader |

要点：

- cuDNN 符号隔离。ORT 1.30 的 CUDA EP 假定 cu12 风格的 cudnn 9 布局（`libcudnn.so` / `libcudnn_ops.so` / `libcudnn_graph.so` 无版本号名 + `cudnnCreate` v8 符号）；cu13 的 `nvidia-cudnn-cu13` 9.24 改了模块拆分与符号命名（引入 `_cnn` / `_adv`），ORT 的 dlopen 探测会失败。两条执行流各用各的库路径：PyTorch 读自身 ELF 的 `$RPATH`，加载 site-packages 内 nvidia 包自带的 cu13 cudnn 9.24；ORT 的 CUDA EP 走 `$LD_LIBRARY_PATH`，命中 `~/opt/ort-cudnn/libs` 里 cu12 布局的 cudnn（从 TUNA 的 `nvidia-cudnn-cu12==9.26.0.51` 解压，补齐无版本号 soname 软链）。分流由 conda 激活钩子完成：`~/miniconda3/envs/soydngp312/etc/conda/activate.d/env_vars.sh` 在激活 `soydngp312` 时把 `~/opt/ort-cudnn/libs` 前置进 `LD_LIBRARY_PATH`，作用域限于该环境激活期间的 shell，未写入 `~/.bashrc` 或系统配置；未配 `deactivate.d`，反激活后该 shell 的前置项仍在，无副作用但不清理。
- 已验证：CUDA EP 正常激活不回退 CPU；卷积、矩阵乘在 GPU 上正确执行；`import soydngpnext`、`cudf` 及 GPU Reader 模块加载无 undefined symbol；RTX 5090 D 32GB 显存正常分配。
- 重建 `requirements.txt` 的包名注意。若沿用通用包名（`cudf`、`cupy`、`onnxruntime`），在本机这套栈上不能直接安装：`cudf` 元包无 cp312 wheel 会构建失败，须装 `cudf-cu13`；`onnxruntime` 应换成 `onnxruntime-gpu`；`cupy` 须装 `cupy-cuda13x`。

## 运行

代码与数据就绪后（环境见「运行环境」）：

```bash
conda activate soydngp312
python train.py        # 训练（需自备数据）
python forward.py      # 推断
```

导出 ONNX 模型时须显式钉住已发布的 opset（如 `torch.onnx.export(..., opset_version=20)`）：ORT 1.30 只保证支持到 opset 26，onnx 1.22 默认的 opset 27 是开发版，会被拒载。

## 训练前数值核对

reader / reader_cpu 来自上游包，已通过 G1 / G2 契约检查（见 `upstream/results/package_baseline/`）。下面两项是启动长时间训练前的数值核对参考，用于论文数据训练前的精度确认。

- VCF GPU Reader 与三通道编码。耗时环节是把 VCF 变异矩阵转成三通道 2D 特征图。位点契约：reshape 为 `(3, 206, 206)`，即 42436 个位点。`reader.load_vcf(path)` 返回 `(genotype_df, variant_ids)`，`encode(genotype_df, variant_ids)` 得到 `(N, 3, 206, 206)` CuPy 张量。`encode` 是查表加 reshape，可先用合成 cuDF 基因型矩阵单独核对，不依赖 `load_vcf`。32GB 显存下测 batch 64 / 128 的占用，确认无残留泄漏。
- 预训练权重前向与精度核对。用官方预训练 ONNX 权重对基准样本前向，核对 ORT 在 Blackwell 上的输出与 CPU 浮点一致。各跑一个数量性状（protein / oil）与一个质量性状（maturity_group）。数量性状按 `n_trait.yaml` 逆归一化 `min + y*(max-min)`；protein 的 [min, max] = [31.7, 57.9]，预测应落在此区间，多数落在典型 35~55。质量性状按 `p_trait.yaml` 把分类序号映射回级别（如 maturity_group，MG 共 11 个级别）。权重需 `utils.downloads` 拉取（先填 `PRETRAIN_SERVER`，或自行导出 opset 26 以内的 ONNX），并把 `forward.py` 的 provider 换成 CUDA 后与 CPU 结果对照。

## 复现进度

- G0 完成：upstream 冻结于 ebda01d（`upstream-legacy` 标签）。
- G1 完成：数据契约冒烟（`upstream/results/package_baseline/g1_smoke/`）。
- G2 关闭：CPU/GPU parity 在对抗夹具与官方全 SNP 推断夹具上均通过（`upstream/results/package_baseline/g2_full_parity/parity.log`）。
- F1 接受：cuDF 兼容性补丁，无算法改动（`upstream/COMPATIBILITY_PATCHES.md`）。
- G3 完成：模型契约（package yaml / 论文 CA 模型 / 2025 残差变体）与论文架构冻结（`upstream/results/package_baseline/g3_model_contract/`）。
- G4 完成：修正版训练器在 `upstream/src/soydngp_repro/`（T1–T4 契约，upstream `Train` 未动）；10 个契约门全 PASS（`upstream/results/trainer_validation/g4/contract.json`）。
- G5 完成：合成微型过拟合，两门全 PASS（回归 PCC 0.9895 / loss 降幅 0.9985；分类 acc 1.0 / macro-F1 1.0，`upstream/results/trainer_validation/g5/`）。
- D0-Raw 关闭：SoyBase SoySNP50K gnm2 全基因型 VCF 已入库（`data/raw/soybase_snp50k_gnm2/`，20,087 样本 × 41,726 位点，MD5 与官方 CHECKSUM 一致）；41,726 为原始数据事实。
- D1 完成：有序 32,032 论文 SNP 契约（32,033 源行 − 1 pandas 表头）；按 (chrom, gnm2 pos) 连到 41,726，命中 31,673 / 缺失 359（→ `./.` 码 3）（`upstream/results/data_contract/d1/`）。
- D2 完成：样本 × 表型对齐，精确 16,960 + 零填充 1,441（用户接受）→ 队列 18,401，非缺失 15,899；表型缺失为行级，23 性状共享同一队列（`upstream/results/data_contract/d2/`）。
- D3 完成：论文输入矩阵 15,899 × 32,032（`data/derived/d3_matrix/`），one-hot 206×206 wrap parity PASS（`upstream/results/data_contract/d3/`）。
- G6 完成：真实数据冒烟（1,024 子集 × 50 ep，非 150ep），回归 / 分类两门全 PASS（`upstream/results/trainer_validation/g6/`）。
- G7 进行中（2026-09-12）：论文忠实基线 2 性状试点（protein 回归 + maturity_group 分类，10-fold × 150 epochs，PAPER_REPRO，全 D3 矩阵 15,899）；setsid 脱离 + 逐折 checkpoint（`upstream/results/trainer_validation/g7/g7_progress.jsonl`）+ 看门狗自动重启。逐 epoch 训练曲线可视化：matplotlib 版 `upstream/scripts/plot_g7_curves.py` + rfig 版（双栏 183mm、中文，`upstream/results/trainer_validation/g7/g7_curves_rfig.png`）。
- 下一步：G8 修正 / 公平基线；作者 .pt 权重导出 ONNX（opset 26 以内）可并行。
