# SoyDNGPNext Reproduction

## 复现目标
复现 SoyDNGP / SoyDNGPNext 的基因型 → 表型预测流程，包括公开数据重建、SNP 编码、模型训练与交叉验证。

## 数据复建
- SoySNP50K gnm2（Song & Hyten 2015）：20,087 样本 × 41,726 位点，官方 checksum 校验通过。
- 论文 SNP 列表：32,032 个有效 SNP；按 (染色体, 坐标) 匹配原始数据，命中 31,673，缺失 359（缺失按作者默认逻辑处理）。
- 训练队列：15,899 样本（reconstructed cohort，见「当前结论与限制」）。
- 模型输入：3 通道 × 206 × 206 基因型特征图。

## 当前结果
| 性状 | 任务 | 协议 | 指标 |
|---|---|---|---|
| Protein | 回归 | 10-fold × 150 epochs | PCC 0.6701 ± 0.0153 |
| Maturity group | 分类 | 10-fold × 150 epochs | Macro-F1 0.5544 ± 0.0173 |

## 当前结论与限制
数据、编码与训练流程已复建并可稳定运行，上表是两个性状的初步 baseline。当前结果不是严格的论文数值复现，原因有三：

- 网络：当前结果用 SoyDNGPNext 官方 package 的 `model.yaml`（经 `remodel` 构建）。论文原始 SoyDNGP 模型含两个 CA block（首卷阶段后 32ch@206×206、flatten 前 1024ch@7×7），已追溯到 `upstream/src/paper_model/soydngp_ca.py`，但尚未在同协议下完成正式训练。
- 损失：当前回归用 MSE；官方训练器默认 SmoothL1Loss。
- 队列：15,899 为 reconstructed cohort。其中 1,441 个 leading-zero accession 映射是按规则人工接受的，不能声称与作者原始训练队列完全一致（该决定已在仓库记录）。

综上，当前结果定位为 preliminary baseline，而非论文数值的严格复现。

## Repository
- `upstream-legacy`（tag）：冻结的官方代码基线。
- `repro`（默认分支）：复现代码、数据契约与实验结果。
- `upstream/REPRODUCTION.md`：详细审计记录。
