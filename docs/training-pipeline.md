# 小数据训练与验收

## 图片准备

第一批按可获得的数量准备，20–30 张也可以开始一次针对性迭代，例如 15–20 张易混角色负样本、5–10 张对应的大肥鱼正样本。准确标签和不同来源比数量更重要，不必凑数量。这不是保证最终误报率的样本量。

优先补 X.zip 中仍误报的角色和同画风对照，保留完整系列。不要把压缩、改尺寸、截图或同一动图的多个帧算作独立作品。按系列建目录，写明角色；不确定的放“待确认”。无需手工划分 train/test。

人工审核后导入 `data/curated/meta.jsonl`，每行包含 `path`、`label`、`category`、`subcategory`、`group_id`、`source`。`subcategory` 尽量表示具体负角色，不要全部叫“其它”。可补 `template_id`、`series_id`、`author_id`；已填写的同一 ID 将整组划分。合影只要包含大肥鱼就为正，不能因为文件名是其它角色而标负。

X.zip 已用于诊断，保留为已知回归集。未知验收应另外保留完整作品/模板组；小样本报告误报的实际张数和置信界，不能宣称普遍“零误报”。

## 持久划分

```bash
uv run python scripts/build_dataset.py
```

首次从旧格式迁移时保留原 val/test 的归属，只从旧 train 预留独立 calibration。新建数据集目标比例是 train/val/calibration/test = 70/10/10/10，按组分配，数量很少的角色可能无法出现在全部集合。

`split_state.json` 保存历史身份与分配，`build_report.json` 记录本次结果。新增作者/模板信息若连接了两个已有集合，或相同像素有矛盾标签，构建失败并保留原 CSV。需要人工审核冲突；不能通过删除 split state 重洗测试集。近似图可能是不同角色，保留其真实标签并整组划分，不能自动让正标签胜出。

真实鱼、鲸、鲨鱼和海豚等归入 `negative_unrelated`，训练集默认按物种均衡保留约 60 张，原始文件仍保留。只缩减纯水生动物来源组；与其它样本相连的混合组完整保留，所以最终数量可能略多。旧 val/calibration/test 不参与缩减。插画鱼／鲸继续保留为 `negative_fish`。可用 `--real-aquatic-train-limit` 配置数量，`-1` 用于保留全部的对照实验，排除清单见 `build_report.json`。

## 训练、校准、部署

```bash
uv run python -m is_dafeiyu.train --config configs/mobilenetv3_large.yaml
uv run python -m is_dafeiyu.evaluate --config configs/mobilenetv3_large.yaml
uv run python -m is_dafeiyu.export \
  --checkpoint artifacts/mobilenetv3_large_100_hard_v2/best.pt \
  --browser --quantize fp16
uv run python scripts/eval_onnx.py \
  --onnx artifacts/mobilenetv3_large_100_hard_v2/model_browser_fp16.onnx
```

训练默认正类权重为 1，按来源组采样；可配置 `training.subcategory_loss_weights` 对审核过的困难负类加权，但需要在固定 val 上评估收益。增强保持全图；冻结的骨干子树保持 eval 状态。每阶段通过 `training_audit.json` 验证冻结参数和缓冲区没有变化。

训练使用 BF16，选模与 PyTorch 校准使用 FP32。配置明确指定 `target_precision`、`min_recall`、`max_fpr`、`max_subcategory_fpr`。先满足所有约束，再优先召回率。只有符合条件的 checkpoint 才命名为 `best.pt`；没有达标模型时保存 `candidate.pt` 和历史并以非零退出。不满足条件的校准保存 `calibration_attempt.json`，不会静默回退 F-beta。

蒸馏配置使用教师 checkpoint，对相同增强图像在线计算教师输出，并转换教师所需归一化。教师与真实标签冲突的样本仅使用真实标签损失。教师须有匹配的数据划分记录；旧教师不能把已经见过的图片重新当作未知留出集。

导出先写候选文件，核对完整 calibration 集的单图/批量推理差异，再按最终 ONNX 输出选择部署阈值。通过后才替换目标文件，并写入 `model_browser_fp16.calibration.json`。浏览器集成应读取该文件的 `threshold`；温度已烘焙进模型，不能再次缩放。部署时模型和 sidecar 应一起发布，当前静态网页的已发布模型目录不会自动更新。

`calibration.json` 绑定 PyTorch checkpoint，`<model>.calibration.json` 绑定最终 ONNX SHA256，不能混用。新训练配置写入 `_v2` 目录，已有训练输出会被拒绝覆盖；更换实验时选择新目录。旧 checkpoint 缺少划分记录，不能用于新的正式校准，仍可用于原有推理或独立诊断。

## 真实端到端验证

```bash
uv run python tests/e2e/training_pipeline.py --output ../is-dafeiyu-reports/my_pipeline_e2e
uv run python tests/e2e/aligned_distillation.py --run ../is-dafeiyu-reports/my_pipeline_e2e
uv run python tests/e2e/reject_known_failures.py \
  --output ../is-dafeiyu-reports/my_pipeline_rejections \
  --legacy-model-dir legacy/gemini_batch2 \
  --legacy-weighted-model legacy/weighted/model_browser_fp16.onnx \
  --different-checkpoint legacy/teacher/best.pt
```

通过 `--output` 指定独立的验收目录。依赖数据集、缓存的预训练 MobileNetV3 权重和 `data/regression/x` 中的回归图片。该命令验证增量构建、跨集合模板冲突、矛盾标签、两阶段训练、FP32 校准、FP16 导出与实际 CLI 推理，保留命令、日志、哈希及 `report.json`。不会修改 `data/splits` 或既有模型。

历史故障回归另需指定保存的 Gemini-batch2 模型、weighted 模型与不同的教师检查点；这些诊断权重不随候选模型发布。

短训练的统计成绩不作为新模型精度声明；若真实数据下训练或校准不达标，应保留失败产物，不通过降低生产约束伪造成功。另见 `tests/e2e/ACCEPTANCE.md` 的实施前验收条件。
