# 最终实验结果

日期：`2026-03-21`

这份文档是当前仓库的结果总表，目的不是只给一个“最好数字”，而是把当前项目已经拿到的主结果、对照结果、分类别结果、阈值选择和历史记录尽量完整地写清楚，方便后续写申报书、中期检查、结题材料和答辩。

## 1. 当前主结果对应的实验

当前最优、也是最贴近申报书路线的实验是：

- 实验名称：`vlm_stage2_proposal_mscope_large_384`

对应路线：

1. 先训练纯视觉 `MobileUNet-FPN`
2. 用纯视觉最佳权重初始化提示式视觉语言模型视觉分支
3. 用公开数据做辅助预训练
4. 回到 A4C `13` 类目标域精调
5. 最终以 `13` 类结果作为正式评估口径

当前主结果使用的关键配置：

- 视觉骨架：`Promptable MobileUNet-FPN`
- 文本编码器：`clip-vit-large-patch14`
- 文本编码器来源：`ModelScope` 本地缓存
- 视觉初始化权重：`a4c13_mobileunet_fpn_pretrained_512/best.pt`
- 图像尺寸：`384`
- 单提示词二值阈值：`0.5`
- `13` 类重组阈值：`0.15`
- 推理策略：每个类别使用多提示词概率平均后再重组

对应主要文件：

- 最优权重：
  [best.pt](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/best.pt)
- 训练历史：
  [history.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/history.json)
- 主结果指标：
  [test_metrics_with_map_ensemble_tuned.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json)
- 不带调优的测试指标：
  [test_metrics_with_map.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map.json)
- `13` 类测试评估：
  [a4c13_eval_testing_tuned.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/a4c13_eval_testing_tuned.json)
  [a4c13_eval_testing.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/a4c13_eval_testing.json)

## 2. 数据规模与标签口径

当前 `13` 类目标域数据来自 `prepared/a4c13`，标签固定为：

- `DAO`
- `LA`
- `RA`
- `LV`
- `RV`
- `VS`
- `IS`
- `SP`
- `RB`
- `LVW`
- `RVW`
- `LL`
- `RL`

目标域划分如下：

| 划分 | 图像数 |
| --- | ---: |
| `training` | `25` |
| `validation` | `15` |
| `testing` | `10` |

测试集各类别覆盖情况：

- `13` 个类别在测试集 `10` 张图中全部出现
- `IS` 有 `15` 个实例
- `RB` 有 `20` 个实例
- 其余多数类别有 `10` 个实例

测试集像素量统计：

| 类别 | 像素数 |
| --- | ---: |
| `DAO` | `10969` |
| `LA` | `49419` |
| `RA` | `44323` |
| `LV` | `58472` |
| `RV` | `46863` |
| `VS` | `26402` |
| `IS` | `18637` |
| `SP` | `99702` |
| `RB` | `191264` |
| `LVW` | `29611` |
| `RVW` | `29109` |
| `LL` | `270273` |
| `RL` | `334303` |

来源文件：

- [prepared/a4c13/summary.json](/Users/mico/Documents/za/git/segmentation/prepared/a4c13/summary.json)

## 3. 当前最优主结果

### 3.1 正式汇报推荐值

当前推荐用于汇报、部署和对外展示的结果，是多提示词平均后、用验证阶段选出的阈值 `0.15` 在测试集上的表现：

| 指标 | 数值 |
| --- | ---: |
| `threshold` | `0.15` |
| `mean_iou_fg` | `0.4909195303916931` |
| `mean_dice_fg` | `0.6474736928939819` |
| `mean_iou_all` | `0.5206787586212158` |
| `mean_dice_all` | `0.6691922545433044` |
| `mAP@50` | `0.443046398046398` |
| `mAP@50:95` | `0.13897069597069597` |

这组数据对应文件：

- [test_metrics_with_map_ensemble_tuned.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json)

### 3.2 不带最终调优时的测试值

如果不使用最终调优后的多提示词集成结果，当前模型在测试集上的另一组结果是：

| 指标 | 数值 |
| --- | ---: |
| `mean_iou_fg` | `0.47731906175613403` |
| `mean_dice_fg` | `0.6333135962486267` |
| `mean_iou_all` | `0.507836639881134` |
| `mean_dice_all` | `0.6559262871742249` |
| `mAP@50` | `0.44754884004884005` |
| `mAP@50:95` | `0.14526648351648352` |

对应文件：

- [test_metrics_with_map.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map.json)

说明：

- 调优后的正式主结果在 `IoU` 和 `Dice` 上更高
- 不调优那组在整体 `mAP` 上略高
- 由于项目主任务是语义分割，当前正式汇报仍以调优后的 `IoU/Dice` 主结果为主

## 4. 当前最优主结果的分类别详细指标

以下表格对应当前正式主结果，也就是：

- 多提示词平均
- `13` 类阈值 `0.15`
- 测试集结果

| 类别 | IoU | Dice | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: | ---: |
| `DAO` | `0.520332` | `0.684498` | `0.414286` | `0.122857` |
| `LA` | `0.623265` | `0.767916` | `0.722222` | `0.238056` |
| `RA` | `0.668253` | `0.801141` | `0.788889` | `0.342000` |
| `LV` | `0.544586` | `0.705154` | `0.571429` | `0.171429` |
| `RV` | `0.460863` | `0.630946` | `0.483333` | `0.125833` |
| `VS` | `0.515065` | `0.679924` | `0.700000` | `0.085000` |
| `IS` | `0.434029` | `0.605328` | `0.166667` | `0.033333` |
| `SP` | `0.432327` | `0.603671` | `0.187500` | `0.050000` |
| `RB` | `0.465609` | `0.635380` | `0.271111` | `0.030444` |
| `LVW` | `0.240793` | `0.388127` | `0.000000` | `0.000000` |
| `RVW` | `0.239822` | `0.386865` | `0.016667` | `0.001667` |
| `LL` | `0.598551` | `0.748867` | `0.637500` | `0.291250` |
| `RL` | `0.638460` | `0.779342` | `0.800000` | `0.314750` |
| `13 类平均` | `0.490920` | `0.647474` | `0.443046` | `0.138971` |

当前可以直接读出的结论：

- 最稳定的大结构：`RA`、`RL`、`LA`、`LL`
- 中等表现结构：`DAO`、`LV`、`VS`、`RV`、`RB`
- 较难结构：`IS`、`SP`
- 最难结构：`LVW`、`RVW`

## 5. 纯视觉基线对照组

当前纯视觉最佳基线为：

- 实验名称：`a4c13_mobileunet_fpn_pretrained_512`

对应文件：

- [test_metrics.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321/experiments/a4c13_mobileunet_fpn_pretrained_512/test_metrics.json)

纯视觉基线总体指标：

| 指标 | 数值 |
| --- | ---: |
| `mean_iou_fg` | `0.4285956025123596` |
| `mean_dice_fg` | `0.5885782837867737` |
| `mean_iou_all` | `0.4629155993461609` |
| `mean_dice_all` | `0.614563524723053` |
| `loss` | `0.48466441631317136` |

之前项目记录过的纯视觉 `mAP` 汇总值为：

- `mAP@50`：`0.402747`
- `mAP@50:95`：`0.129728`

当前本地已保存的纯视觉分类别 `IoU/Dice` 如下：

| 类别 | IoU | Dice |
| --- | ---: | ---: |
| `background` | `0.909075` | `0.952372` |
| `DAO` | `0.481197` | `0.649741` |
| `LA` | `0.556221` | `0.714836` |
| `RA` | `0.545417` | `0.705851` |
| `LV` | `0.493499` | `0.660863` |
| `RV` | `0.449714` | `0.620417` |
| `VS` | `0.468391` | `0.637965` |
| `IS` | `0.367780` | `0.537777` |
| `SP` | `0.371792` | `0.542053` |
| `RB` | `0.456047` | `0.626418` |
| `LVW` | `0.201403` | `0.335280` |
| `RVW` | `0.159971` | `0.275819` |
| `LL` | `0.433916` | `0.605218` |
| `RL` | `0.586395` | `0.739280` |

### 5.1 当前主结果与纯视觉对照

| 指标 | 纯视觉基线 | 当前主结果 | 提升 |
| --- | ---: | ---: | ---: |
| `mean_iou_fg` | `0.428596` | `0.490920` | `+0.062324` |
| `mean_dice_fg` | `0.588578` | `0.647474` | `+0.058895` |
| `mean_iou_all` | `0.462916` | `0.520679` | `+0.057763` |
| `mean_dice_all` | `0.614564` | `0.669192` | `+0.054629` |
| `mAP@50` | `0.402747` | `0.443046` | `+0.040299` |
| `mAP@50:95` | `0.129728` | `0.138971` | `+0.009243` |

当前结论很明确：

- 当前最优的申报书对齐模型已经整体超过纯视觉基线
- 提升主要体现在 `IoU` 和 `Dice`

## 6. 同路线的其他实验结果

除了当前主结果，本地还保存了几组已经实际跑过的实验，用来说明模型优化过程。

### 6.1 第一版较弱的 CLIP-LoRA 原型

实验文件：

- [test_metrics.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321/experiments/vlm_stage2_final_384_lr2e4/test_metrics.json)
- [a4c13_eval_testing.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321/experiments/vlm_stage2_final_384_lr2e4/a4c13_eval_testing.json)

该版本测试结果：

| 指标 | 数值 |
| --- | ---: |
| 提示式二值测试 `dice` | `0.32712086209884056` |
| 提示式二值测试 `iou` | `0.21932328137067647` |
| `13` 类 `mean_iou_fg` | `0.21657603979110718` |
| `13` 类 `mean_dice_fg` | `0.3456185460090637` |
| `13` 类 `mean_iou_all` | `0.2613568603992462` |
| `13` 类 `mean_dice_all` | `0.3862965703010559` |

这条线说明：

- 只做弱融合时效果明显不够
- 后续“纯视觉权重初始化 + 更强文本编码器 + 更贴近申报书的融合”是必要的

### 6.2 当前主线实验矩阵中的其他结果

当前还保存了几组对照试验：

| 实验 | 测试 loss | 测试 dice | 测试 iou |
| --- | ---: | ---: | ---: |
| `stage1_test_metrics` | `0.2894054046043983` | `0.5068641401254214` | `0.37982818736479834` |
| `stage2_384_lr5e5` | `0.24601767636262453` | `0.5723089749996478` | `0.4428874208376958` |
| `stage2_512_lr5e5` | `0.26890349021324744` | `0.5365492234024434` | `0.40235725436896946` |
| `stage2_final_384_lr2e4` 旧原型 | `0.38224298541362467` | `0.32712086209884056` | `0.21932328137067647` |
| `stage2_proposal_mscope_large_384` 当前主线 | `0.23240489684618437` | `0.5951622885007125` | `0.4626314415381505` |

结论：

- 当前主线在二值提示式任务上也是这批结果里最优
- `384` 分辨率、较强文本编码器和纯视觉初始化更适合当前任务
- `512` 分辨率并没有在当前小样本条件下带来更优结果

## 7. 训练过程摘要

当前主线训练历史文件：

- [history.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/history.json)

已保存训练轮数：

- `25` 个 epoch

按验证集 `IoU` 排序，前五个 epoch 为：

| 排名 | epoch | val loss | val dice | val iou |
| --- | ---: | ---: | ---: | ---: |
| 1 | `18` | `0.24269525775897136` | `0.5925383801561281` | `0.45538370851415894` |
| 2 | `17` | `0.24637507416845597` | `0.5887075693600403` | `0.45038307176599346` |
| 3 | `16` | `0.24457353030897908` | `0.5882470724760758` | `0.4499838837895083` |
| 4 | `24` | `0.2471580732729017` | `0.5853629819164615` | `0.44823145807625536` |
| 5 | `14` | `0.24543850984155519` | `0.5870667003832047` | `0.44768520083324026` |

最后一个保存 epoch：

| epoch | val loss | val dice | val iou |
| --- | ---: | ---: | ---: |
| `25` | `0.2537948595187099` | `0.5715432904726401` | `0.4341704071478695` |

说明：

- 最优验证性能出现在 `epoch 18`
- 后续 epoch 并没有继续稳定提升
- 这说明当前模型在小样本条件下已经接近平台期

## 8. 阈值选择记录

### 8.1 不做多提示词集成时的阈值扫参

文件：

- [threshold_sweep.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/threshold_sweep.json)

在这份记录里，按 `mean_iou_fg` 最优的阈值是：

| 阈值 | mean_iou_fg | mean_dice_fg |
| --- | ---: | ---: |
| `0.1` | `0.45808008313179016` | `0.6190768480300903` |

### 8.2 多提示词集成时的阈值扫参

文件：

- [threshold_sweep_ensemble.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/threshold_sweep_ensemble.json)

在这份记录里，按 `mean_iou_fg` 最优的阈值是：

| 阈值 | mean_iou_fg | mean_dice_fg |
| --- | ---: | ---: |
| `0.15` | `0.4549965262413025` | `0.6163468956947327` |

因此，当前部署和最终评估采用：

- 多提示词集成
- `13` 类重组阈值 `0.15`

## 9. 已拉回本地的关键结果文件

这些文件已经在本地，可以直接用于换服务器部署、继续分析或写报告：

- 权重：
  [best.pt](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/best.pt)
- 主结果：
  [test_metrics_with_map_ensemble_tuned.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json)
- 非调优测试结果：
  [test_metrics_with_map.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map.json)
- `13` 类测试结果：
  [a4c13_eval_testing_tuned.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/a4c13_eval_testing_tuned.json)
- 训练历史：
  [history.json](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/history.json)
- 早期纯视觉和弱融合结果：
  [server_artifacts/20260321](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321)

## 10. Git 历史中的旧记录

在提交 `ffeb999` 中，`README.md` 曾记录过一条非常简短的旧实验结果：

```text
2026 年 2 月 25 日：
Epoch 175 | IoU: 0.8697 | mAP50: 1.0000 | mAP50-95: 0.7900
```

这条记录来自更早的实验阶段，但当前仓库里没有与之完全对应的完整配置、数据口径和结果文件，因此：

- 可以把它视为历史记录
- 不能把它当作当前正式主结果
- 当前正式主结果仍以上文完整保存的 `server_artifacts` 为准

## 11. 当前最适合直接写进材料的结论

如果要写进申报书、中期检查或结题材料，当前最稳的表述是：

1. 项目最终仍然围绕胎儿四腔心 `13` 类结构分割展开，没有偏离主题。
2. 在现有小样本目标域数据和公开辅助数据条件下，当前最优的提示式视觉语言模型在 `13` 类测试集上取得：
   `mean_iou_fg = 0.4909`，`mean_dice_fg = 0.6475`，`mAP@50 = 0.4430`，`mAP@50:95 = 0.1390`。
3. 相比纯视觉 `MobileUNet-FPN` 基线，当前主结果在 `IoU`、`Dice` 和 `mAP@50` 上均有提升。
4. `LVW`、`RVW` 仍是最难类别，说明当前工作的主要瓶颈依然是小结构分割与样本规模不足。
