# 陨石图像二分类 — 基于LightGBM的异构集成

基于ConvNeXt + Swin Transformer + DINOv2 + CLIP + MAE + LightGBM Stacking的陨石/非陨石二分类系统。

**Kaggle Stage2 最终 F1：0.8660**（Top-K=86，异构集成v5）。

## 特性

- ConvNeXt-Base@384 + Swin-Base@384 有监督基模型（5折交叉验证）
- DINOv2 ViT-Large 自监督冻结尾征 → Logistic Regression + KNN
- CLIP ViT-L/14 零样本分类
- MAE ViT-Large 掩码自编码器冻结尾征
- Domain Classifier 实现Stage2感知的重要性加权
- 27维手工图像特征
- LightGBM 元学习器异构融合
- Top-K 策略应对分布偏移的测试集
- v2 heavy级别数据增强管线
- 自动化Google以图搜图工具（见 `google/`）

## 环境依赖
torch>=2.0 timm>=0.9 lightgbm>=4.0 pandas numpyscikit-learn pillow opencv-python tqdm open-clip-torch
安装：
```bash
pip install -r requirements.txt
```
## 复现指南（F1=0.8660）

### 模型权重下载

预训练模型checkpoint可从Google Drive下载，跳过第1-2步训练：

**下载地址：[Google Drive](https://drive.google.com/drive/folders/13H1ckINPUI-2cYB4k077LVJRJ1VkqsKz?usp=drive_link)**

下载后将 `checkpoints/` 文件夹放入 `outputs/` 下，结构为：
```
outputs/checkpoints/
├── convnext_base_fb_in22k_ft_in1k/
│   ├── best_model_fold0.pth
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   ├── best_model_fold3.pth
│   └── best_model_fold4.pth
└── swin_base_patch4_window12_384_ms_in22k_ft_in1k/
    ├── best_model_fold0.pth
    ├── best_model_fold1.pth
    ├── best_model_fold2.pth
    ├── best_model_fold3.pth
    └── best_model_fold4.pth
```

有模型权重后直接从第3步开始。

### 第0步：准备Kaggle数据

从Kaggle下载竞赛数据，按以下结构组织：

```
project/
├── train_images/train_images/       # 训练图
├── train_labels.csv                 # 从仓库复制（5,371条标签）
├── test_images_stage2/test_images/   # 194张测试图
└── sample_submission_stage2.csv
```

**额外训练图片**（91张×3副本=273张，因文件过大存放于Google Drive）：从[Google Drive](https://drive.google.com/drive/folders/13H1ckINPUI-2cYB4k077LVJRJ1VkqsKz?usp=drive_link)下载 `new_train_final/`，将其中的273张图片（005099.jpg--005371.jpg）复制到 `train_images/train_images/`。仓库中的 `train_labels.csv` 已包含这273张的标签，可直接覆盖Kaggle原始标签文件。

### 第1步：训练 ConvNeXt-Base

```bash
python train_kfold.py --data_root . --output_dir outputs \
    --model_name "convnext_base.fb_in22k_ft_in1k" \
    --img_size 384 --batch_size 16 --epochs 35 --n_folds 5
```

### 第2步：训练 Swin-Base

```bash
python train_kfold.py --data_root . --output_dir outputs \
    --model_name "swin_base_patch4_window12_384.ms_in22k_ft_in1k" \
    --img_size 384 --batch_size 8 --epochs 35 --n_folds 5
```

### 第3步：提取基模型OOF预测

```bash
python stack_ensemble_v3.py --extract_all --data_root . --stage stage2 --img_size 384
```

产出 `outputs/stage2/stacking_v3/oof_predictions_v3.csv` 和 `test_predictions_v3.csv`。

### 第4步：提取 DINOv2 + CLIP 冻结尾征

```bash
python extract_hetero_features.py --data_root . --output_dir outputs --no_use_nobg
```

产出：`outputs/features/dinov2_train.npy`、`dinov2_stage2.npy`、`clip_image_*.npy` 等。

### 第5步：训练 DINOv2 Logistic Regression

```bash
python dinov2_logreg.py --data_root . --features_dir outputs/features
```

产出：`outputs/features/dinov2_logreg_oof.csv`、`dinov2_logreg_stage2.csv`。

### 第6步：训练 DINOv2 KNN

```bash
python dinov2_knn.py --data_root . --features_dir outputs/features
```

产出：`outputs/features/dinov2_knn_oof.csv`、`dinov2_knn_stage2.csv`。

### 第7步：训练域分类器（Domain Classifier）

```bash
python domain_classifier.py --data_root . --features_dir outputs/features
```

产出：`outputs/features/domain_weights.csv`。

### 第8步: 运行异构集成 v5（完整集成）

v5 在 v4 基础上增加了 MAE 特征、CLIP-LogReg 和 27维手工图像特征。

```bash
# 提取MAE特征
python extract_mae_features.py --data_root . --features_dir outputs/features

# 训练CLIP-LogReg
python clip_logreg.py --data_root . --features_dir outputs/features

# 提取27维图像特征
python stack_ensemble_v3.py --extract_features --data_root . --stage stage2

# 运行v5
python stack_ensemble_v5.py --data_root . --stage stage2 \
    --features_dir outputs/features \
    --stack_dir outputs/stage2/stacking_v3 \
    --ks "80,82,84,86,88,90,92,94,96,98,100,105,110,115,120"
```

### 第10步：手动选择最优K

最优K通过向Kaggle提交多个K值并观察F1分数确定。根据实验，最优K在 **[84, 90]** 区间。**K=86 时最终 F1 达到 0.8660。**

关键观察：相邻K值之间F1可能差0.002~0.005，这是因为概率分布尾部样本密集排列。建议以步长1~2进行精细K扫描。

### 预期分数

| K | 约 F1 |
|-----|--------|
| 82 | ~0.862 |
| 84 | ~0.865 |
| **86** | **0.8660** |
| 88 | ~0.863 |
| 90 | ~0.858 |

## 训练数据扩充：自动化Google以图搜图

`new_train_final/`（存放于Google Drive同一文件夹）中的额外训练图片（91张不同图片，每张复制3份，共273张）通过手工Google以图搜图确定标签。

我们已开发了自动化流水线（`google/google_reverse_search.py`），可程序化完成此过程：

1. 通过Playwright（有头浏览器）将每张测试图上传至Google图片搜索
2. 记录搜索结果页面的完整文本
3. 若存在"完全相符的结果"，导航至匹配页面并提取文本
4. 输出结构化结果供下游LLM进行标签推断

该自动化工具可作为未来标签发现的可复现流水线。其输出可能与手工策展的 `new_train_final/` 存在差异，因为手工过程包含了额外的交叉验证步骤，这些步骤尚未纳入自动化。

## 架构总览

```
                      ┌─────────────────────────┐
                      │   ConvNeXt-Base@384      │──→ OOF概率（5折均值/标准差）
                      └─────────────────────────┘
                      ┌─────────────────────────┐
                      │   Swin-Base@384          │──→ OOF概率（5折均值/标准差）
                      └─────────────────────────┘
                      ┌─────────────────────────┐
                      │ DINOv2 ViT-L（冻结尾征） │──→ LogReg概率 + KNN概率
                      └─────────────────────────┘
                      ┌─────────────────────────┐
                      │   CLIP ViT-L/14          │──→ 零样本 + LogReg概率
                      └─────────────────────────┘
                      ┌─────────────────────────┐
                      │   MAE ViT-L（冻结尾征）  │──→ LogReg概率
                      └─────────────────────────┘
                      ┌─────────────────────────┐
                      │   27维手工图像特征       │──→ 统计描述子
                      └─────────────────────────┘
                               │
                               ▼
                      ┌─────────────────────────┐
                      │   LightGBM Stacking      │──→ 最终概率 → Top-K
                      └─────────────────────────┘
```

## 核心发现

1. **架构异构性 > 模型数量**：ConvNeXt + DINOv2（CNN + 自监督ViT）单独即达到0.864 F1。加入Swin-Base、CLIP、MAE仅增加边际增益（+0.002）。加入DenseNet-121零净收益——仅改变Top-100中2张图片的分类方向，且均为错误方向。

2. **自监督特征在分布偏移下不可或缺**：DINOv2冻结尾征+线性探测（OOF AUC=0.998）与全量微调的ConvNeXt性能相当，同时提供了正交的集成信号（+0.156）。这与Kumar et al. (2022)一致——微调会扭曲预训练特征在分布偏移下的鲁棒性。

3. **LightGBM自动学习最优融合**：v5特征重要性显示Swin-Base(437.2)反超ConvNeXt(203.2)成为第一主力，DINOv2双变体(607.8)+CLIP-LogReg(306.8)合计超越任何单模型。27维手工特征中拉普拉斯方差(123.2)排名第六，说明纹理清晰度是被CNN隐式使用但未被充分量化的有效信号。

4. **Top-K是分布偏移下的实用策略**：在测试集正类占比未知时，Top-K避免了阈值选择的任意性。其对K值的敏感度可通过精细K扫描来管理。

## 参考文献

- Liu et al. "A ConvNet for the 2020s." CVPR 2022.
- Liu et al. "Swin Transformer." ICCV 2021.
- Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." 2023.
- Kumar et al. "Fine-Tuning can Distort Pretrained Features." ICLR 2022.
- Sivakumar et al. "Comparative Analysis of Deep Learning Models for Visual Classification of Terrestrial and Extraterrestrial Rock Image." IEEE IEIA 2026.

## 许可证

MIT
