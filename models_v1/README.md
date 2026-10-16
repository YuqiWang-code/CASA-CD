# models/ — ChangeViT baseline 代码（CASA-CD 改造版）

上游：https://github.com/zhuduowang/ChangeViT
（ChangeViT: Unleashing Plain Vision Transformers for Change Detection in Remote
Sensing Images, Pattern Recognition 2025）

## 与上游的差异（改造点）

| 文件 | 改造 |
|---|---|
| `train.py` | 新训练入口（替代上游 `main.py`）：A/B/label + list 格式、config header、每 epoch 单行日志（Loss + 六指标）、`last.pth`/`best_F1=x.pth` 保存、开头与结尾输出参数量/FLOPs |
| `eval.py` | 独立测试入口，加载 `best_F1=*.pth`，输出与 train.py 一致的 `=== TEST RESULTS ===` 区块 |
| `smoke_test.py` | 冒烟测试（无需真实数据）：forward/backward + 参数量/FLOPs |
| `dataset/dataset.py` | list 格式 DataLoader（`<root>/{A,B,label}/` + `<root>/list/*.txt`） |
| `dataset/Transforms.py` | label 二值化改为 lab 统一约定 `gray >= 128` |
| `model/encoder.py` | 预训练权重路径参数化（`--pretrained_weight_path`） |
| `model/resnet.py` | `model_zoo.load_url` → `torch.hub.load_state_dict_from_url`（torch 2.x 兼容） |
| `main.py` | 保留上游原版（未改动），仅作参考 |

训练协议与上游一致：BCE+Dice loss、Adam(lr=2e-4, β=(0.9,0.99), wd=1e-4)、poly LR（0.9 次方 + 200 iter warmup）、`max_steps=80000`、batch 16、256×256、
test 集当验证集按 test F1 选 best、seed 16。

## 用法（服务器）

```bash
cd /home/yqwang/projects/CASA-CD/models
python train.py \
    --dataset LEVIR-CD-256 \
    --dataset_root /share_datasets/CD/LEVIR-CD-256 \
    --train_list /share_datasets/CD/LEVIR-CD-256/list/train.txt \
    --test_list  /share_datasets/CD/LEVIR-CD-256/list/test.txt \
    --pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
    --ckpt_dir /share_datasets/yqwang/checkpoints/CASA-CD/baseline/Run1/LEVIR-CD-256 \
    --model_type tiny --max_steps 80000 --batch_size 16 --gpu_id 0
```
