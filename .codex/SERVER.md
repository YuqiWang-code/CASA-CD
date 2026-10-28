# RSML-3 / CASA-CD

| 用途 | 路径 |
|---|---|
| 本地工作区 | `F:\Code_Repositories_2\CursorCode\CASA-CD` |
| 服务器项目 | `/home/yqwang/projects/CASA-CD` |
| 数据集 | `/share_datasets/CD/<dataset>/{A,B,label,list}/` |
| 日志与运行源码 | `/home/yqwang/outputs/CASA-CD/<tag>/<Run>` |
| checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/<tag>/<Run>` |
| MobileNetV2 预训练 | `/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth` |

## 环境

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
cd /home/yqwang/projects/CASA-CD
```

本地 PyTorch 存在 DLL 加载限制，模型检查可在服务器独立临时目录中用 CPU 执行；真实数据探针和显存测量使用对应轮次的 GPU 检查脚本。

## 训练

先读取实时队列、进程和 GPU 占用。每轮脚本集中在 `train_scripts/<tag>/<Run>/`；`pipeline.py` 为唯一调度器，`pipeline_status.json`、`placement.json` 记录实验和物理 GPU 分配。启动前查看已有状态以避免重复队列。

CASA-Pyramid Run6 使用 `models/train_CASA.py`，协议 `casa_pyramid_40k_taskgeometry_v6`。仅物理GPU0，四库各双曲/CTRL共8项，按实测显存与1GiB留量补位，数据集优先级 LEVIR→SYSU→WHU→CDD，允许跨库并行。模型、数值几何、梯度和部署折叠检查通过后做真实数据短训，随后从预训练骨干全新初始化正式实验。探针不是正式指标。

当前Run6每epoch使用test F1选模，manifest标注非独立测试。训练数据使用同步裁剪、翻转、90度旋转与时相交换，RGB归一化配置见README。

在下一轮部署前保存该轮源码、参数与环境。训练运行期间不覆盖其模型或训练代码。按精确 PID / 本轮停止脚本处理本项目队列，保留旧日志和 checkpoint；失败先读具体错误，不循环重启。

Shell 文件使用 LF，修改过的启动脚本执行 `bash -n`；模型改动检查对应的数值、梯度、接口和部署行为。测试范围应覆盖具体改动引入的失败风险。

## 结果

```powershell
python tools/download_training_results.py --tag CASA-Pyramid --run Run6
python train_scripts/CASA-Pyramid/Run6/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

CASA-Pyramid Run6 collector 从本地 `outputs/CASA-Pyramid/Run6/source_code.tar.gz` 提取实际训练源码，生成该轮冻结快照。当前 `models/` 的快照使用 `current`，不覆盖历史轮次。

完成条件为每项日志具有完整 `TEST RESULTS` 和 `END TEST RESULTS`。训练或验证指标、探针、停止项不写为正式结果。更新指标时逐单元格保留历史记录；新正式结果对应的 README、XLSX、TXT 和说明提交推送，输出、权重与连接配置保持本地。
