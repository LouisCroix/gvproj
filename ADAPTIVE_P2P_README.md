# 内容自适应软融合 P2P

三个主要文件：`adaptive_p2p.py` 实现模型与注意力控制，`evaluate_adaptive_p2p.py` 负责 PIE 数据、调度和评测，`adaptive_p2p_colab.ipynb` 提供 Colab 操作入口。项目目标见 `report/`，原版代码与论文保留。

## 在 Colab 运行

1. 网页 Colab 或 VS Code Colab 扩展选择 GPU kernel（T4或A100），使用 Python 3.12。扩展的远程 kernel 无法读取本地 `C:\Users\...` 路径。
2. 挂载 Google Drive。将 `adaptive_p2p_colab_bundle.zip` 上传到远程 `/content/`，或保存到 Drive 并修改第一节 `BUNDLE_PATH_OVERRIDE`。默认解压到 `/content/adaptive_p2p`。
3. 运行依赖安装格，首次安装后 Restart Kernel，重跑路径格并跳过安装格。保留 Colab 配套 PyTorch/CUDA。
4. 设置 `DATA_ROOT`，其下需要官方 `mapping_file.json` 和 `annotation_images/`。`ti2i_benchmark` 文件可保留，但不参与实验。没有数据时停止，不运行生成图。
5. 在第4节设置数据配置、实验名称和预算，先执行 CPU 预检查，再预下载模型，最后运行实验。
6. 同一实验目录重跑第6节可续跑。`paused_budget` 表示预算暂停，`failed` 表示异常，`complete` 表示计划完成。查看 `coverage.json` 的待完成ID。

默认结果根目录为已挂载的 `/content/drive/MyDrive/p2p_research`；可用 `RESULTS_ROOT_OVERRIDE` 修改。没有 Drive 时使用临时磁盘，关闭服务器前导出 checkpoint。Drive 上的数据与结果无需重新上传；每台新服务器仍需安装依赖并解压代码。

模型缓存默认在临时 `/content/p2p_model_cache`。若需持久化，在导入研究模块前设置 `HF_HOME`、`TORCH_HOME` 为 Drive 路径。Drive 读取模型可能较慢。代码包更新后 Restart Kernel 再解压。模型固定 `CompVis/stable-diffusion-v1-4` 并记录实际 revision，访问失败不会换模型。

## 数据配置

```python
from evaluate_adaptive_p2p import DatasetConfig

# 默认：四类，每类1张开发、3张测试
dataset_config = DatasetConfig()

# 四类全量，测试所有合格非开发记录
dataset_config = DatasetConfig(sampling="all", test_per_category=None)

# 全部10类中符合当前协议的记录
dataset_config = DatasetConfig(categories="all", sampling="all", test_per_category=None)

# 每类不同配额；字典需覆盖所选类别
dataset_config = DatasetConfig(
    categories=("1", "5"),
    dev_per_category={"1": 2, "5": 1},
    test_per_category={"1": 8, "5": 4},
    split_seed=2026,
)

# 当前官方数据类别9仅1条符合完整协议，可全部用于测试，其他类别提供开发集
dataset_config = DatasetConfig(
    categories="all", sampling="all", test_per_category=None,
    dev_per_category={str(i): (0 if i == 9 else 1) for i in range(10)},
)
```

四类ID为对象1、姿态5、颜色6、材质7，原始共200条；整个 PIE 共700条。全量指通过当前替换协议筛选的记录，扣除开发集，不保证原始200／700条都可评测。要求提示词等空格词数、token可对齐且未截断、存在文本变化、图像512²可读、编辑和背景区域均非空。扫描全部类别不会自动增加新增／删除对象等算法支持。

`prepare_dataset(data_root, output_dir, dataset_config, tokenizer=None, model_config=None)` 只加载分词器，输出 `dataset_preflight.json` 和冻结的 `manifest.json`。不加载扩散模型。空类别跳过，非空类别不足配额报错。开发配额允许某类设0，但整体至少有1张开发图用于全局τ选择；测试配额必须为正。各类独立随机排序；扩大测试配额保留前缀，增加其他类别不扰动已有类别。开发与测试分离。

续跑校验数据配置、标注文件、选定图像、模型、代码和环境。变更后使用新实验目录；新协议不兼容旧 manifest。旧 notebook 运行输出已存为 `results/legacy_pie_small_notebook.ipynb`（如原文件有运行输出）；已有 Drive 实验目录保留。

## 算法与调度

三方法共享初始 latent、DDIM 日程、token mapper 和 self-attention 设置，只改变编辑词 cross-attention 的源注入权重：

| 方法 | 编辑词源权重 |
|---|---|
| `p2p` | 原版二值时间窗口 |
| `soft_fixed` | 余弦窗口 × 0.75 |
| `soft_adaptive` | 余弦窗口 × clip(1 − 0.5s − 0.5前步EMA, 0.2, 1) |

文本距离s由SD文本编码器计算；注意力JS差异使用注入前、空间不超过32²的cross-attention层，EMA更新为0.8旧值+0.2当前值，下步使用。所有层/head共用强度。未修改词全程注入；self-attention沿用官方SD代码的低分辨率替换，前40%步、空间位置数≤256；关闭LocalBlend。保留目标value，不额外行归一化。

默认512²、30步DDIM、eta=0、编辑CFG7.5、反演CFG1、VAE均值编码、UNet/文本FP16、VAE/距离FP32。单样本推理，A100也不自动增大batch。

开发集P2P比较τ∈{0.4,0.6,0.8}；以平均CLIP Edit降序和背景LPIPS升序名次和选择全局τ，平局选较大值。三方法共用τ，测试不调参。开发集软融合诊断单独保存。

测试按类别轮转、每区块最多12张，每张反演一次并缓存，三方法分别运行。每个方法即刻保存。区块完成释放扩散模型，依次加载指标模型。续跑优先补指标和中断结果。`limit_seconds=3600` 是每次调用的活跃时间软上限，10小时可设36000；历史累计用时另存。下载应预先完成，预留至少600秒评测；预算不足暂停，完整名单保留。

## 评测与输出

固定 PnP Inversion 提交和DINO提交，使用 `vendor/reference_manifest.json` 校验来源。官方RLE、边界和区域置零处理保持一致，CLIP Edit使用掩码后的整图，DINO保留官方0–255浮点输入与预处理。LPIPS使用SqueezeNet，CLIP使用ViT-L/14。保持指标始终对原图计算。

- `raw_metrics.csv`：逐样本七项指标、编辑／反演时间、峰值显存。
- `summary.csv`：类别均值、`all`样本平均、`all_category_equal`类别等权平均；同时保存原值与论文缩放值。
- `paired_bootstrap.csv`：自适应减去各对照的配对差值，按类别分层2000次，95%区间。仅比较对应指标三方法完整的样本。
- `coverage.json`：计划数、编辑完成数、全指标完成数、各指标配对数和待完成ID。未完成结果为中间结果。
- `qualitative/`：所有完成测试样本的对比图、掩码、提示词与权重曲线；notebook默认每页8张，可筛类别／ID。
- `blind_review/scores.csv`：稳定A/B/C标签，续跑补新行并保留评分。密钥另存；单人评分仅作主观辅助。
- `budget.json`、`environment.json`、`status.json`：累计操作记录、环境和状态；完整环境快照由notebook保存。

探索性支持标准仍为CLIP Edit均值提高且背景LPIPS相对恶化不超过5%。这是项目阈值；小样本区间跨零只报告趋势，30步结果不与论文完整设置直接比较。

## 本地检查与打包

在 `gvproj` 中先安装匹配的 `torch==2.6.0 torchvision==0.21.0`（本地CPU），再安装 `requirements-adaptive.txt`。Colab不安装本地CPU wheel。

```text
python -m pytest tests -q
python scripts/build_colab_notebook.py
python evaluate_adaptive_p2p.py prepare --data-root PIE-Bench --output-dir results/preflight
python evaluate_adaptive_p2p.py verify-metrics --device cpu --output-dir results/metric-check
```

测试只保留关键算法、指标口径、数据划分、路径及续跑检查。流程测试使用1张开发图＋1张测试图；分块检查仅使用13张测试图覆盖12＋1边界。测试中的bootstrap缩为32次，正式实验仍为2000次。小张量测试使用单CPU线程，不改变实验的线程设置。Notebook的 `RUN_LOGIC_TESTS` 默认关闭，修改代码后可手动开启。

CLI的 `prepare` 和 `run` 支持 `--dataset-config path.json`，JSON字段与DatasetConfig一致；`run` 支持 `--budget-seconds 36000`。固定官方指标一致性检查需要下载真实指标权重。

Notebook是唯一内容来源，构建脚本只检查语法并打包，不覆盖用户设置；兼容旧的 `--bundle-only` 调用。包内含运行代码、离线测试和必要的官方参考文件，不含历史结果、数据和模型。`scripts/runtime_paths.py` 负责路径；`scripts/fetch_reference.py` 仅用于维护固定参考来源，不是实验步骤。

本地模拟流程测试不代表GPU验收。Colab需完成四类各1张测试的三方法对照、跨调用续跑，再运行全量并检查覆盖报告。
