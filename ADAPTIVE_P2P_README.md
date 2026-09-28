# 内容自适应软融合 P2P

这是针对 SD 1.4 的小规模研究实现。三种方法共享同一模型、DDIM latent、token 对齐及低分辨率 self-attention 替换，只改变编辑词的 cross-attention 融合。

| method | 编辑词的 source attention 权重 |
|---|---|
| `p2p` | 原版二值逐词时间窗口 |
| `soft_fixed` | 余弦时间窗口 × 0.75 |
| `soft_adaptive` | 余弦时间窗口 × clip(1 − 0.5 × 文本距离 − 0.5 × 前一步注意力 EMA, 0.2, 1) |

所有层和 head 共用权重；没有训练、逐层参数或逐头参数。未修改词全程沿用源注意力。`LocalBlend` 关闭；self-attention 在前 40% 步、空间位置数≤256 时替换。新控制器保持原版 steps+1 的整数时间窗口边界，因此 hard 与 soft 窗口的离散端点不完全相同。

## 在 VS Code 的 Colab 扩展运行

本地 notebook 和远端 kernel 是两台机器：`C:\Users\...` 不能被 Colab 的 Linux Python 读取。
本地 `gvproj` 仅供开发/测试；选择 Colab kernel 后，依赖必须安装在远端的 `sys.executable` 中。
Notebook 已改为无网页弹窗的启动流程，也可以用于网页 Colab。

1. 在 VS Code 打开 `adaptive_p2p_colab.ipynb`。右上角 **Select Kernel → Colab → New Colab Server**，选择 T4。
2. 在**本地** Explorer 右键 `adaptive_p2p_colab_bundle.zip` → **Upload to Colab**。不要上传整个项目中的 `.model_cache` 或本地 conda 环境。
3. 在 Colab 侧栏 **Contents** 确认 ZIP 的远端路径。默认 `/content/adaptive_p2p_colab_bundle.zip`；不一致时修改 notebook 第一节的 `BUNDLE_PATH_OVERRIDE`。默认解压到 `/content/adaptive_p2p`。
4. 推荐使用命令 **Colab: Mount Google Drive to Server...**，执行扩展插入的挂载单元格，再运行项目第一节。程序只检测已挂载 Drive，不调用网页授权挂载。
5. 运行安装单元格后，点击 notebook 工具栏 **Restart Kernel**。重新运行第一节，跳过安装，再继续。不要删除服务器；删除服务器与重启 kernel 不同。
6. 没有 PIE-Bench 时保留 `DATA_ROOT=None`，先完成四组生成图。要做真实图实验，从 [官方入口](https://github.com/cure-lab/PnPInversion#benchmark-download) 获取数据，通过扩展上传并解压，或放在已挂载 Drive，将 `DATA_ROOT` 指向**远端**数据目录。

```text
PIE-Bench/
  mapping_file.json
  annotation_images/
    ...官方图像目录...
```

6. 运行预下载，再运行实验。预下载较大，使用原模型和官方指标模型，不下载无关模型；下载不计入一小时实验预算。
7. 默认结果位置：已挂载 Drive 时为 `/content/drive/MyDrive/p2p_research/`，否则为临时目录 `/content/p2p_results/`。可通过 `RESULTS_ROOT_OVERRIDE` 明确指定；已开始实验后保持路径不变。没有挂载时指定 Drive 路径会报错，避免误把普通临时文件夹当成 Drive。
8. 运行预下载、实验及导出单元格。通过扩展的远端文件视图获取 `*_report.zip` 和 `*_checkpoint.zip`，或保存在 Drive。保存本地 notebook 不会自动下载远端图像和 CSV。

| 设置 | 示例 | 含义 |
|---|---|---|
| `PROJECT_DIR_OVERRIDE` | `/content/project` | 已上传/克隆源码时指定；否则默认解压路径 |
| `BUNDLE_PATH_OVERRIDE` | `/content/uploads/adaptive_p2p_colab_bundle.zip` | ZIP 的真实远端路径 |
| `RESULTS_ROOT_OVERRIDE` | `/content/drive/MyDrive/p2p_research` | 结果根目录，Drive 需预先挂载 |
| `DATA_ROOT` | `/content/PIE-Bench` | 包含 mapping_file.json 和 annotation_images 的目录 |

本地修改 `.py` 后，运行 `conda run -n gvproj python scripts/build_colab_notebook.py` 重建包并重新上传。
若远端已导入旧模块，先 **Restart Kernel** 再运行第一节；新 ZIP 会被检测并解压。同一个 ZIP 重跑不会覆盖远端源码。
也可上传完整源码目录并设置 `PROJECT_DIR_OVERRIDE`，此时不自动解压默认 ZIP。不要假定本地保存代码就会同步到远端。

断线但服务器仍在时，重新连接同一服务器并使用原路径即可续跑。更换/删除服务器后，临时 `/content` 会丢失，但 Drive 中已写入的文件保留。
只有结果原来在临时磁盘时，才需要将备份的 `*_checkpoint.zip` 解压到**原 `RESULTS_ROOT`**，恢复同名实验目录。
结果直接写入 Drive 时，重新挂载并沿用原目录即可，不需要恢复结果包。新服务器仍需安装依赖。
该包包括 latent；`*_report.zip` 不含 latent，只用于分享。两者都不含模型权重，checkpoint也不包含整个原始数据集。
当前缓存中存有绝对路径，因此恢复时项目、数据和输出路径、环境与代码都要保持一致；路径改变时新开结果目录。

### 一次上传到 Drive，多次使用

先将代码 ZIP 上传到 Drive 根目录，并将 PIE-Bench 解压到 Drive 的 `PIE-Bench` 目录。每次连接新服务器先挂载 Drive，再设置：

```python
PROJECT_DIR_OVERRIDE = None  # 每次从 Drive ZIP 解压到 /content/adaptive_p2p
BUNDLE_PATH_OVERRIDE = "/content/drive/MyDrive/adaptive_p2p_colab_bundle.zip"
RESULTS_ROOT_OVERRIDE = "/content/drive/MyDrive/p2p_research"
# 第四节，真实数据准备好后：
DATA_ROOT = "/content/drive/MyDrive/PIE-Bench"
```

也可把完整源码放到 Drive，设置 `PROJECT_DIR_OVERRIDE` 为其目录，`BUNDLE_PATH_OVERRIDE=None`。
Drive 路径保持固定；不要将已有实验的数据路径从临时磁盘改为 Drive 后直接复用原结果目录。
模型默认缓存于 `/content/p2p_model_cache`，换服务器仍需下载。如希望保留模型，在第一节调用 `setup_runtime` **之前**设置
`os.environ["HF_HOME"]="/content/drive/MyDrive/p2p_model_cache"` 和
`os.environ["TORCH_HOME"]="/content/drive/MyDrive/p2p_model_cache/torch"`。
Drive 空间需充足，许多小文件的读取可能更慢；不能把整个已安装 Python 环境复制到 Drive 来替代安装。

### 第二节安装失败：Python 版本与日志

已保存的 Colab 报错栈包含 `/usr/lib/python3.13/subprocess.py`，但固定的 NumPy 1.26.4 仅支持 Python 3.9–3.12。
这是确定的版本不兼容；旧单元格未保存 pip 输出，所以不能仅凭 `CalledProcessError` 确认 pip 首个失败包。
当前单元格先检查 Python 3.12，再实时显示并保存完整安装日志到 `RESULTS_ROOT/setup_logs/`。
旧报错保存在 `validation/colab_install_failure.json`。

网页 Colab：Runtime → Change runtime type → Runtime Version，选择 `2026.07`（Python 3.12.13），GPU 选择 T4。
如果 VS Code 扩展没有运行时版本选项，先用网页入口运行同一 notebook；不要假设网页选项会自动改变扩展创建的服务器。
选择版本后用 `print(sys.version)` 核实，安装成功后重启 kernel，再运行第一节和第三节。
这解决已知版本门槛，仍需在实际 GPU 运行时完成第三节检查；本地 CPU 检查不能替代远端验证。
参考：[Colab 运行时版本](https://research.google.com/colaboratory/runtime-version-faq.html)、
[NumPy 1.26.4 支持范围](https://numpy.org/devdocs/release/1.26.4-notes.html)。

扩展操作参考 [Google 官方 User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)。
如看不到上传/挂载命令，先检查官方 Google Colab 扩展的版本。

如模型需要认证，使用 `huggingface_hub.login()` 交互输入或远端环境变量 `HF_TOKEN`，不依赖网页 Colab secrets，也不要把 token 写入 notebook。访问失败不会自动更换模型。模型 revision 与依赖版本被记录。

生成图和真实图应使用不同结果目录：第一次使用 `DATA_ROOT=None` 创建的 manifest 不会在以后被悄悄改成真实图实验。

## PIE-Bench 自动获取与重复性检查

Notebook 新增 **3b**：先通过[官方申请入口](https://forms.gle/hVMkTABb4uvZVjme9)获取下载链接，再设置
`USE_PIE_BENCH=True`。可填写 `PIE_DOWNLOAD_URL`（HTTPS ZIP / Google Drive 文件链接），
或将已有 ZIP 放到 `PIE_ARCHIVE_PATH`（默认 `/content/drive/MyDrive/PIE-Bench.zip`）。
默认数据目录为 `/content/drive/MyDrive/PIE-Bench`。程序自动识别 ZIP 内的外层目录；已有完整数据时只检查，不重复下载。
Google Drive 下载使用 `gdown==5.2.0`；如遇权限或配额限制，使用官方手动下载后上传 ZIP，不自动更换数据源。
不支持 RAR、Google Drive 文件夹分享链接或申请表本身作为下载地址。

`dataset_audit.json` 记录所有图像哈希、缺失/损坏文件、相同 RGB 像素的图像和相同编辑任务。
相同图像可能用于不同编辑任务，因此报告重复项但不删除原始数据。
`split_duplicate_audit.json` 检查选定样本的重复图像及开发/测试重叠，重叠时停止且不自动重新抽样。
抽样保持 seed=2026；新 manifest 固定 annotation 文件哈希，续跑时检查标注和选定图像是否变化。
这提供数据与划分的可重复性检查，不等于两次 GPU 推理逐像素一致性验证，也不检测近似/裁剪重复图。

启用真实数据后使用独立的 `pie_small_vscode_v1` 输出目录。已有同名旧实验时手动选择新目录；
本次程序更新后不要混用旧代码的结果缓存。更新 Drive ZIP 后重启 kernel 并重跑第一节；第二节补装新增 gdown。
本地测试使用微型构造数据验证下载/解压后的流程；实际官方数据下载仍需有效授权链接。

## 本地 gvproj（环境）

本地只做 CPU 逻辑测试；实际 512×512 推理由 Colab T4 完成。

```powershell
conda run -n gvproj python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
conda run -n gvproj python -m pip install -r requirements-adaptive.txt
conda run -n gvproj python -m pytest tests -q
```

固定 PyTorch CPU 构建只用于本地。Colab 保留其自带、相互匹配的 torch/torchvision/CUDA，不能在那里安装上述 CPU wheel。Notebook 导出完整运行环境；不同 Colab PyTorch 版本的实验不应混在一个结果目录中。

## API

```python
from adaptive_p2p import EditConfig, prepare_case, invert_case, edit_case, release_editor

config = EditConfig(output_dir="/content/drive/MyDrive/p2p_runs/run01")
case = prepare_case("a photo of a red car", "a photo of a blue car", seed=42)
# source_image="/path/to/512x512.png" 表示真实图；省略则共享随机初始噪声。
initial = invert_case(case, config)
result = edit_case(case, "soft_adaptive", config)
release_editor()
```

完整实验：

```python
from evaluate_adaptive_p2p import prefetch_assets, run_experiment
config = prefetch_assets(config)  # CPU 预下载；返回固定 revision 的 config
run_experiment(config, data_root=None, limit_seconds=3600)
```

命令行入口也可用：`python evaluate_adaptive_p2p.py prefetch`、`run`、`evaluate`、`figures`。`evaluate` 是显式的额外评测操作，不受主实验的预算上限限制；如使用它，应计入新增实验时间。

## 数据与公平性

- 生成图固定为 red/blue car、wooden/glass bowl、bicycle/motorcycle、sitting/standing dog，seed=42–45。
- 真实图筛选四类：对象、姿态、颜色、材质。要求相同空格分词数、可对齐、未超 token 限制、512×512、有效前景/背景掩码；保留 BPE 数不同的替换。
- seed=2026，每类先选 1 张开发图和 3 张测试图；完整 manifest 与排除原因在生成结果前冻结。
- 不支持对象增删和复杂句子改写。相同词数是原 P2P 的操作边界，不等于自动理解所有句子的编辑意图；结果仅适用于该筛选子集。
- 开发集运行原版 τ=0.4/0.6/0.8，根据 CLIP Edit 与背景 LPIPS 的平均名次之和选择全局 τ；平局取较大值。其他方法共享 τ，自适应系数固定。
- 无真实数据时采用预注册 τ=0.6，只做生成图验证，不能称为经过开发集调参。
- 掩码仅供评测，编辑模块不接收官方区域掩码。真实图指标与原输入图比较，不与重建图比较。
- 只计算同一样本上三方法都完成的配对结果；各指标缺失时同步排除该样本的三方法值，并报告有效 n。

## PnP Inversion 指标口径

参考论文是 **PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code (ICLR 2024)**，不是 PnP Diffusion Features。此项目借用评测方法，真实图两端均使用普通 DDIM 反演。

官方来源：

- [评测代码](https://github.com/cure-lab/PnPInversion/tree/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation)
- [论文](https://arxiv.org/abs/2310.01506)
- [DINO 固定提交](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a)

`vendor/reference_manifest.json` 保存官方代码哈希。适配器直接调用未修改的指标方法，仅将模型加载拆分为顺序加载、重置 TorchMetrics 状态、把 DINO 代码来源固定到提交。每次评测导出该运行的协议和环境。

| 指标 | 原始值 | 表格显示 |
|---|---|---|
| Structure Distance ↓ | DINO ViT-B/8 block 11 的 key 自相似矩阵 MSE | ×1000 |
| PSNR 背景 ↑ | 掩码外图像（编辑区域置零），data_range=1 | 原值 |
| LPIPS 背景 ↓ | 同样置零，SqueezeNet | ×1000 |
| MSE 背景 ↓ | 同样置零 | ×10000 |
| SSIM 背景 ↑ | 同样置零 | ×100 |
| CLIP Whole / Edit ↑ | CLIP ViT-L/14，TorchMetrics CLIPScore | 原值，已包含 ×100 |

**保留官方细节：** RLE 是 0-based start/length，C-order；边界行列视作编辑区域。区域指标不按有效像素数量重新归一化。CLIP Edit 将区域外置零后保留完整图像，不裁剪。官方 DINO 实现输入是 0–255 的 float32，随后 Resize(224,max_size=480) 和 ImageNet normalize；这里刻意不添加常见的 `/255`。换口径必须另建结果表，不应继续标成官方口径。

无区域掩码的生成图只报告 CLIP Whole 和结构距离。空掩码或无背景区域指标记作缺失；非有限值在 JSON 记录为 null，不进入均值。理论上完全一致图像的 PSNR=∞ 也按非有限值处理。

## 结果和解释

主要输出：

- `manifest.json`、`run_spec.json`、`environment.json`：固定样本、配置、环境。
- `cache/`：一次 DDIM 反演、源图、重建与初始 latent；缓存绑定代码、模型、数据、配置及环境。
- `runs/`、`registry.json`：三方法图像、source branch、逐步权重、耗时与显存。
- `raw_metrics.csv`、`summary.csv`：逐样本指标、各类别/总体均值和有效 n。
- `paired_bootstrap.csv`：按类别分层的 2,000 次配对 bootstrap，自适应减去对照的差值及95%区间。
- `coverage.json`、`test_selection.json`：未完成样本与按耗时选择的测试规模。
- `qualitative/`：全部完成样本的原图、重建、三方法、评测掩码与权重曲线。
- `blind_review/`：随机 A/B/C 图像、原图、提示词及空评分表。评分前不要打开方法对应表。
- `exploratory_criteria.json`：CLIP Edit 平均提高且背景 LPIPS 恶化≤5%的探索性条件；必须结合图像检查。
- `budget.json`、`status.json`：实际操作耗时、当前完成状态。

统计只作探索性解释：测试集最多12张，每类最多3张，置信区间可能不稳定；每类仅1张时 bootstrap 可能退化。区间跨零说明证据不足，单人定性评分也不能视为独立用户研究。不承诺方法有效，更不能据此声称达到最新水平。

## T4 资源控制与边界

默认512×512、30步DDIM、η=0、编辑CFG=7.5、反演CFG=1、FP16 UNet/文本编码器、FP32 VAE/统计/指标。每次一对源/目标；高分辨率 self-attention 使用SDPA，不储存完整注意力历史。

预下载在计时前完成；运行中记录各任务实际耗时。实验累计预算默认3600秒，编辑预留至少600秒用于指标。根据开发实测，从12/8/4/0张真实测试图中选取预先固定顺序的前缀，不看结果做选择。开发、生成图、模型切换和指标也消耗预算；特别慢的运行可能无法完成真实图测试，应如实查看 coverage。

预算是操作间的软上限，正在运行的单张图或单个指标不会中途杀掉；实际耗时可能小幅超过上限。断线时把未完成操作的预留估计记入预算，真实未记录时长未知。若发生OOM，保留错误并检查其他占用；本实现不静默缩小分辨率或改变步数来混合不同设置的结果。

30步与原论文完整设置不同，且 LocalBlend 被统一关闭：这是受控小规模比较，不是原论文数值复现。自适应距离是启发式，可能因注意力噪声和源轨迹偏差而失效。颜色或姿态替换失败也可能来自基模型或固定self-attention约束。

## 验证状态

本地验证记录见 `validation/`。CPU 测试覆盖原控制器张量一致性、真实tokenizer、多BPE与重复词、CFG分支隔离、实际Diffusers小型随机UNet、DDIM时间表、官方像素指标、区域解码、抽样与统计。

另外已经在CPU加载真实CLIP、DINO、LPIPS预训练权重，使用确定性的测试图像对核对全部七项指标，与固定提交的官方函数在容差内一致；见 `validation/pretrained_metric_parity.json`。这验证评测实现，不是图像编辑效果实验。

**未在本地运行完整 SD 1.4 或 Colab T4 图像实验。** 真实图数据尚未提供；实际编辑结果、T4显存和实际时长必须通过notebook验证。测试图或随机网络输出不能作为研究结论。

## 来源

原 Google P2P 文件保持原状，使用其 Apache-2.0 的 `seq_aligner.py` 与原 notebook 控制器作为验证参考。官方 PnP Inversion 文件按固定提交原样保存，来源和校验信息见 `vendor/README.md` 与 manifest。模型遵循各自模型卡条款。
