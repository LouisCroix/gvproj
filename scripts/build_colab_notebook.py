"""Rebuild the distributable Colab notebook and compact upload bundle."""
import json
import sys
import textwrap
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SOURCE = (ROOT / "scripts" / "runtime_paths.py").read_text(encoding="utf-8")


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md("""
    # 内容自适应软融合 P2P — Colab T4 小规模比较

    比较 `p2p`、`soft_fixed`、`soft_adaptive`。不训练模型，不采用逐层或逐头权重。
    真实图均使用共享 DDIM 反演，指标参考 **PnP Inversion / PIE-Bench**。

    **VS Code：** 右上角 Select Kernel → Colab → New Colab Server，选择 **T4 GPU**。
    本地 `gvproj` 用于开发和测试，Colab kernel 使用远端 Python；本地文件不会自动同步。
    本笔记本也支持网页 Colab，但不依赖网页上传弹窗。本笔记本没有预填实验结果。
    首次下载 SD 1.4、CLIP、DINO、LPIPS 较大；先下载，再开始约一小时的实验预算。
    原文件保持原状；完整说明见 `ADAPTIVE_P2P_README.md`。
    """),
    md("""
    ## 1. 项目与持久化目录

    在 VS Code 本地资源管理器右键随附的 `adaptive_p2p_colab_bundle.zip` → **Upload to Colab**。
    在 Colab 侧栏 **Contents** 查看实际远端路径；默认检查 `/content/adaptive_p2p_colab_bundle.zip`。
    若上传路径不同，设置 `BUNDLE_PATH_OVERRIDE`；若已上传/克隆完整源码，设置 `PROJECT_DIR_OVERRIDE`。

    推荐先执行命令 **Colab: Mount Google Drive to Server...** 并运行扩展插入的挂载单元格。
    本程序不自行执行网页挂载。已挂载时默认保存到 Drive，否则使用 `/content/p2p_results`（临时）。
    重启 kernel 后重跑本节。只有放在临时 `/content` 的文件才需在更换服务器后恢复。
    可把代码 ZIP 一次上传到 Drive，然后设置
    `BUNDLE_PATH_OVERRIDE="/content/drive/MyDrive/adaptive_p2p_colab_bundle.zip"`；
    每台新服务器会从 Drive 自动解压到同一个 `/content/adaptive_p2p`，无需再次从电脑上传。
    数据放在 Drive，结果直接写入 Drive 时，重新挂载并保持原路径即可，无需解压结果包。
    新服务器仍需重新安装依赖；默认模型缓存位于临时磁盘，需重新下载。
    """),
    code(textwrap.dedent("""
    import os, sys
    from pathlib import Path
    PROJECT_DIR_OVERRIDE = None  # 例如 /content/project；远端不要填写 C:\\Users\\...
    BUNDLE_PATH_OVERRIDE = None  # 例如 /content/uploads/adaptive_p2p_colab_bundle.zip
    RESULTS_ROOT_OVERRIDE = None  # 例如 /content/drive/MyDrive/p2p_research（先挂载）
    REMOTE_ROOT = "/content" if sys.platform == "linux" and Path("/content").is_dir() else None
    IN_COLAB = REMOTE_ROOT is not None
    """) + BOOTSTRAP_SOURCE + "\n\n" + textwrap.dedent("""
    PROJECT_DIR, RESULTS_ROOT = setup_runtime(
        PROJECT_DIR_OVERRIDE, BUNDLE_PATH_OVERRIDE, REMOTE_ROOT, RESULTS_ROOT_OVERRIDE)
    """)),
    md("""
    ## 2. 安装固定研究依赖

    Colab 保留自带的匹配 PyTorch/torchvision/CUDA，不安装本地 CPU wheel。
    本研究依赖使用 **Python 3.12**；Python 3.13 不兼容固定的 NumPy 1.26.4。
    若当前为 3.13，请先切换 Python 3.12 运行时。网页 Colab 可在
    Runtime → Change runtime type → Runtime Version 选择 `2026.07`（Python 3.12.13）。
    若 VS Code 扩展未提供版本选项，可先使用网页 Colab 的上述入口运行；不要替换系统 Python。
    版本列表见 https://research.google.com/colaboratory/runtime-version-faq.html 。
    安装输出实时显示，并保存在结果根目录 `setup_logs/`，便于定位具体 pip 错误。
    **首次安装完成后，点击 VS Code notebook 工具栏 Restart Kernel，再从第 1 节重新运行，跳过安装单元格。**
    本地请选择 `gvproj` kernel，按 README 安装 CPU PyTorch 后运行测试。
    """),
    code("""
    import subprocess, sys
    from datetime import datetime, timezone
    print("Python:", sys.version, "Executable:", sys.executable, flush=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("本项目固定依赖请使用 Python 3.12；当前为 " + sys.version.split()[0]
                           + "。请切换 Colab Runtime Version 后重跑第一节，不要在当前会话替换系统 Python。")
    log_dir = RESULTS_ROOT / "setup_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (datetime.now(timezone.utc).strftime("pip-install-%Y%m%dT%H%M%S%fZ") + ".log")
    command = [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements-adaptive.txt")]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Python: " + sys.version + "\\nCommand: " + repr(command) + "\\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    print("完整安装日志:", log_path)
    if returncode:
        raise RuntimeError(f"pip 安装失败（exit={returncode}）；请查看上方具体 ERROR 或日志 {log_path}")
    print("首次安装后请重启 Python 会话，重新运行第 1 节，再继续第 3 节。")
    """),
    md("""
    ## 3. 环境检查与离线逻辑测试

    测试使用小型随机 UNet 和固定 tokenizer，不下载 SD 权重，也不证明编辑质量。
    原 P2P notebook 仅抽取控制器定义作张量对照，不执行其中的模型下载或演示。
    """),
    code("""
    import json, subprocess
    import torch, torchvision, diffusers, transformers
    from adaptive_p2p import EditConfig, environment_snapshot, save_json
    from evaluate_adaptive_p2p import prefetch_assets, run_experiment, generated_samples, verify_pretrained_metrics

    assert diffusers.__version__ == "0.35.1", "依赖版本不符，请安装后重启"
    print(json.dumps(environment_snapshot(), indent=2, ensure_ascii=False))
    RUN_LOGIC_TESTS = True
    if RUN_LOGIC_TESTS:
        subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings"], check=True)
    """),
    md("""
    ## 4. 数据与运行配置

    尚无真实图时，保留 `DATA_ROOT=None`，四组生成图可以独立运行。
    真实图需先从 [PIE-Bench 官方入口](https://github.com/cure-lab/PnPInversion#benchmark-download)
    获取数据；`DATA_ROOT` 下必须含 `mapping_file.json` 与 `annotation_images/`。

    每类一张开发图、最多三张测试图，类别为对象/姿态/颜色/材质。固定筛选规则和 seed=2026；
    不符合词替换要求的样本会记录排除原因，不能根据编辑效果换图。官方掩码仅供评测。

    **保持 OUTPUT_DIR 不变可续跑；修改代码、模型、依赖或数据时使用新目录。**
    生成图实验和加入真实数据后的完整实验也应使用不同目录。
    """),
    code("""
    DATA_ROOT = None
    # DATA_ROOT = "/content/PIE-Bench"  # 通过扩展上传并解压后的远端目录
    # DATA_ROOT = "/content/drive/MyDrive/PIE-Bench"  # 已挂载 Drive
    if DATA_ROOT is not None:
        DATA_ROOT = str(runtime_path(DATA_ROOT, base=PROJECT_DIR))
    OUTPUT_DIR = RESULTS_ROOT / ("synthetic_vscode_v1" if DATA_ROOT is None else "pie_small_vscode_v1")
    BUDGET_SECONDS = 3600

    assert torch.cuda.is_available(), "完整实验需要 GPU：请在 Colab 选择 T4。本地仅运行前面的测试。"
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
    if DATA_ROOT is not None:
        assert (Path(DATA_ROOT) / "mapping_file.json").exists()
        assert (Path(DATA_ROOT) / "annotation_images").is_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = EditConfig(output_dir=str(OUTPUT_DIR))  # 512px, 30 DDIM steps, CFG=7.5, FP16
    print("Results:", OUTPUT_DIR)
    for sample in generated_samples():
        print(sample["id"], sample["case"]["source_prompt"], "->", sample["case"]["target_prompt"])
    """),
    md("""
    ## 5. 预下载（不计入实验预算）

    包括 SD 1.4、CLIP ViT-L/14、DINO ViT-B/8、LPIPS SqueezeNet。指标模型在 CPU 预加载后释放。
    如果模型需要认证，使用 `huggingface_hub.login()` 的交互输入或远端环境变量 `HF_TOKEN`。
    不依赖网页 Colab secrets，不要把明文 token 保存进 notebook。
    下载失败时先解决网络/访问问题；不会自动切换模型或指标。
    """),
    code("""
    config = prefetch_assets(config)
    print("Pinned SD revision:", config.model_revision)
    # 在CPU核对真实指标权重的官方实现一致性；不是编辑效果实验。
    CHECK_METRIC_PARITY = True
    if CHECK_METRIC_PARITY:
        save_json(OUTPUT_DIR / "pretrained_metric_parity.json", verify_pretrained_metrics("cpu"))
    save_json(OUTPUT_DIR / "prefetch_environment.json", environment_snapshot())
    (OUTPUT_DIR / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    """),
    md("""
    ## 6. 运行或断点续跑

    有真实数据时：开发集原 P2P τ=0.4/0.6/0.8 → CLIP Edit/背景 LPIPS 选 τ →
    固定软融合和自适应开发消融 → 四组生成图 → 按剩余预算选择12/8/4/0张真实测试图 → 指标和图表。

    未修改词采用全程注入，self-attention 前40%步保持原策略，三方法均关闭 LocalBlend。
    为评测预留至少10分钟；计时是操作间的软预算，单次操作或断线可能带来估计误差。
    一小时内是否能完成12张真实图取决于实际吞吐；`coverage.json` 会公开未完成情况。
    """),
    code("""
    result_dir = run_experiment(config, data_root=DATA_ROOT, limit_seconds=BUDGET_SECONDS)
    print("Finished/checkpointed:", result_dir or OUTPUT_DIR)
    for filename in ("status.json", "selected_tau.json", "test_selection.json", "coverage.json"):
        path = OUTPUT_DIR / filename
        if path.exists():
            print(filename, path.read_text(encoding="utf-8"))
    """),
    md("""
    ## 7. 定量结果与资源

    `summary.csv` 的 mean 是原始值、display_mean 是论文样式缩放值；CLIPScore 不重复乘100。
    背景区域按官方置零方式计算，DINO 保留官方0–255 float输入；不与论文不同配置的完整数值直接比较。
    生成图没有人工掩码，只报告结构距离与 CLIP Whole。开发图不进入测试均值。
    bootstrap 是自适应减去对照的配对差值：CLIP正值好，距离负值好；小样本区间可能退化。
    """),
    code("""
    import pandas as pd
    from IPython.display import display, Markdown, Image as DisplayImage
    for filename in ("summary.csv", "paired_bootstrap.csv"):
        path = OUTPUT_DIR / filename
        if path.exists() and path.stat().st_size > 1:
            display(Markdown("**" + filename + "**"))
            display(pd.read_csv(path))
    criteria = OUTPUT_DIR / "exploratory_criteria.json"
    if criteria.exists():
        print(criteria.read_text(encoding="utf-8"))
    registry_path = OUTPUT_DIR / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        timings = [{"id": r["sample_id"], "method": r["result"]["method"],
                    "tau": r["result"]["config"]["tau"],
                    "edit_seconds": r["result"]["edit_seconds"],
                    "peak_memory_mib": r["result"]["peak_memory_mib"]} for r in registry["records"]]
        display(pd.DataFrame(timings))
    """),
    md("""
    ## 8. 全样本定性检查

    查看全部完成样本，不只挑选成功例子。特别检查目标对象仍像源对象、姿态没改动、背景漂移和伪影。
    原图/重建区别用于判断反演误差。若要盲评，请先使用 `blind_review/` 中 A/B/C 图片和 scores.csv，
    暂时不要运行本节或打开方法对应表。评分分别为编辑成功0/1/2、保持0/1/2、伪影有/无。
    """),
    code("""
    for folder in sorted((OUTPUT_DIR / "qualitative").glob("*")):
        prompts = json.loads((folder / "prompts.json").read_text())
        display(Markdown(f"**{folder.name}**\\n\\n{prompts['source_prompt']} → {prompts['target_prompt']}"))
        display(DisplayImage(filename=str(folder / "comparison.png")))
        display(DisplayImage(filename=str(folder / "weights.png")))
    """),
    md("""
    ## 9. 保存运行验收状态与导出

    T4验收包含：至少一张生成图（有真实数据时也至少一张真实图）的三方法运行完成、
    文件可读、source branch跨方法一致、指标覆盖。真正的效果需要阅读指标和图像；
    随机网络CPU测试通过不能替代这一步。没有真实数据时明确标注 real_smoke=false。
    """),
    code("""
    import numpy as np
    from PIL import Image
    from evaluate_adaptive_p2p import valid_result
    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text())
    registry = json.loads((OUTPUT_DIR / "registry.json").read_text())["records"]
    passed = {"generated": [], "real": []}
    for sample in manifest["samples"]:
        if sample["split"] == "dev":
            continue
        runs = [r["result"] for r in registry if r["sample_id"] == sample["id"]]
        if {r["method"] for r in runs} != {"p2p", "soft_fixed", "soft_adaptive"}:
            continue
        assert all(valid_result(r) for r in runs)
        sources = [np.asarray(Image.open(r["source_branch_path"])).astype(int) for r in runs]
        assert all(np.abs(x - sources[0]).max() <= 1 for x in sources)
        passed[sample["kind"]].append(sample["id"])
    validation = {"gpu": torch.cuda.get_device_name(0),
                  "generated_smoke": bool(passed["generated"]), "real_smoke": bool(passed["real"]),
                  "complete_sample_ids": passed, "metrics_report_exists": (OUTPUT_DIR / "summary.csv").exists(),
                  "note": "Runtime integrity only; see coverage and paired metrics for actual effectiveness."}
    save_json(OUTPUT_DIR / "colab_validation.json", validation)
    print(validation)
    # report用于分享；checkpoint包含初始latent，用于恢复到相同的远端绝对路径。
    # 不包含模型缓存或 PIE-Bench 原始数据；更换服务器后需另行准备它们。
    import zipfile
    for kind in ("report", "checkpoint"):
        export_path = OUTPUT_DIR.parent / (OUTPUT_DIR.name + "_" + kind + ".zip")
        with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in OUTPUT_DIR.rglob("*"):
                if path.is_file() and (kind == "checkpoint" or path.suffix != ".pt"):
                    archive.write(path, path.relative_to(OUTPUT_DIR.parent))
        print(kind, "remote path:", export_path)
    print("从 VS Code 的 Colab Contents 获取上述ZIP，或保留在已挂载Drive。保存notebook本身不会下载结果文件。")
    """),
]

for i, cell in enumerate(cells):
    cell["id"] = f"p2p-cell-{i:02d}"
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (gvproj / Colab)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "accelerator": "GPU", "colab": {"name": "adaptive_p2p_colab.ipynb", "provenance": []}})
nbf.validate(notebook)
if "--bundle-only" not in sys.argv:
    nbf.write(notebook, ROOT / "adaptive_p2p_colab.ipynb")

with zipfile.ZipFile(ROOT / "adaptive_p2p_colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for filename in ("adaptive_p2p.py", "evaluate_adaptive_p2p.py", "adaptive_p2p_colab.ipynb",
                     "requirements-adaptive.txt", "ADAPTIVE_P2P_README.md", "prompt-to-prompt/seq_aligner.py",
                     "prompt-to-prompt/ptp_utils.py", "prompt-to-prompt/LICENSE"):
        archive.write(ROOT / filename, filename)
    for directory in ("tests", "vendor", "scripts", "validation"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, path.relative_to(ROOT))
    # Reference code cells only; stored display images are not needed for AST tests.
    reference = json.loads((ROOT / "prompt-to-prompt" / "prompt-to-prompt_stable.ipynb").read_text())
    for cell in reference["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    archive.writestr("prompt-to-prompt/prompt-to-prompt_stable.ipynb", json.dumps(reference))
print("Notebook and Colab upload bundle built.")
