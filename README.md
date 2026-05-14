# Music Denoising Lab

基于传统频谱门限、小型 U-Net 频谱增强、轻量 Conv-TasNet-style TCN、残差 CNN、tiny DiT-style patch Transformer，以及 Demucs/Denoiser 预训练模型拓展实验的音乐去噪工程。输入带噪音频，输出去噪音频，并生成波形图、频谱图和合成测试集上的 SNR improvement 指标。

本项目没有从零训练大型模型，而是训练一个小型 U-Net 频谱去噪模型。MUSAN 数据集中 `music` 子集作为 clean music，`noise` 子集作为 noise，通过随机 SNR 混合合成训练对。老师给定的 `raw.MP3` 只作为最终验收测试音频，不参与训练，也不计算真实 SNR。

## 目录

```text
music_denoising_lab/
  configs/default.yaml
  data/raw/raw.MP3
  data/raw/musan/
  data/processed/
  outputs/
  scripts/
  src/
```

## 环境配置

推荐 Python 3.10 或 3.11。

```bash
pip install -r requirements.txt
```

MP3 读写需要 FFmpeg。可以安装系统 FFmpeg，也可以使用依赖中的 `imageio-ffmpeg`。

## 数据准备

将课程提供的固定验收音频复制到：

```text
data/raw/raw.MP3
```

下载 MUSAN 官方数据：

```bash
python scripts/download_musan.py --source openslr --out_dir data/raw
```

如果 OpenSLR 较慢，可用 Hugging Face 镜像：

```bash
python scripts/download_musan.py --source hf-mirror --out_dir data/raw
```

解压后应包含：

```text
data/raw/musan/music/
data/raw/musan/noise/
```

当前没有真实数据时，可只为跑通流程生成 demo 资产：

```bash
python scripts/make_demo_assets.py --out_dir data/raw
```

demo 资产只用于 smoke test，正式报告请替换为课程 `raw.MP3` 和 MUSAN。

## 生成训练数据

默认生成 1000/100/100：

```bash
python scripts/prepare_musan_dataset.py \
  --musan_dir data/raw/musan \
  --out_dir data/processed \
  --train_samples 1000 \
  --val_samples 100 \
  --test_samples 100 \
  --segment_seconds 4 \
  --sample_rate 16000
```

小规模跑通：

```bash
python scripts/prepare_musan_dataset.py --train_samples 20 --val_samples 5 --test_samples 5
```

## Baseline 去噪

```bash
python scripts/spectral_gate_baseline.py \
  --input data/raw/raw.MP3 \
  --output outputs/baseline/raw_denoised_spectral_gate.wav
```

输出：

```text
outputs/baseline/raw_denoised_spectral_gate.wav
outputs/baseline/raw_denoised_spectral_gate.mp3
```

## 训练 U-Net

```bash
python scripts/train_unet.py \
  --config configs/default.yaml \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3
```

小规模跑通：

```bash
python scripts/train_unet.py --config configs/default.yaml --epochs 1 --batch_size 2 --num_workers 0
```

训练输出：

```text
outputs/checkpoints/best_unet.pt
outputs/checkpoints/last_unet.pt
outputs/metrics/train_log.csv
outputs/figures/loss_curve.png
```

## 训练轻量 TCN

TCN 是第二个深度学习方法，采用 Conv-TasNet-style 的时域编码、时序卷积 mask 和解码流程，用于和 U-Net 的频谱方法对比。

```bash
python scripts/train_tcn.py \
  --config configs/default.yaml \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3
```

输出：

```text
outputs/checkpoints/best_tcn.pt
outputs/checkpoints/last_tcn.pt
outputs/metrics/train_tcn_log.csv
outputs/figures/tcn_loss_curve.png
```

## 训练残差 CNN 和 Tiny DiT-style

残差 CNN 是轻量频谱卷积对照组；Tiny DiT-style 是小规模 patch Transformer 探索组。后者借鉴 DiT 的 patch-token Transformer 形式，但为课程实验采用监督式 mask 训练，不是完整扩散模型复现。

```bash
python scripts/train_cnn.py --config configs/default.yaml --epochs 10 --batch_size 8
python scripts/train_dit.py --config configs/default.yaml --epochs 30 --batch_size 8
```

推理：

```bash
python scripts/infer_cnn.py \
  --checkpoint outputs/checkpoints/best_cnn.pt \
  --input data/raw/raw.MP3 \
  --output outputs/cnn/raw_denoised_cnn.wav

python scripts/infer_dit.py \
  --checkpoint outputs/checkpoints/best_dit.pt \
  --input data/raw/raw.MP3 \
  --output outputs/dit/raw_denoised_dit.wav
```

## Demucs / Denoiser 预训练拓展实验

Demucs 和 Denoiser 是老师推荐列表中的开源模型。本项目将它们作为 zero-shot 预训练模型参考组：不使用 MUSAN 重新训练，而是直接测试其迁移到本任务上的效果。

```bash
python scripts/run_demucs_pretrained.py \
  --input data/raw/raw.MP3 \
  --output outputs/demucs/raw_denoised_demucs.wav

python scripts/run_denoiser_pretrained.py \
  --input data/raw/raw.MP3 \
  --output outputs/denoiser/raw_denoised_denoiser.wav \
  --model dns64
```

评价：

```bash
python scripts/evaluate_demucs_pretrained.py \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_demucs.csv

python scripts/evaluate_denoiser_pretrained.py \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_denoiser.csv
```

注意：Demucs 主要面向音乐源分离，Denoiser 主要面向语音增强，因此二者在本项目中作为迁移实验和失败案例分析，不作为主方法。

## U-Net 推理 raw.MP3

```bash
python scripts/infer_unet.py \
  --checkpoint outputs/checkpoints/best_unet.pt \
  --input data/raw/raw.MP3 \
  --output outputs/unet/raw_denoised_unet.wav
```

输出：

```text
outputs/unet/raw_denoised_unet.wav
outputs/unet/raw_denoised_unet.mp3
```

## TCN 推理 raw.MP3

```bash
python scripts/infer_tcn.py \
  --checkpoint outputs/checkpoints/best_tcn.pt \
  --input data/raw/raw.MP3 \
  --output outputs/tcn/raw_denoised_tcn.wav
```

输出：

```text
outputs/tcn/raw_denoised_tcn.wav
outputs/tcn/raw_denoised_tcn.mp3
```

## 评价

只在合成 test set 上计算有参考指标：

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/best_unet.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics.csv
```

评价 TCN：

```bash
python scripts/evaluate_tcn.py \
  --checkpoint outputs/checkpoints/best_tcn.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_tcn.csv
```

评价 CNN 和 Tiny DiT-style：

```bash
python scripts/evaluate_cnn.py \
  --checkpoint outputs/checkpoints/best_cnn.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_cnn.csv

python scripts/evaluate_dit.py \
  --checkpoint outputs/checkpoints/best_dit.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_dit.csv
```

`raw.MP3` 没有 clean reference，因此不计算真实 SNR/SDR，只做主观听感、波形图和频谱图对比。

## 可视化

```bash
python scripts/plot_audio.py \
  --raw data/raw/raw.MP3 \
  --baseline outputs/baseline/raw_denoised_spectral_gate.wav \
  --unet outputs/unet/raw_denoised_unet.wav \
  --tcn outputs/tcn/raw_denoised_tcn.wav \
  --cnn outputs/cnn/raw_denoised_cnn.wav \
  --dit outputs/dit/raw_denoised_dit.wav \
  --demucs outputs/demucs/raw_denoised_demucs.wav \
  --denoiser outputs/denoiser/raw_denoised_denoiser.wav \
  --out_dir outputs/figures
```

关键图片：

```text
outputs/figures/waveform_comparison.png
outputs/figures/spectrogram_comparison.png
outputs/figures/waveform_comparison_4way.png
outputs/figures/spectrogram_comparison_4way.png
outputs/figures/waveform_comparison_all.png
outputs/figures/spectrogram_comparison_all.png
outputs/figures/loss_curve.png
```

## 一键运行

Linux / WSL：

```bash
bash scripts/run_all.sh
```

Windows PowerShell 建议按 README 中的命令逐条运行。

## 本组完成内容

- 使用 MUSAN `music + noise` 合成 paired noisy/clean 数据。
- 实现传统频谱门限 baseline。
- 实现 U-Net Spectrogram Denoising，输入 noisy magnitude，输出 mask，并使用 noisy phase 重建。
- 实现轻量 Conv-TasNet-style TCN，作为第二个深度学习对比方法。
- 实现残差 CNN 频谱 mask 和 tiny DiT-style patch Transformer，作为补充对照。
- 测试 Demucs 和 Denoiser 预训练模型，作为老师推荐方法的 zero-shot 拓展实验。
- 对 `raw.MP3` 输出 baseline、U-Net、TCN、CNN、Tiny DiT-style、Demucs 和 Denoiser 多种去噪结果。
- 在合成 test set 上计算 SNR、SNR improvement。
- 输出波形图、频谱图、loss 曲线和指标 CSV，便于实验报告插图。

## 可复现说明

默认配置在 `configs/default.yaml` 中，包含采样率、片段长度、STFT 参数、训练轮数、学习率、随机种子和数据量。数据合成脚本会记录 `data/processed/metadata.csv`，其中包含每条样本使用的 clean/noise 来源和目标 SNR。
