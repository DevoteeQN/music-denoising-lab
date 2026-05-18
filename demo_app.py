from __future__ import annotations

import csv
import os
import socket
import subprocess
import time
import webbrowser
from pathlib import Path

import gradio as gr
import torch

from scripts.infer_cnn import denoise_full_audio as run_cnn
from scripts.infer_cnn import load_model as load_cnn_model
from scripts.infer_dit import denoise_full_audio as run_dit
from scripts.infer_dit import load_model as load_dit_model
from scripts.infer_tcn import denoise_full_audio as run_tcn
from scripts.infer_tcn import load_model as load_tcn_model
from scripts.infer_unet import denoise_full_audio as run_unet
from scripts.infer_unet import load_model as load_unet_model
from scripts.run_demucs_pretrained import combine_stems
from scripts.run_denoiser_pretrained import enhance_audio as run_denoiser
from scripts.run_denoiser_pretrained import load_pretrained as load_denoiser_model
from src.audio_utils import ensure_dir, read_audio, spectral_gate_denoise, write_audio
from src.visualize import plot_multi_spectrogram_comparison, plot_multi_waveform_comparison


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEB_OUT = ROOT / "outputs" / "web_demo"
MODEL_CACHE: dict[str, object] = {}
DEMUX_ENV_NAME = "music_demucs"

LABELS = {
    "baseline": "Spectral Gate",
    "unet": "U-Net",
    "tcn": "TCN",
    "cnn": "Residual CNN",
    "dit": "Tiny DiT-style",
    "demucs": "Demucs",
    "denoiser": "Denoiser",
}

TRAINED = {
    "unet": {
        "checkpoint": ROOT / "outputs" / "checkpoints" / "best_unet.pt",
        "loader": load_unet_model,
        "runner": run_unet,
    },
    "tcn": {
        "checkpoint": ROOT / "outputs" / "checkpoints" / "best_tcn.pt",
        "loader": load_tcn_model,
        "runner": run_tcn,
    },
    "cnn": {
        "checkpoint": ROOT / "outputs" / "checkpoints" / "best_cnn.pt",
        "loader": load_cnn_model,
        "runner": run_cnn,
    },
    "dit": {
        "checkpoint": ROOT / "outputs" / "checkpoints" / "best_dit.pt",
        "loader": load_dit_model,
        "runner": run_dit,
    },
}

DEFAULT_METHODS = ["unet", "tcn", "cnn", "dit"]
OPTIONAL_METHODS = ["demucs", "denoiser"]

CUSTOM_CSS = """
.gradio-container {
  max-width: 1360px !important;
  background:
    radial-gradient(circle at top left, rgba(196, 226, 255, 0.55), transparent 26%),
    radial-gradient(circle at top right, rgba(255, 221, 181, 0.45), transparent 24%),
    linear-gradient(180deg, #f7f3ea 0%, #f3efe5 100%);
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
#hero {
  background: linear-gradient(135deg, #133c55 0%, #24576f 45%, #d98c4b 100%);
  color: #fdf8ef;
  border-radius: 20px;
  padding: 20px 24px;
  box-shadow: 0 16px 38px rgba(19, 60, 85, 0.18);
}
#hero h1, #hero p {
  margin: 0;
}
.metric-note {
  color: #1f3d4d;
}
"""


def load_metrics_table() -> list[list[str]]:
    metrics_path = ROOT / "outputs" / "metrics" / "method_summary.csv"
    if not metrics_path.exists():
        return [["No metrics file", "-", "-", "-"]]

    rows: list[list[str]] = []
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                [
                    row["method"],
                    row["type"],
                    f'{float(row["mean_snr"]):.4f}',
                    f'{float(row["mean_snri"]):+.4f}',
                ]
            )
    return rows


def get_trained_model(name: str):
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]

    spec = TRAINED[name]
    checkpoint_path = spec["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    MODEL_CACHE[name] = spec["loader"](checkpoint_path, DEVICE)
    return MODEL_CACHE[name]


def get_denoiser_model():
    cache_key = "denoiser_dns64"
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_denoiser_model("dns64", DEVICE)
    return MODEL_CACHE[cache_key]


def save_result_audio(job_dir: Path, name: str, audio, sample_rate: int) -> str:
    output_path = job_dir / f"{name}.wav"
    write_audio(output_path, audio, sample_rate)
    return str(output_path)


def get_conda_command() -> str | None:
    configured = os.environ.get("DEMO_CONDA_CMD")
    if configured:
        return configured
    for candidate in [r"D:\anaconda\Scripts\conda.exe", r"D:\anaconda\condabin\conda.bat"]:
        if Path(candidate).exists():
            return candidate
    return "conda"


def run_demucs_isolated(input_path: Path, work_dir: Path, device: str, segment: float = 4.0) -> None:
    conda_cmd = get_conda_command()
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(ROOT / "data" / "raw" / ".torch_home"))
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    cmd = [
        conda_cmd,
        "run",
        "-n",
        DEMUX_ENV_NAME,
        "python",
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "-d",
        device,
        "--shifts",
        "0",
        "--segment",
        str(int(round(segment))),
        "-o",
        str(work_dir),
        str(input_path),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def find_available_port(start_port: int = 7860, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available localhost port in {start_port}-{start_port + attempts - 1}")


def build_status(job_dir: Path, errors: list[str]) -> str:
    status_lines = [
        f"Device: `{DEVICE}`",
        f"Output folder: `{job_dir}`",
        "Note: uploaded audio has no clean reference, so the demo only shows listening results and visualizations.",
    ]
    if errors:
        status_lines.append("Partial failures:")
        status_lines.extend([f"- {message}" for message in errors])
    return "\n".join(status_lines)


def set_running_state() -> tuple[str, dict]:
    return (
        "Running denoising pipeline...\n\nThe first run may take longer while models are loaded.",
        gr.update(value="Running...", interactive=False),
    )


def set_idle_state() -> dict:
    return gr.update(value="Run Denoising", interactive=True)


def process_audio(
    input_path: str,
    methods: list[str],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not input_path:
        raise gr.Error("Please upload an audio file first.")
    selected_methods = list(dict.fromkeys(["baseline", *(methods or [])]))

    started_at = time.perf_counter()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_dir = ensure_dir(WEB_OUT / f"{timestamp}_{Path(input_path).stem}")

    progress(0, desc="Reading uploaded audio")
    raw_audio, sample_rate = read_audio(input_path, sample_rate=16000, mono=True)
    comparison_items = [("Raw noisy", raw_audio)]
    errors: list[str] = []
    completed: list[str] = []
    output_paths = {method: None for method in LABELS}
    selected_methods = [method for method in LABELS if method in selected_methods]
    total_steps = max(len(selected_methods) + 2, 1)
    current_step = 1

    try:
        progress(current_step / total_steps, desc="Running Spectral Gate")
        denoised = spectral_gate_denoise(raw_audio, sample_rate=sample_rate)
        output_paths["baseline"] = save_result_audio(job_dir, "baseline", denoised, sample_rate)
        comparison_items.append((LABELS["baseline"], denoised))
        completed.append(LABELS["baseline"])
    except Exception as exc:
        errors.append(f"baseline failed: {exc}")
    current_step += 1

    for method_name in ["unet", "tcn", "cnn", "dit"]:
        if method_name not in selected_methods:
            continue
        try:
            progress(current_step / total_steps, desc=f"Running {LABELS[method_name]}")
            model, config = get_trained_model(method_name)
            denoised = TRAINED[method_name]["runner"](model, raw_audio, config, DEVICE)
            output_paths[method_name] = save_result_audio(job_dir, method_name, denoised, sample_rate)
            comparison_items.append((LABELS[method_name], denoised))
            completed.append(LABELS[method_name])
        except Exception as exc:
            errors.append(f"{method_name} failed: {exc}")
        current_step += 1

    if "demucs" in selected_methods:
        try:
            progress(current_step / total_steps, desc="Running Demucs")
            demucs_output = job_dir / "demucs.wav"
            demucs_work_dir = job_dir / "demucs_work"
            run_demucs_isolated(
                Path(input_path),
                demucs_work_dir,
                device=("cuda" if DEVICE.type == "cuda" else "cpu"),
                segment=4.0,
            )
            track_dir = demucs_work_dir / "htdemucs" / Path(input_path).stem
            combine_stems(track_dir, demucs_output, sample_rate=sample_rate)
            demucs_audio, _ = read_audio(demucs_output, sample_rate=sample_rate, mono=True)
            output_paths["demucs"] = str(demucs_output)
            comparison_items.append((LABELS["demucs"], demucs_audio))
            completed.append(LABELS["demucs"])
        except Exception as exc:
            errors.append(f"demucs failed: {exc}")
        current_step += 1

    if "denoiser" in selected_methods:
        try:
            progress(current_step / total_steps, desc="Running Denoiser")
            model = get_denoiser_model()
            denoised = run_denoiser(
                model,
                raw_audio,
                DEVICE,
                chunk_seconds=10.0,
                sample_rate=sample_rate,
            )
            output_paths["denoiser"] = save_result_audio(job_dir, "denoiser", denoised, sample_rate)
            comparison_items.append((LABELS["denoiser"], denoised))
            completed.append(LABELS["denoiser"])
        except Exception as exc:
            errors.append(f"denoiser failed: {exc}")
        current_step += 1

    if len(comparison_items) == 1:
        raise gr.Error("All selected methods failed. Check dependencies and checkpoints.")

    progress(current_step / total_steps, desc="Generating comparison figures")
    waveform_path = job_dir / "waveform_comparison.png"
    spectrogram_path = job_dir / "spectrogram_comparison.png"
    plot_multi_waveform_comparison(comparison_items, sample_rate, waveform_path)
    plot_multi_spectrogram_comparison(comparison_items, sample_rate, spectrogram_path)

    progress(1, desc="Done")
    elapsed = time.perf_counter() - started_at
    status = build_status(job_dir, errors)
    if completed:
        status += "\n\nCompleted: " + ", ".join(completed)
    status += f"\nElapsed: {elapsed:.1f}s"

    return (
        status,
        output_paths["baseline"],
        output_paths["unet"],
        output_paths["tcn"],
        output_paths["cnn"],
        output_paths["dit"],
        output_paths["demucs"],
        output_paths["denoiser"],
        str(waveform_path),
        str(spectrogram_path),
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Music Denoising Demo") as demo:
        gr.Markdown(
            """
<div id="hero">
  <h1>Music Denoising Demo</h1>
  <p>Upload one noisy music clip and compare the baseline and all trained baselines in one page.</p>
</div>
"""
        )

        with gr.Row():
            with gr.Column(scale=5):
                audio_input = gr.Audio(label="Upload audio", type="filepath")
            with gr.Column(scale=4):
                method_input = gr.CheckboxGroup(
                    choices=DEFAULT_METHODS + OPTIONAL_METHODS,
                    value=["unet", "tcn", "cnn", "dit"],
                    label="Methods to run",
                    info="Spectral Gate baseline is always included. Demucs and Denoiser are slower optional comparisons.",
                )
                run_button = gr.Button("Run Denoising", variant="primary")

        status_output = gr.Markdown(elem_classes=["metric-note"])
        metrics_output = gr.Dataframe(
            headers=["Method", "Type", "Mean SNR (dB)", "Mean SNRi (dB)"],
            value=load_metrics_table(),
            interactive=False,
            label="Existing synthetic test-set summary",
        )
        _ = metrics_output

        with gr.Row():
            baseline_output = gr.Audio(label="Spectral Gate")
            unet_output = gr.Audio(label="U-Net")
            tcn_output = gr.Audio(label="TCN")

        with gr.Row():
            cnn_output = gr.Audio(label="Residual CNN")
            dit_output = gr.Audio(label="Tiny DiT-style")
            demucs_output = gr.Audio(label="Demucs")

        denoiser_output = gr.Audio(label="Denoiser")

        with gr.Row():
            waveform_output = gr.Image(label="Waveform comparison")
            spectrogram_output = gr.Image(label="Spectrogram comparison")

        run_event = run_button.click(
            fn=set_running_state,
            inputs=None,
            outputs=[status_output, run_button],
            queue=False,
        ).then(
            fn=process_audio,
            inputs=[audio_input, method_input],
            outputs=[
                status_output,
                baseline_output,
                unet_output,
                tcn_output,
                cnn_output,
                dit_output,
                demucs_output,
                denoiser_output,
                waveform_output,
                spectrogram_output,
            ],
            concurrency_limit=1,
        )
        run_event.then(
            fn=set_idle_state,
            inputs=None,
            outputs=run_button,
            queue=False,
        )

    return demo


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    preferred_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    server_port = find_available_port(preferred_port)
    _, local_url, _ = build_interface().queue().launch(
        server_name="localhost",
        server_port=server_port,
        inbrowser=False,
        prevent_thread_lock=True,
        css=CUSTOM_CSS,
    )
    webbrowser.open(local_url)
    print(f"Demo is running at: {local_url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
