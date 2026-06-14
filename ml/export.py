"""Model export pipeline: PyTorch -> ONNX -> TensorFlow Lite.

Clinical purpose:
    The whole value proposition is on-body, always-on inference: the chip must
    score the patient continuously at micro-watt power, with no cloud round-trip.
    That requires shrinking a research-grade PyTorch model into a quantised
    TFLite-Micro artifact small enough for a sub-1MB microcontroller.

Technical purpose:
    Demonstrates the deployment pathway end to end. The trained PyTorch model is
    (1) exported to ONNX (open, framework-agnostic), (2) converted to TFLite
    (microcontroller runtime), (3) INT8-quantised (~4x size reduction). The TFLite
    steps are provided as runnable stubs because they require a separate
    TensorFlow install not needed for training.

Target hardware: Nordic nRF9161 SiP running TFLite Micro.
Power budget: <50uW for continuous inference.

Usage:
    python ml/export.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- path bootstrap ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

import constants as C
from ml.baseline import LSTMAutoencoder
from ml.predictor import InfectionRiskTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.export")


class _PredictorExportWrapper(nn.Module):
    """Wraps the predictor so ONNX sees tensor (not dict) outputs."""

    def __init__(self, model: InfectionRiskTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return torch.sigmoid(out["risk_logit"]), out["severity"], out["time_to_event"]


def _try_load(model: nn.Module, path: Path, name: str) -> nn.Module:
    """Load weights if present; otherwise export the randomly-initialised graph."""
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info("Loaded trained weights for %s from %s", name, path)
    else:
        logger.warning("No trained weights for %s at %s — exporting graph only", name, path)
    model.eval()
    return model


def export_baseline_onnx() -> Path:
    """Export the LSTM autoencoder to ONNX with a 2-hour dummy window."""
    C.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = _try_load(
        LSTMAutoencoder(),
        C.MODELS_DIR / next(iter(C.PATIENT_ARCHETYPES)) / "baseline.pt",
        "baseline-autoencoder",
    )
    dummy = torch.randn(1, C.BASELINE_WINDOW_MINUTES, C.N_SENSORS)
    out_path = C.EXPORT_DIR / "baseline_autoencoder.onnx"
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["sensor_window"],
        output_names=["reconstruction"],
        dynamic_axes={"sensor_window": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported baseline -> %s", out_path)
    return out_path


def export_predictor_onnx() -> Path:
    """Export the Transformer predictor to ONNX with a 6-hour dummy window."""
    C.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = _try_load(InfectionRiskTransformer(), C.MODELS_DIR / "predictor.pt", "predictor-transformer")
    wrapper = _PredictorExportWrapper(model).eval()
    dummy = torch.randn(1, C.PREDICTOR_WINDOW_MINUTES, C.N_SENSORS)
    out_path = C.EXPORT_DIR / "predictor_transformer.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        out_path.as_posix(),
        input_names=["sensor_window"],
        output_names=["risk_score", "severity", "time_to_event"],
        dynamic_axes={"sensor_window": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported predictor -> %s", out_path)
    return out_path


def convert_onnx_to_tflite(onnx_path: Path) -> None:
    """Stub: ONNX -> TensorFlow SavedModel -> TFLite -> INT8 quantisation.

    This requires ``tensorflow`` and ``onnx-tf`` installed separately (they are
    not training dependencies). The full pipeline is shown so a reviewer can see
    exactly how the chip artifact would be produced.

    Clinical note:
        INT8 post-training quantisation typically cuts model size ~4x and inference
        energy substantially, which is what brings continuous scoring inside the
        <50uW power budget — at the cost of a small, validated accuracy drop that
        must be re-checked against the clinical safety thresholds before release.
    """
    logger.info("[stub] TFLite conversion for %s", onnx_path.name)
    tflite_path = C.EXPORT_DIR / f"{onnx_path.stem}_int8.tflite"
    snippet = f'''
    # Requires: pip install tensorflow onnx-tf
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # 1) ONNX -> TensorFlow SavedModel
    onnx_model = onnx.load("{onnx_path.as_posix()}")
    saved_dir = "{(C.EXPORT_DIR / onnx_path.stem).as_posix()}_tf"
    prepare(onnx_model).export_graph(saved_dir)

    # 2) SavedModel -> TFLite with INT8 post-training quantisation
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # converter.representative_dataset = <100 normal windows for calibration>
    tflite_bytes = converter.convert()
    open("{tflite_path.as_posix()}", "wb").write(tflite_bytes)
    '''
    logger.info("TFLite conversion recipe:\n%s", snippet)
    logger.info("[stub] would write quantised model -> %s", tflite_path)


def _report_sizes(onnx_paths: list[Path]) -> None:
    """Print estimated artifact sizes at each export stage."""
    logger.info("--- estimated model sizes ---")
    for p in onnx_paths:
        if p.exists():
            fp32_kb = p.stat().st_size / 1024
            logger.info(
                "%-32s ONNX(fp32)=%6.1f KB | est. TFLite(int8)~=%6.1f KB",
                p.name, fp32_kb, fp32_kb / 4,
            )


def run_export() -> None:
    """Export both models and emit the TFLite conversion recipe."""
    baseline = export_baseline_onnx()
    predictor = export_predictor_onnx()
    for path in (baseline, predictor):
        convert_onnx_to_tflite(path)
    _report_sizes([baseline, predictor])
    logger.info("Export artifacts in %s", C.EXPORT_DIR)


if __name__ == "__main__":
    run_export()
