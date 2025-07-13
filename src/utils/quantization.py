"""Model quantization utilities for NPU optimization"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import nncf
from nncf import QuantizationPreset, TargetDevice
import openvino as ov

logger = logging.getLogger(__name__)

def quantize_model_for_npu(
    model_path: str,
    calibration_data: Any,
    output_path: str,
    preset: str = "mixed"
) -> str:
    """Quantize model for NPU deployment"""
    
    logger.info(f"Quantizing model {model_path} for NPU...")
    
    # Load model
    core = ov.Core()
    model = core.read_model(model_path)
    
    # Select quantization preset
    if preset == "performance":
        q_preset = QuantizationPreset.PERFORMANCE
    elif preset == "mixed":
        q_preset = QuantizationPreset.MIXED
    else:
        q_preset = QuantizationPreset.DEFAULT
    
    # Quantize model
    quantized_model = nncf.quantize(
        model,
        calibration_data,
        preset=q_preset,
        target_device=TargetDevice.NPU,
        model_type=nncf.ModelType.TRANSFORMER
    )
    
    # Save quantized model
    ov.save_model(quantized_model, output_path)
    logger.info(f"Quantized model saved to {output_path}")
    
    return output_path
