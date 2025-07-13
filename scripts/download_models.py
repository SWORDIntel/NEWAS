#!/usr/bin/env python3
"""Download and prepare models for NEMWAS"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.npu_manager import NPUManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model registry
MODEL_REGISTRY = {
    "tinyllama-1.1b": {
        "name": "TinyLlama-1.1B-Chat",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "669M",
        "format": "gguf",
        "description": "Lightweight LLM ideal for NPU deployment",
        "sha256": "c1dabb95c18c4614ddc4f911df48546f6e0dac1a6e5d41ac07fb0550ff9fd768"
    },
    "mistral-7b": {
        "name": "Mistral-7B-Instruct",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": "4.1G",
        "format": "gguf",
        "description": "Powerful 7B parameter model",
        "sha256": "3a7b64dc96c91a2cf38c18c5427cd23b42e2f476bc85c012364565359f37cc67"
    },
    "codebert-base": {
        "name": "CodeBERT-base",
        "url": "https://huggingface.co/microsoft/codebert-base/resolve/main/pytorch_model.bin",
        "size": "420M",
        "format": "pytorch",
        "description": "Code understanding model",
        "sha256": "a2c37e3c7fef0cf91b5c38d8c7c826ce6cf7d5fa9ad7e1d5b9e75d44a5f8b893"
    },
    "all-minilm-l6": {
        "name": "all-MiniLM-L6-v2",
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin",
        "size": "23M",
        "format": "pytorch",
        "description": "Lightweight sentence embeddings",
        "sha256": "c2d8a4e65fb25558f8e27b1dc12d87c87b8d37cb8c4e9e3ef5d5e27bebe5c3f5"
    }
}


def download_file(url: str, filepath: Path, expected_hash: str = None) -> bool:
    """Download file with progress bar"""
    
    try:
        # Check if file already exists
        if filepath.exists():
            if expected_hash and verify_file_hash(filepath, expected_hash):
                logger.info(f"File already exists and verified: {filepath}")
                return True
            else:
                logger.warning(f"File exists but hash mismatch, re-downloading: {filepath}")
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify hash if provided
        if expected_hash:
            if not verify_file_hash(filepath, expected_hash):
                logger.error(f"Hash verification failed for {filepath}")
                filepath.unlink()
                return False
        
        logger.info(f"Successfully downloaded: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def verify_file_hash(filepath: Path, expected_hash: str) -> bool:
    """Verify file SHA256 hash"""
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_hash = sha256_hash.hexdigest()
    return actual_hash == expected_hash


def convert_to_openvino(model_path: Path, output_dir: Path, model_type: str) -> bool:
    """Convert model to OpenVINO format"""
    
    try:
        from openvino.tools import mo
        
        output_path = output_dir / f"{model_path.stem}.xml"
        
        if output_path.exists():
            logger.info(f"OpenVINO model already exists: {output_path}")
            return True
        
        logger.info(f"Converting {model_path} to OpenVINO format...")
        
        # Model-specific conversion parameters
        if model_type == "gguf":
            # GGUF models need special handling
            # For now, we'll need manual conversion
            logger.warning("GGUF conversion requires manual steps. Please use convert-gguf-to-onnx.py first.")
            return False
            
        elif model_type == "pytorch":
            # Convert PyTorch model
            mo_args = [
                "--input_model", str(model_path),
                "--output_dir", str(output_dir),
                "--model_name", model_path.stem,
                "--compress_to_fp16"
            ]
            
            mo.main(mo_args)
            
        logger.info(f"Successfully converted to OpenVINO: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        return False


def optimize_for_npu(model_path: Path, npu_manager: NPUManager) -> Path:
    """Optimize model for NPU deployment"""
    
    try:
        logger.info(f"Optimizing {model_path} for NPU...")
        
        optimized_path = npu_manager.optimize_model_for_npu(
            str(model_path),
            model_type="llm" if "llama" in str(model_path).lower() else "embedder",
            quantization_preset="mixed"
        )
        
        logger.info(f"Successfully optimized for NPU: {optimized_path}")
        return Path(optimized_path)
        
    except Exception as e:
        logger.error(f"Failed to optimize for NPU: {e}")
        return model_path


def download_models(models: List[str], output_dir: Path, optimize_npu: bool = False) -> Dict[str, Path]:
    """Download and prepare specified models"""
    
    downloaded = {}
    
    # Initialize NPU manager if optimization requested
    npu_manager = None
    if optimize_npu:
        try:
            npu_manager = NPUManager()
            if "NPU" not in npu_manager.available_devices:
                logger.warning("NPU not available, skipping NPU optimization")
                npu_manager = None
        except Exception as e:
            logger.warning(f"Could not initialize NPU manager: {e}")
    
    for model_key in models:
        if model_key not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_key}")
            continue
        
        model_info = MODEL_REGISTRY[model_key]
        logger.info(f"\nProcessing {model_info['name']}...")
        
        # Download model
        filename = Path(model_info['url']).name
        model_path = output_dir / "original" / filename
        
        if not download_file(model_info['url'], model_path, model_info.get('sha256')):
            continue
        
        # Convert to OpenVINO format
        if model_info['format'] in ['pytorch', 'onnx']:
            openvino_dir = output_dir / "openvino"
            if convert_to_openvino(model_path, openvino_dir, model_info['format']):
                model_path = openvino_dir / f"{model_path.stem}.xml"
        
        # Optimize for NPU if requested
        if optimize_npu and npu_manager and model_path.suffix == '.xml':
            optimized_path = optimize_for_npu(model_path, npu_manager)
            downloaded[model_key] = optimized_path
        else:
            downloaded[model_key] = model_path
    
    return downloaded


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Download models for NEMWAS")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default=["tinyllama-1.1b", "all-minilm-l6"],
        help="Models to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models"),
        help="Output directory for models"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Download minimal set for quick start"
    )
    parser.add_argument(
        "--optimize-npu",
        action="store_true",
        help="Optimize models for NPU deployment"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        print("\nAvailable models:")
        for key, info in MODEL_REGISTRY.items():
            print(f"  {key:15} - {info['name']:25} ({info['size']:>6}) - {info['description']}")
        return
    
    # Select models
    if args.minimal:
        models = ["tinyllama-1.1b", "all-minilm-l6"]
    elif "all" in args.models:
        models = list(MODEL_REGISTRY.keys())
    else:
        models = args.models
    
    # Download models
    logger.info(f"Downloading models: {models}")
    downloaded = download_models(models, args.output_dir, args.optimize_npu)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    for model_key, path in downloaded.items():
        print(f"  {MODEL_REGISTRY[model_key]['name']:30} -> {path}")
    
    if not downloaded:
        print("  No models successfully downloaded")
        sys.exit(1)
    
    print("\nModels are ready for use!")
    
    # Create model config
    config_path = args.output_dir / "model_config.yaml"
    with open(config_path, 'w') as f:
        import yaml
        model_config = {
            "models": {
                model_key: {
                    "name": MODEL_REGISTRY[model_key]['name'],
                    "path": str(path),
                    "format": MODEL_REGISTRY[model_key]['format']
                }
                for model_key, path in downloaded.items()
            }
        }
        yaml.dump(model_config, f)
    
    print(f"\nModel configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
