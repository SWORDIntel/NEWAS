# NEMWAS Environment Setup Guide

## Overview

NEMWAS now includes intelligent environment detection that can automatically discover and utilize advanced data science environments on your system. This allows you to leverage existing GPU/CUDA setups, conda environments, and pre-installed scientific computing libraries.

## Environment Detection

The quickstart script automatically checks for an advanced data science environment at `~/datascience`. When detected, you'll be prompted to choose between:

1. **Advanced Environment** - Recommended if you have GPU/CUDA support
2. **Local Environment** - Lightweight, project-specific virtual environment
3. **Auto-detect** - Automatically selects based on hardware capabilities

## Usage Options

### Interactive Setup (Default)
```bash
./quickstart.sh
```
This will detect environments and prompt you for your choice.

### Command Line Options
```bash
# Automatically use advanced environment if available
./quickstart.sh --use-advanced

# Force use of local virtual environment
./quickstart.sh --use-local

# Skip all prompts and auto-detect
./quickstart.sh --auto
```

### Using Make Commands
```bash
# Run quickstart with environment detection
make quickstart

# Force advanced environment
make quickstart-advanced

# Force local environment
make quickstart-local

# Just check what environments are available
make check-env
```

## Advanced Environment Benefits

When using the advanced environment, you get:

- **GPU Acceleration**: Automatic CUDA configuration if available
- **Extended Model Support**: Access to larger models that require more memory
- **Optimized Dependencies**: Reuse of existing scientific computing libraries
- **Memory Optimizations**: Better handling of large datasets

## Environment Variables

The script sets these environment variables when using advanced environment:

- `CUDA_VISIBLE_DEVICES=0` - GPU device selection
- `TF_FORCE_GPU_ALLOW_GROWTH=true` - Dynamic GPU memory allocation
- `OMP_NUM_THREADS=4` - OpenMP thread optimization
- `MKL_NUM_THREADS=4` - Intel MKL thread optimization

## Persistence

Your environment choice is saved to `.env.quickstart` for future runs. To change your selection, either:
- Delete `.env.quickstart` and run quickstart again
- Use the command line options to override

## Troubleshooting

### Advanced Environment Not Detected
If you have a data science environment but it's not being detected:

1. Check that it's located at `~/datascience`
2. Ensure it has a virtual environment directory (venv, env, or .venv)
3. Run `make check-env` to see what's being detected

### GPU Not Recognized
If you have a GPU but it's not being used:

1. Check CUDA installation: `nvidia-smi`
2. Verify CUDA_HOME is set: `echo $CUDA_HOME`
3. Ensure the advanced environment has GPU libraries installed

### Dependency Conflicts
If you encounter dependency conflicts when using the advanced environment:

1. Consider using the local environment instead
2. Create a dedicated conda environment for NEMWAS
3. Use Docker for complete isolation

## Best Practices

1. **Development**: Use local environment for development and testing
2. **Production**: Use advanced environment for production workloads with GPU
3. **Benchmarking**: Always use the same environment for consistent results
4. **Team Work**: Document which environment your team should use

## Integration with CI/CD

For automated deployments:

```bash
# CI/CD script example
if [ -n "$GPU_AVAILABLE" ]; then
    ./quickstart.sh --use-advanced
else
    ./quickstart.sh --use-local
fi
```

## Docker Alternative

If environment detection doesn't work for your setup, consider using Docker:

```bash
make docker-build
make docker-up
```

This provides a consistent environment regardless of your system configuration.