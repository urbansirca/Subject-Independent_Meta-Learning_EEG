# CPU Compatibility for motor_braindecode

This document describes the changes made to make the motor_braindecode codebase compatible with both CUDA and CPU environments.

## Overview

The original codebase was hardcoded to use CUDA, which would cause crashes when CUDA was not available. The changes implemented here add automatic device detection and fallback to CPU when CUDA is not available.

## Changes Made

### 1. New Device Utility Module

Created `motor_braindecode/torch_ext/device_utils.py` with the following functions:

- `get_device(gpu_id=0)`: Automatically selects the best available device (CUDA if available, otherwise CPU)
- `to_device(tensor_or_module, device)`: Moves tensors or modules to the specified device
- `set_cuda_device_safely(gpu_id=0)`: Safely sets CUDA device if available, otherwise does nothing
- `set_random_seeds_safe(seed, gpu_id=0)`: Sets random seeds safely for both CUDA and CPU

### 2. Updated Training Scripts

Modified the following training scripts to use device-aware logic:

- `train_motor_LOSO_all.py`
- `train_motor_LOSO.py` 
- `train_motor_adapt_all.py`
- `eval_motor_base.py`

**Key changes:**
- Replaced hardcoded `.cuda()` calls with `.to(device)`
- Added automatic device detection
- Updated random seed setting to handle both CUDA and CPU cases
- Modified model checkpoint loading to use device-aware map_location

### 3. Updated Core Model Classes

**BaseModel (`motor_braindecode/models/base.py`):**
- Added `.to(device)` method for device selection
- Modified `.cuda()` method to check CUDA availability
- Added device attribute tracking
- Updated tensor movement logic in predict_outs method

**Experiment (`motor_braindecode/experiments/experiment.py`):**
- Added `_move_to_device()` helper function
- Updated all CUDA tensor movements to use device-aware logic
- Modified setup_training to handle CUDA unavailability gracefully

**Modules (`motor_braindecode/torch_ext/modules.py`):**
- Updated AvgPool2dWithConv to handle device selection properly

**Utilities (`motor_braindecode/torch_ext/util.py`):**
- Modified `confirm_gpu_availability()` to not crash when CUDA is unavailable

## Usage

### Automatic Device Selection

The code now automatically selects the best available device:

```python
from motor_braindecode.torch_ext.device_utils import get_device

# Automatically selects CUDA if available, otherwise CPU
device = get_device(0)  # 0 is the GPU ID to use if CUDA is available

# Create and move model to device
model = Deep5Net(...).to(device)
```

### Training Scripts

All training scripts now work automatically with or without CUDA:

```bash
# With CUDA (will use GPU)
python train_motor_LOSO_all.py data.h5 output/ -gpu 0

# Without CUDA (will automatically use CPU)
python train_motor_LOSO_all.py data.h5 output/ -gpu 0
```

The `-gpu` argument is still accepted for compatibility but is ignored when CUDA is not available.

### Manual Device Control

You can also manually control device selection:

```python
import torch
from motor_braindecode.torch_ext.device_utils import get_device

# Force CPU usage
device = torch.device('cpu')

# Or let it auto-detect
device = get_device(0)

# Move model to device
model = model.to(device)
```

## Testing

Run the test script to verify CPU compatibility:

```bash
python test_cpu_compatibility.py
```

This script tests:
- Device utility functions
- Model creation and device movement
- Basic tensor operations on the selected device

## Benefits

1. **No More Crashes**: The codebase no longer crashes when CUDA is not available
2. **Automatic Fallback**: Automatically falls back to CPU when CUDA is unavailable
3. **Backward Compatibility**: All existing functionality is preserved
4. **Flexible Deployment**: Can now run on machines without CUDA support
5. **Development Friendly**: Easier to develop and test on machines without GPUs

## Performance Considerations

- **CPU Training**: Training on CPU will be significantly slower than GPU training
- **Memory Usage**: CPU training may use more memory due to different memory management
- **Batch Size**: You may need to reduce batch size when running on CPU

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Slow Training**: This is expected on CPU - consider using a smaller model or reducing data
3. **Import Errors**: Ensure all dependencies are installed for both CPU and GPU versions

### Debugging

Enable logging to see device selection:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

The device utilities will log which device is being used.

## Future Improvements

1. **Mixed Precision**: Add support for mixed precision training on both CPU and GPU
2. **Distributed Training**: Extend device utilities for multi-GPU and multi-node training
3. **Performance Monitoring**: Add device-specific performance metrics
4. **Automatic Optimization**: Automatically optimize hyperparameters based on available device

## Contributing

When adding new features to the codebase:

1. Always use the device utilities instead of hardcoded `.cuda()` calls
2. Test both with and without CUDA
3. Consider performance implications on both CPU and GPU
4. Update this documentation if adding new device-related functionality
