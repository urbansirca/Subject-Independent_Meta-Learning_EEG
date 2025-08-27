#!/usr/bin/env python
# coding: utf-8
"""
Test script to verify CPU compatibility of the motor_braindecode codebase.
This script tests the device utilities and basic model creation without requiring CUDA.
"""

import torch
import logging
import sys
import os

# Add the current directory to the path so we can import motor_braindecode
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motor_braindecode.torch_ext.device_utils import get_device, set_random_seeds_safe, set_cuda_device_safely
from motor_braindecode.models.deep4 import Deep5Net

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_device_utils():
    """Test the device utility functions."""
    print("Testing device utilities...")
    
    # Test device selection
    device = get_device(0)
    print(f"Selected device: {device}")
    
    # Test random seed setting
    set_random_seeds_safe(seed=42, gpu_id=0)
    print("Random seeds set successfully")
    
    # Test CUDA device setting
    set_cuda_device_safely(0)
    print("CUDA device setting completed")
    
    return True

def test_model_creation():
    """Test creating a model and moving it to the selected device."""
    print("Testing model creation...")
    
    try:
        # Create a simple model
        model = Deep5Net(in_chans=62, n_classes=2,
                        input_time_length=1000,
                        final_conv_length='auto')
        
        # Get device
        device = get_device(0)
        
        # Move model to device
        model = model.to(device)
        
        print(f"Model created and moved to {device} successfully")
        print(f"Model device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

def test_tensor_operations():
    """Test basic tensor operations on the selected device."""
    print("Testing tensor operations...")
    
    try:
        device = get_device(0)
        
        # Create a simple tensor
        x = torch.randn(10, 62, 1000)
        x = x.to(device)
        
        print(f"Tensor created on {device} successfully")
        print(f"Tensor device: {x.device}")
        
        # Test basic operations
        y = x.mean(dim=1)
        print(f"Mean operation completed, result device: {y.device}")
        
        return True
        
    except Exception as e:
        print(f"Error with tensor operations: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing CPU compatibility of motor_braindecode")
    print("=" * 50)
    
    tests = [
        ("Device Utilities", test_device_utils),
        ("Model Creation", test_model_creation),
        ("Tensor Operations", test_tensor_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name} passed" if result else f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The codebase is CPU compatible.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
