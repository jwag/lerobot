#!/usr/bin/env python3
"""Test script for per-camera image dropout functionality.

This tests the new per-camera dropout rates feature to ensure:
1. Uniform dropout (float) still works as before (backward compatibility)
2. Per-camera dropout (dict) works with different rates per camera
3. Dataset loaders correctly pass camera_name parameter
"""

import sys
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "libs" / "lerobot" / "src"))

import torch
from lerobot.datasets.transforms import ImageDropout, ImageTransforms, ImageTransformsConfig


def test_uniform_dropout():
    """Test backward compatibility with uniform float dropout probability."""
    print("\n=== Test 1: Uniform Dropout (Backward Compatibility) ===")
    
    # Create dropout with uniform probability
    dropout = ImageDropout(dropout_probs=0.5)
    
    # Create dummy image tensor
    image = torch.randn(3, 224, 224)
    
    # Test without camera_name (should work for uniform)
    dropped_count = 0
    for _ in range(100):
        result = dropout(image)
        if torch.all(result == 0):
            dropped_count += 1
    
    print(f"  Dropout probability: 0.5")
    print(f"  Dropped {dropped_count}/100 images")
    print(f"  Expected: ~50, Actual: {dropped_count}")
    print(f"  ✓ PASS" if 35 <= dropped_count <= 65 else "  ✗ FAIL")
    
    return 35 <= dropped_count <= 65


def test_per_camera_dropout():
    """Test per-camera dropout with different probabilities."""
    print("\n=== Test 2: Per-Camera Dropout Rates ===")
    
    # Create dropout with per-camera probabilities
    dropout_probs = {
        "image": 0.1,   # 10% dropout
        "image2": 0.5,  # 50% dropout
        "image3": 0.9,  # 90% dropout
    }
    dropout = ImageDropout(dropout_probs=dropout_probs)
    
    # Create dummy image tensor
    image = torch.randn(3, 224, 224)
    
    # Test each camera
    results = {}
    for camera_name, expected_prob in dropout_probs.items():
        dropped_count = 0
        for _ in range(100):
            result = dropout(image, camera_name=camera_name)
            if torch.all(result == 0):
                dropped_count += 1
        
        results[camera_name] = dropped_count
        print(f"  Camera '{camera_name}': dropped {dropped_count}/100 (expected ~{int(expected_prob * 100)})")
    
    # Check if results are within reasonable bounds
    checks = [
        (0 <= results["image"] <= 25, "image"),    # 10% ± 15%
        (35 <= results["image2"] <= 65, "image2"), # 50% ± 15%
        (results["image3"] >= 75, "image3"),       # 90% - 15%
    ]
    
    all_passed = all(check[0] for check in checks)
    print(f"  {'✓ PASS' if all_passed else '✗ FAIL'}")
    
    return all_passed


def test_camera_not_in_config():
    """Test behavior when camera is not in dropout config (should use 0.0)."""
    print("\n=== Test 3: Camera Not in Config ===")
    
    # Create dropout with only some cameras configured
    dropout = ImageDropout(dropout_probs={"image": 0.5})
    
    # Create dummy image tensor
    image = torch.randn(3, 224, 224)
    
    # Test with unconfigured camera (should never drop)
    dropped_count = 0
    for _ in range(100):
        result = dropout(image, camera_name="unknown_camera")
        if torch.all(result == 0):
            dropped_count += 1
    
    print(f"  Camera 'unknown_camera' not in config")
    print(f"  Dropped {dropped_count}/100 images (expected 0)")
    print(f"  ✓ PASS" if dropped_count == 0 else "  ✗ FAIL")
    
    return dropped_count == 0


def test_image_transforms_integration():
    """Test ImageTransforms properly passes camera_name to dropout."""
    print("\n=== Test 4: ImageTransforms Integration ===")
    
    # Create config with per-camera dropout
    config = ImageTransformsConfig(
        enable=True,
        image_dropout_enable=True,
        image_dropout_prob={"image": 0.1, "image2": 0.9},
        max_num_transforms=0,  # Disable other transforms for cleaner test
        tfs={}
    )
    
    transforms = ImageTransforms(config)
    
    # Create dummy image
    image = torch.randn(3, 224, 224)
    
    # Test with low dropout camera
    low_dropped = 0
    for _ in range(100):
        result = transforms(image, camera_name="image")
        if torch.all(result == 0):
            low_dropped += 1
    
    # Test with high dropout camera
    high_dropped = 0
    for _ in range(100):
        result = transforms(image, camera_name="image2")
        if torch.all(result == 0):
            high_dropped += 1
    
    print(f"  Camera 'image' (10%): dropped {low_dropped}/100")
    print(f"  Camera 'image2' (90%): dropped {high_dropped}/100")
    
    passed = (low_dropped <= 25) and (high_dropped >= 75)
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed


def test_validation():
    """Test input validation."""
    print("\n=== Test 5: Input Validation ===")
    
    tests_passed = 0
    
    # Test invalid probability value
    try:
        ImageDropout(dropout_probs=1.5)
        print("  ✗ FAIL: Should reject prob > 1.0")
    except ValueError:
        print("  ✓ PASS: Rejects prob > 1.0")
        tests_passed += 1
    
    # Test invalid type
    try:
        ImageDropout(dropout_probs="invalid")
        print("  ✗ FAIL: Should reject string type")
    except TypeError:
        print("  ✓ PASS: Rejects invalid type")
        tests_passed += 1
    
    # Test invalid dict value
    try:
        ImageDropout(dropout_probs={"cam1": -0.1})
        print("  ✗ FAIL: Should reject negative prob")
    except ValueError:
        print("  ✓ PASS: Rejects negative prob")
        tests_passed += 1
    
    # Test missing camera_name for per-camera config
    try:
        dropout = ImageDropout(dropout_probs={"cam1": 0.5})
        image = torch.randn(3, 224, 224)
        dropout(image)  # No camera_name provided
        print("  ✗ FAIL: Should require camera_name for per-camera config")
    except ValueError:
        print("  ✓ PASS: Requires camera_name for per-camera config")
        tests_passed += 1
    
    return tests_passed == 4


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Per-Camera Image Dropout Implementation")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Uniform Dropout", test_uniform_dropout()))
    results.append(("Per-Camera Dropout", test_per_camera_dropout()))
    results.append(("Unknown Camera", test_camera_not_in_config()))
    results.append(("ImageTransforms Integration", test_image_transforms_integration()))
    results.append(("Input Validation", test_validation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("=" * 60)
    
    return total_passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
