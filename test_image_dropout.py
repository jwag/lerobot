#!/usr/bin/env python3
"""
Test script for Image Dropout augmentation feature.

This script verifies that the ImageDropout transform is working correctly
and integrates properly with the LeRobot training pipeline.

Usage:
    python test_image_dropout.py
    
Or with your dataset:
    python test_image_dropout.py --dataset your/dataset-repo-id
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add libs/lerobot to path if running from workspace root
sys.path.insert(0, str(Path(__file__).parent / "libs" / "lerobot" / "src"))

from lerobot.datasets.transforms import ImageDropout, ImageTransforms, ImageTransformsConfig


def test_1_basic_dropout():
    """Test 1: Basic ImageDropout functionality"""
    print("\n" + "="*70)
    print("TEST 1: Basic ImageDropout Functionality")
    print("="*70)
    
    # Create dropout with 80% probability for easy testing
    dropout = ImageDropout(dropout_prob=0.8)
    
    # Create non-zero test image
    test_image = torch.ones(3, 256, 256) * 127.0
    print(f"✓ Created test image shape: {test_image.shape}")
    print(f"✓ Test image sum (non-zero): {test_image.sum().item()}")
    
    # Apply dropout multiple times and count drops
    num_trials = 100
    num_dropped = 0
    
    for _ in range(num_trials):
        result = dropout(test_image)
        if result.sum() == 0:  # Image was dropped
            num_dropped += 1
    
    drop_rate = num_dropped / num_trials
    print(f"✓ Dropped {num_dropped}/{num_trials} images (rate: {drop_rate:.2%})")
    print(f"  Expected ~80%, Got {drop_rate:.2%}")
    
    # Verify drop rate is reasonable (within 3 sigma for binomial)
    # For p=0.8, n=100: mean=80, std=4, so 3*sigma = 12
    assert 68 < num_dropped < 92, f"Drop rate outside expected range: {drop_rate:.2%}"
    print("✓ TEST 1 PASSED: Dropout rate is within expected range\n")
    
    return True


def test_2_zero_dropout():
    """Test 2: Zero dropout probability (should never drop)"""
    print("="*70)
    print("TEST 2: Zero Dropout Probability")
    print("="*70)
    
    dropout = ImageDropout(dropout_prob=0.0)
    test_image = torch.ones(3, 256, 256) * 127.0
    
    num_trials = 100
    num_dropped = 0
    
    for _ in range(num_trials):
        result = dropout(test_image)
        if result.sum() == 0:
            num_dropped += 1
    
    print(f"✓ Applied dropout with prob=0.0 {num_trials} times")
    print(f"✓ Number of drops: {num_dropped}/{num_trials}")
    
    assert num_dropped == 0, "Images should never be dropped with prob=0.0"
    print("✓ TEST 2 PASSED: Zero probability works correctly\n")
    
    return True


def test_3_validation():
    """Test 3: Input validation"""
    print("="*70)
    print("TEST 3: Input Validation")
    print("="*70)
    
    # Test valid probabilities
    try:
        ImageDropout(dropout_prob=0.0)
        ImageDropout(dropout_prob=0.5)
        ImageDropout(dropout_prob=1.0)
        print("✓ Valid probabilities (0.0, 0.5, 1.0) accepted")
    except Exception as e:
        print(f"✗ FAILED: Valid probabilities rejected: {e}")
        return False
    
    # Test invalid probabilities
    invalid_probs = [-0.1, 1.5, "0.5"]
    for prob in invalid_probs:
        try:
            ImageDropout(dropout_prob=prob)
            print(f"✗ FAILED: Invalid probability {prob} was accepted")
            return False
        except (ValueError, TypeError):
            print(f"✓ Invalid probability {prob} correctly rejected")
    
    print("✓ TEST 3 PASSED: Input validation works correctly\n")
    return True


def test_4_integration_with_transforms():
    """Test 4: Integration with ImageTransforms pipeline"""
    print("="*70)
    print("TEST 4: Integration with ImageTransforms")
    print("="*70)
    
    # Create config with dropout enabled
    config = ImageTransformsConfig(
        enable=True,
        image_dropout_enable=True,
        image_dropout_prob=0.5,
        max_num_transforms=0,  # Disable other transforms for clear testing
    )
    
    transforms = ImageTransforms(config)
    print("✓ Created ImageTransforms with dropout enabled")
    
    # Test application
    test_image = torch.ones(3, 256, 256) * 127.0
    num_trials = 100
    num_dropped = 0
    
    for _ in range(num_trials):
        result = transforms(test_image)
        if result.sum() == 0:
            num_dropped += 1
    
    drop_rate = num_dropped / num_trials
    print(f"✓ Applied transforms {num_trials} times")
    print(f"✓ Dropped {num_dropped}/{num_trials} images (rate: {drop_rate:.2%})")
    print(f"  Expected ~50%, Got {drop_rate:.2%}")
    
    # Verify integration works (within 3 sigma)
    assert 35 < num_dropped < 65, f"Integration test failed: {drop_rate:.2%}"
    print("✓ TEST 4 PASSED: Integration with ImageTransforms works\n")
    
    return True


def test_5_disabled_dropout():
    """Test 5: Dropout disabled by default"""
    print("="*70)
    print("TEST 5: Dropout Disabled by Default")
    print("="*70)
    
    # Default config should have dropout disabled
    config = ImageTransformsConfig(enable=True)
    transforms = ImageTransforms(config)
    
    print(f"✓ Default config: image_dropout_enable={config.image_dropout_enable}")
    assert config.image_dropout_enable == False, "Dropout should be disabled by default"
    
    # Test that no dropout occurs
    test_image = torch.ones(3, 256, 256) * 127.0
    num_trials = 100
    num_dropped = 0
    
    for _ in range(num_trials):
        result = transforms(test_image)
        if result.sum() == 0:
            num_dropped += 1
    
    print(f"✓ Applied transforms with default config {num_trials} times")
    print(f"✓ Number of drops: {num_dropped}/{num_trials}")
    
    assert num_dropped == 0, "No drops should occur when dropout is disabled"
    print("✓ TEST 5 PASSED: Dropout is disabled by default\n")
    
    return True


def test_6_batch_processing():
    """Test 6: Batch processing with multiple images"""
    print("="*70)
    print("TEST 6: Batch Processing")
    print("="*70)
    
    dropout = ImageDropout(dropout_prob=0.3)
    
    # Test with batched images [B, C, H, W]
    batch_size = 10
    test_batch = torch.ones(batch_size, 3, 64, 64) * 127.0
    print(f"✓ Created batch of {batch_size} images")
    
    num_trials = 50
    drop_counts = []
    
    for _ in range(num_trials):
        # Process each image in batch independently (as happens in dataset loader)
        num_dropped = 0
        for i in range(batch_size):
            result = dropout(test_batch[i])
            if result.sum() == 0:
                num_dropped += 1
        drop_counts.append(num_dropped)
    
    mean_drops = np.mean(drop_counts)
    std_drops = np.std(drop_counts)
    expected_drops = batch_size * 0.3
    
    print(f"✓ Processed {num_trials} batches of {batch_size} images")
    print(f"✓ Average drops per batch: {mean_drops:.1f} ± {std_drops:.1f}")
    print(f"  Expected: {expected_drops:.1f}")
    
    # Should be within reasonable range
    assert abs(mean_drops - expected_drops) < 1.5, "Batch drop rate outside expected range"
    print("✓ TEST 6 PASSED: Batch processing works correctly\n")
    
    return True


def test_7_different_image_sizes():
    """Test 7: Different image sizes and formats"""
    print("="*70)
    print("TEST 7: Different Image Sizes and Formats")
    print("="*70)
    
    dropout = ImageDropout(dropout_prob=1.0)  # Always drop for testing
    
    test_cases = [
        (3, 64, 64),      # Small image
        (3, 256, 256),    # Medium image
        (3, 512, 512),    # Large image
        (1, 128, 128),    # Grayscale
        (4, 224, 224),    # 4-channel (e.g., RGBA)
    ]
    
    for shape in test_cases:
        test_image = torch.ones(*shape) * 127.0
        result = dropout(test_image)
        
        print(f"✓ Shape {shape}: ", end="")
        assert result.shape == test_image.shape, f"Shape mismatch: {result.shape} != {shape}"
        assert result.sum() == 0, f"Image not zeroed for shape {shape}"
        print("correctly zeroed")
    
    print("✓ TEST 7 PASSED: Works with different image sizes\n")
    return True


def test_8_dataset_integration(dataset_repo_id=None):
    """Test 8: Integration with actual dataset (if available)"""
    print("="*70)
    print("TEST 8: Dataset Integration (Optional)")
    print("="*70)
    
    if dataset_repo_id is None:
        print("⊘ SKIPPED: No dataset specified")
        print("  To test with your dataset, run:")
        print("  python test_image_dropout.py --dataset your/dataset-id\n")
        return True
    
    try:
        from lerobot.datasets import LeRobotDataset
        
        print(f"✓ Loading dataset: {dataset_repo_id}")
        
        # Test with dropout enabled
        config = ImageTransformsConfig(
            enable=True,
            image_dropout_enable=True,
            image_dropout_prob=0.5,
            max_num_transforms=0,
        )
        
        dataset = LeRobotDataset(
            repo_id=dataset_repo_id,
            image_transforms=ImageTransforms(config)
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"✓ Camera keys: {dataset.meta.camera_keys}")
        
        # Sample a few items and check for dropouts
        num_samples = min(20, len(dataset))
        num_dropped_per_camera = {cam: 0 for cam in dataset.meta.camera_keys}
        
        for i in range(num_samples):
            sample = dataset[i]
            for cam in dataset.meta.camera_keys:
                if cam in sample:
                    img = sample[cam]
                    if img.sum() == 0:
                        num_dropped_per_camera[cam] += 1
        
        print(f"✓ Sampled {num_samples} items from dataset")
        for cam, count in num_dropped_per_camera.items():
            rate = count / num_samples
            print(f"  {cam}: {count}/{num_samples} dropped (rate: {rate:.2%})")
        
        # At least some images should have been dropped
        total_dropped = sum(num_dropped_per_camera.values())
        assert total_dropped > 0, "No images were dropped - dropout may not be applying"
        
        print("✓ TEST 8 PASSED: Dataset integration works\n")
        return True
        
    except ImportError as e:
        print(f"⊘ SKIPPED: Could not import dataset: {e}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: Dataset integration error: {e}\n")
        return False


def run_all_tests(dataset_repo_id=None):
    """Run all tests and report results"""
    print("\n" + "="*70)
    print(" IMAGE DROPOUT TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Dropout", test_1_basic_dropout),
        ("Zero Dropout", test_2_zero_dropout),
        ("Input Validation", test_3_validation),
        ("ImageTransforms Integration", test_4_integration_with_transforms),
        ("Disabled by Default", test_5_disabled_dropout),
        ("Batch Processing", test_6_batch_processing),
        ("Different Image Sizes", test_7_different_image_sizes),
        ("Dataset Integration", lambda: test_8_dataset_integration(dataset_repo_id)),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ TEST FAILED with exception: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    num_passed = sum(1 for _, passed in results if passed)
    num_total = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("="*70)
    print(f"Results: {num_passed}/{num_total} tests passed")
    print("="*70 + "\n")
    
    if num_passed == num_total:
        print("🎉 All tests passed! Image dropout is working correctly.")
        return 0
    else:
        print(f"⚠️  {num_total - num_passed} test(s) failed. Please review the output above.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test Image Dropout augmentation feature"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional: HuggingFace dataset repo ID to test integration (e.g., 'your/dataset-id')"
    )
    
    args = parser.parse_args()
    
    exit_code = run_all_tests(args.dataset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
