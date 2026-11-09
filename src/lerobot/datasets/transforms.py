#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


CAMERA_KEY_PREFIX = "observation.images."


def short_camera_name(name: str) -> str:
    """Strip the ``observation.images.`` prefix so cameras can be named either way in configs."""
    return name[len(CAMERA_KEY_PREFIX) :] if name.startswith(CAMERA_KEY_PREFIX) else name


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class ImageDropout(Transform):
    """Randomly drop (zero out) images during training to improve robustness.

    This transform is useful for improving policy robustness to camera failures and occlusions.
    When an image is dropped, it is replaced with a tensor of all zeros (black pixels), maintaining
    the original image dimensions.

    Supports both uniform dropout (single probability for all cameras) and per-camera dropout
    (different probability for each camera). This transform follows the standard PyTorch Transform
    API and is applied per-image by the dataset loader.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        dropout_probs: Dropout probability configuration. Can be:
            - float: Uniform probability applied to all cameras (e.g., 0.2 for 20% dropout)
            - dict[str, float]: Per-camera probabilities (e.g., {"image": 0.1, "image2": 0.25})
            All probabilities must be in [0.0, 1.0].

    Examples:
        >>> # Uniform 20% dropout for all cameras
        >>> dropout = ImageDropout(dropout_probs=0.2)
        >>> 
        >>> # Per-camera dropout rates
        >>> dropout = ImageDropout(dropout_probs={"image": 0.1, "image2": 0.25, "image3": 0.25})
    """

    def __init__(self, dropout_probs: dict[str, float] | float = 0.2) -> None:
        super().__init__()
        
        # Validate and store dropout probabilities
        if isinstance(dropout_probs, (int, float)):
            if not 0.0 <= dropout_probs <= 1.0:
                raise ValueError(f"dropout_probs must be in [0.0, 1.0], got {dropout_probs}")
            self.dropout_probs = float(dropout_probs)
            self.uniform = True
        elif isinstance(dropout_probs, dict):
            for cam_name, prob in dropout_probs.items():
                if not isinstance(prob, (int, float)):
                    raise TypeError(f"Probability for camera '{cam_name}' must be a number, got {type(prob)}")
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(f"Probability for camera '{cam_name}' must be in [0.0, 1.0], got {prob}")
            # Keys may be full feature keys ("observation.images.image") or short names ("image").
            self.dropout_probs = {short_camera_name(k): float(v) for k, v in dropout_probs.items()}
            self.uniform = False
        else:
            raise TypeError(f"dropout_probs must be a number or dict, got {type(dropout_probs)}")


    def forward(self, inpt: Any, camera_name: str | None = None) -> Any:
        """Apply dropout to the input.
        
        Args:
            inpt: Input tensor(s) following the torchvision Transform API
            camera_name: Name of the camera (required for per-camera dropout, optional for uniform)
            
        Returns:
            Input tensor(s), potentially zeroed out based on dropout probability
        """
        # Determine the dropout probability for this camera
        if self.uniform:
            dropout_prob = self.dropout_probs
        else:
            if camera_name is None:
                raise ValueError("camera_name is required when using per-camera dropout probabilities")
            dropout_prob = self.dropout_probs.get(short_camera_name(camera_name), 0.0)
        
        # Skip if dropout probability is 0
        if dropout_prob == 0.0:
            return inpt
        
        # Generate random value and decide whether to drop
        if torch.rand(1).item() < dropout_prob:
            # Zero out the entire image/tensor
            return torch.zeros_like(inpt)
        
        return inpt


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    # Image dropout configuration: randomly zero out images during training
    # Set to True to enable image dropout augmentation (improves robustness to camera failures)
    image_dropout_enable: bool = False
    # Dropout probability: can be uniform (float) or per-camera (dict[str, float])
    # Uniform example: 0.2 means a 20 percent chance for all cameras
    # Per-camera example: {"image": 0.1, "image2": 0.25} for different rates per camera
    # All probabilities must be in [0.0, 1.0].
    image_dropout_prob: dict[str, float] | float = 0.2
    # Cameras that the `tfs` augmentations (color jitter, sharpness, affine, ...) are applied to.
    # None (default) applies them to every camera. Names may be full feature keys
    # ("observation.images.image") or short names ("image"). Image dropout is configured
    # separately via `image_dropout_prob` and is not affected by this list.
    # CLI example: --dataset.image_transforms.transform_camera_keys="[image]"
    transform_camera_keys: list[str] | None = None
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    elif cfg.type == "ImageDropout":
        return ImageDropout(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

        # Create image dropout transform if enabled
        # Dropout is applied after other transforms (augment first, then potentially drop)
        if cfg.image_dropout_enable and cfg.enable:
            self.dropout = ImageDropout(dropout_probs=cfg.image_dropout_prob)
        else:
            self.dropout = None

        self.transform_cameras = (
            None
            if cfg.transform_camera_keys is None
            else {short_camera_name(k) for k in cfg.transform_camera_keys}
        )

    def applies_transforms_to(self, camera_name: str | None) -> bool:
        """Whether the `tfs` augmentations run on this camera (always True if no camera filter is set)."""
        if self.transform_cameras is None or camera_name is None:
            return True
        return short_camera_name(camera_name) in self.transform_cameras

    def validate_camera_keys(self, camera_keys: Sequence[str]) -> None:
        """Raise if the config names cameras that don't exist in the dataset.

        Unmatched names would otherwise be silently ignored (e.g. 0% dropout, or no augmentation).
        """
        available = {short_camera_name(k) for k in camera_keys}
        problems = []
        if self.transform_cameras is not None:
            unknown = sorted(self.transform_cameras - available)
            if unknown:
                problems.append(f"transform_camera_keys: {unknown}")
        if self.dropout is not None and not self.dropout.uniform:
            unknown = sorted(set(self.dropout.dropout_probs) - available)
            if unknown:
                problems.append(f"image_dropout_prob: {unknown}")
        if problems:
            raise ValueError(
                "image_transforms config references cameras not in the dataset ("
                + "; ".join(problems)
                + f"). Dataset cameras: {sorted(camera_keys)}"
            )

    def forward(self, *inputs: Any, camera_name: str | None = None) -> Any:
        # Apply standard transforms (only to selected cameras, if a camera filter is set)
        if self.applies_transforms_to(camera_name):
            outputs = self.tf(*inputs)
        else:
            outputs = inputs[0] if len(inputs) == 1 else inputs
        
        # Apply dropout if enabled (after other transforms)
        if self.dropout is not None:
            outputs = self.dropout(outputs, camera_name=camera_name)
        
        return outputs
