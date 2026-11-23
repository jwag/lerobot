#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Unit tests for SmolVLA hoisted denoise masks and optional CUDA Graph replay."""

from pathlib import Path

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from tests.utils import require_cuda, require_package

LOCAL_SAURON_CKPT = Path("/home/erdc/robotics/ermine_ws/models/smolvla_sauron_star_thawed/160000/pretrained_model")


def test_build_denoise_masks_matches_unhoisted():
    """Verify `_build_denoise_masks` produces bit-identical masks and position IDs."""
    batch_size = 2
    prefix_len = 64
    chunk_size = 50
    prefix_pad_masks = torch.ones((batch_size, prefix_len), dtype=torch.bool)
    prefix_pad_masks[0, -10:] = False
    prefix_pad_masks[1, -5:] = False

    # Reference computation matching unhoisted `denoise_step`
    suffix_pad_masks = torch.ones(batch_size, chunk_size, dtype=torch.bool)
    suffix_att_masks = torch.ones((batch_size, chunk_size), dtype=torch.bfloat16)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, chunk_size, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    ref_full_att = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    ref_pos_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1

    # Construct a lightweight dummy with just config needed for `_build_denoise_masks`
    dummy = type("_Dummy", (), {"config": SmolVLAConfig(chunk_size=chunk_size)})()
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    full_att, pos_ids = VLAFlowMatching._build_denoise_masks(dummy, prefix_pad_masks, dtype=torch.bfloat16)
    assert torch.equal(full_att, ref_full_att)
    assert torch.equal(pos_ids, ref_pos_ids)


@require_package("transformers")
@require_cuda
@pytest.mark.skipif(not LOCAL_SAURON_CKPT.exists(), reason="Local SmolVLA checkpoint not present")
def test_smolvla_hoisted_and_cuda_graph_bit_identical():
    """Verify hoisted eager and optional CUDA Graph replay match unhoisted eager bit-for-bit."""
    policy = SmolVLAPolicy.from_pretrained(LOCAL_SAURON_CKPT).to("cuda").eval()
    model = policy.model

    torch.manual_seed(0)
    batch = {
        "observation.images.image": torch.rand(1, 3, 256, 256, device="cuda"),
        "observation.images.image2": torch.rand(1, 3, 256, 256, device="cuda"),
        "observation.images.image3": torch.rand(1, 3, 256, 256, device="cuda"),
        "observation.images.image4": torch.rand(1, 3, 256, 256, device="cuda"),
        "observation.state": torch.randn(1, 30, device="cuda"),
        "observation.language.tokens": torch.randint(0, 1000, (1, 48), device="cuda"),
        "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool, device="cuda"),
    }
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

    with torch.no_grad():
        images, img_masks = policy.prepare_images(dict(batch))
        state = policy.prepare_state(dict(batch))
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / policy.config.num_steps
        x_ref = noise.clone()
        for step in range(policy.config.num_steps):
            ts = torch.tensor(1.0 + step * dt, dtype=torch.float32, device="cuda").expand(1)
            v_t = model.denoise_step(prefix_pad_masks, past_key_values, x_ref, ts)
            x_ref = x_ref + dt * v_t
        ref_actions = x_ref[:, :, : policy.config.action_feature.shape[0]]

        policy.config.denoise_cuda_graph = False
        eager_actions = policy.predict_action_chunk(dict(batch), noise=noise.clone())
        assert torch.equal(eager_actions, ref_actions)

        policy.config.denoise_cuda_graph = True
        graph_actions_1 = policy.predict_action_chunk(dict(batch), noise=noise.clone())
        graph_actions_2 = policy.predict_action_chunk(dict(batch), noise=noise.clone())
        assert torch.equal(graph_actions_1, ref_actions)
        assert torch.equal(graph_actions_2, ref_actions)
