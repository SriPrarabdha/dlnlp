import gc

import pytest
import torch

from longdoc.config import ModelConfig, RunConfig
from longdoc.models.attention import RotaryEmbedding
from longdoc.models.modeling import DualChannelCompressor, build_model
from longdoc.registry import EXPERIMENTS, apply_experiment


def tiny_config(architecture: str) -> ModelConfig:
    config = ModelConfig(
        architecture=architecture,
        vocab_size=257,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        max_length=64,
        dropout=0.0,
        attention_window=16,
        projection_rank=8,
        segment_size=16,
        memory_size=16,
        compressed_memory_size=8,
    )
    if architecture == "roformer":
        config.position_encoding = "rope"
        config.rope_gated_bands = True
    if architecture == "longformer":
        config.global_selection = "learned"
    if architecture == "linformer":
        config.projection_type = "resampled"
        config.exact_cls = True
    if architecture == "longshort":
        config.summary_type = "segmented"
    if architecture == "compressive":
        config.compressor = "dual"
        config.detach_memory = False
    return config


@pytest.mark.parametrize("architecture", ["exact", "longformer", "linformer", "longshort", "compressive", "roformer"])
def test_every_architecture_forward_backward(architecture):
    config = tiny_config(architecture)
    model = build_model(config, "finetune", num_labels=3, task_type="multiclass")
    input_ids = torch.randint(5, config.vocab_size, (2, config.max_length))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[1, -7:] = False
    output = model(input_ids, attention_mask, torch.tensor([0, 2]))
    output.loss.backward()
    assert output.logits.shape == (2, 3)
    assert torch.isfinite(output.loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    "architecture", ["exact", "longformer", "linformer", "longshort", "compressive", "roformer"]
)
def test_masked_token_values_cannot_change_classification(architecture):
    config = tiny_config(architecture)
    model = build_model(config, "finetune", num_labels=2, task_type="binary").eval()
    first = torch.randint(5, config.vocab_size, (2, config.max_length))
    mask = torch.zeros_like(first, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :37] = True
    second = first.clone()
    second[~mask] = torch.randint(5, config.vocab_size, (int((~mask).sum()),))
    with torch.no_grad():
        first_logits = model(first, mask).logits
        second_logits = model(second, mask).logits
    torch.testing.assert_close(first_logits, second_logits, atol=1e-6, rtol=1e-6)


def test_registry_builds_every_variant():
    for name, spec in EXPERIMENTS.items():
        config = RunConfig()
        config.model.hidden_size = 32
        config.model.intermediate_size = 64
        config.model.num_heads = 4
        config.model.num_layers = 1
        config.model.max_length = 64
        config.train.lengths = [64]
        apply_experiment(config, name)
        model = build_model(config.model, "finetune", 2, "binary")
        assert model.config.architecture == spec.architecture


def test_mlm_head_has_expected_shape():
    config = tiny_config("exact")
    model = build_model(config, "pretrain")
    ids = torch.randint(5, config.vocab_size, (2, 32))
    mask = torch.ones_like(ids, dtype=torch.bool)
    labels = ids.clone()
    labels[:, ::2] = -100
    output = model(ids, mask, labels)
    assert output.logits.shape == (2, 32, config.vocab_size)
    assert torch.isfinite(output.loss)


def test_longformer_straight_through_selector_receives_gradient():
    config = tiny_config("longformer")
    config.global_stride = 16
    model = build_model(config, "finetune", 2, "binary")
    ids = torch.randint(5, config.vocab_size, (2, 64))
    mask = torch.ones_like(ids, dtype=torch.bool)
    model(ids, mask, torch.tensor([0, 1])).loss.backward()
    scorer = model.encoder.layers[0].attention.scorer
    assert scorer is model.encoder.layers[1].attention.scorer
    assert scorer.weight.grad is not None
    assert scorer.weight.grad.abs().sum() > 0


def test_rope_frequency_gates_start_at_identity():
    baseline_config = tiny_config("roformer")
    baseline_config.rope_gated_bands = False
    gated_config = tiny_config("roformer")
    baseline = build_model(baseline_config, "finetune", 2, "binary").eval()
    gated = build_model(gated_config, "finetune", 2, "binary").eval()
    gated.load_state_dict(baseline.state_dict(), strict=False)
    ids = torch.randint(5, baseline_config.vocab_size, (2, 64))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        expected = baseline(ids, mask).logits
        actual = gated(ids, mask).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_rope_band_zero_refers_to_low_frequencies():
    rope = RotaryEmbedding(head_dim=8, base=10_000.0, gated_bands=False, heads=1)
    rope.set_band_ablation(0)
    angles = rope._angles(torch.tensor([1]), torch.float32)[0, 0]
    assert torch.all(angles[:2] != 0)
    assert torch.all(angles[2:] == 0)


def test_longformer_budget_is_cls_plus_additional_tokens():
    config = tiny_config("longformer")
    config.global_stride = 16
    attention = build_model(config, "finetune", 2, "binary").encoder.layers[0].attention
    hidden = torch.randn(2, 64, config.hidden_size)
    mask = torch.ones(2, 64, dtype=torch.bool)
    indices, valid, _ = attention._global_indices(hidden, mask, None)
    assert indices.shape[1] == 5  # CLS + ceil(64 / 16)
    assert valid.all()


def test_segmented_longshort_uses_configured_slots_per_segment():
    config = tiny_config("longshort")
    config.summary_type = "segmented"
    config.total_summary_slots = 16
    attention = build_model(config, "finetune", 2, "binary").encoder.layers[0].attention
    for length, expected_slots in ((32, 4), (64, 8)):
        tensor = torch.randn(2, config.num_heads, length, config.hidden_size // config.num_heads)
        mask = torch.ones(2, length, dtype=torch.bool)
        summary_k, summary_v, valid = attention._summaries(tensor, tensor, mask)
        assert summary_k.shape[2] == summary_v.shape[2] == valid.shape[1] == expected_slots
        assert valid.all()


def test_segmented_longshort_masks_unused_slots_for_shorter_rows():
    config = tiny_config("longshort")
    config.summary_type = "segmented"
    attention = build_model(config, "finetune", 2, "binary").encoder.layers[0].attention
    tensor = torch.randn(2, config.num_heads, 64, config.hidden_size // config.num_heads)
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[1, 24:] = False
    summary_k, _, valid = attention._summaries(tensor, tensor, mask)
    assert summary_k.shape[2] == 8
    assert valid[0].sum() == 8
    assert valid[1].sum() == 4


def test_longshort_dual_layer_norm_is_learnable():
    attention = build_model(
        tiny_config("longshort"), "finetune", 2, "binary"
    ).encoder.layers[0].attention
    names = dict(attention.named_parameters())
    assert "local_norm.weight" in names
    assert "global_norm.weight" in names


def test_mlm_all_ignored_labels_produce_finite_zero_loss():
    config = tiny_config("exact")
    model = build_model(config, "pretrain")
    ids = torch.randint(5, config.vocab_size, (1, 8))
    mask = torch.ones_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    output = model(ids, mask, labels)
    assert output.loss.item() == 0.0
    output.loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_compressive_summaries_exclude_padding():
    compressor = DualChannelCompressor(hidden_size=4, rate=2, mode="dual", max_slots=4)
    valid = torch.randn(1, 4, 4)
    first = torch.cat((valid, torch.zeros(1, 4, 4)), dim=1)
    second = torch.cat((valid, torch.full((1, 4, 4), 1_000.0)), dim=1)
    mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    first_outputs = compressor(first, mask)
    second_outputs = compressor(second, mask)
    for first_value, second_value in zip(first_outputs, second_outputs):
        if first_value is not None:
            torch.testing.assert_close(first_value, second_value)


def test_pretraining_parameter_counts_are_within_five_percent_of_exact():
    counts = {}
    for architecture in ("exact", "longformer", "linformer", "longshort", "compressive", "roformer"):
        config = ModelConfig(architecture=architecture)
        if architecture == "roformer":
            config.position_encoding = "rope"
        if architecture == "longshort":
            config.projection_rank = 16
        model = build_model(config, "pretrain")
        counts[architecture] = sum(parameter.numel() for parameter in model.parameters())
        del model
        gc.collect()
    reference = counts["exact"]
    assert all(abs(count - reference) / reference <= 0.05 for count in counts.values())
