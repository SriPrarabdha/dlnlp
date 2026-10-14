import pytest

from longdoc.config import ModelConfig, RunConfig, stable_config_id


def test_roformer_rejects_odd_head_dimension():
    config = ModelConfig(architecture="roformer", hidden_size=30, num_heads=6)
    with pytest.raises(ValueError, match="even attention head dimension"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [("log_every", 0), ("save_every", 0), ("num_workers", -1), ("mask_probability", 0)],
)
def test_training_values_that_would_fail_unattended_are_rejected(field, value):
    config = RunConfig()
    setattr(config.train, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_segment_summary_capacity_must_cover_max_length():
    config = ModelConfig(
        architecture="longshort",
        max_length=128,
        segment_size=16,
        summary_type="segmented",
        summaries_per_segment=2,
        total_summary_slots=8,
    )
    with pytest.raises(ValueError, match="total_summary_slots"):
        config.validate()


def test_compressed_memory_can_hold_one_compressed_segment():
    config = ModelConfig(
        architecture="compressive",
        segment_size=512,
        compression_rate=4,
        compressed_memory_size=64,
    )
    with pytest.raises(ValueError, match="compressed_memory_size"):
        config.validate()


def test_activation_checkpointing_does_not_change_scientific_config_id():
    first = RunConfig()
    second = RunConfig()
    second.train.gradient_checkpointing = not first.train.gradient_checkpointing

    assert stable_config_id(first) == stable_config_id(second)
