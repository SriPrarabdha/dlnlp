from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from longdoc.training.operations import atomic_write_json


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]


def _round_robin(iterables: list[Iterable[str]]) -> Iterator[str]:
    """Yield one document per corpus in turn so arXiv cannot dominate."""
    active = [iter(iterator) for iterator in iterables]
    while active:
        next_active = []
        for iterator in active:
            try:
                yield next(iterator)
                next_active.append(iterator)
            except StopIteration:
                pass
        active = next_active


def training_texts(data_dir: Path, names: list[str]) -> Iterator[str]:
    from datasets import load_from_disk

    corpora = []
    for name in names:
        dataset = load_from_disk(str(data_dir / "datasets" / name))
        corpora.append(iter(dataset["train"]["text"]))
    yield from _round_robin(corpora)


def train_tokenizer(data_dir: Path, names: list[str], vocab_size: int, force: bool = False) -> Path:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

    output_dir = data_dir / "tokenizer"
    output_path = output_dir / "tokenizer.json"
    if output_path.exists() and not force:
        corpora_path = output_dir / "training_corpora.txt"
        expected = "\n".join(names) + "\n"
        if not corpora_path.exists() or corpora_path.read_text() != expected:
            raise RuntimeError(
                "The existing tokenizer was trained on a different or unknown corpus set. "
                "Pass --force to rebuild it for the requested datasets."
            )
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(training_texts(data_dir, names), trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    temporary = output_path.with_suffix(".json.tmp")
    tokenizer.save(str(temporary))
    temporary.replace(output_path)
    corpora_path = output_dir / "training_corpora.txt"
    corpora_temporary = corpora_path.with_suffix(".txt.tmp")
    corpora_temporary.write_text("\n".join(names) + "\n")
    corpora_temporary.replace(corpora_path)
    atomic_write_json(output_dir / "manifest.json", {
        "training_corpora": names,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
    })
    return output_path


def load_tokenizer(data_dir: str | Path):
    from tokenizers import Tokenizer

    path = Path(data_dir) / "tokenizer" / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}; run download_data.py first")
    return Tokenizer.from_file(str(path))


def configure_token_dependent_fields(model_config, data_dir: str | Path) -> None:
    """Resolve tokenizer-specific IDs without coupling model modules to data code."""
    tokenizer = load_tokenizer(data_dir)
    actual_vocab_size = tokenizer.get_vocab_size()
    if model_config.vocab_size != actual_vocab_size:
        raise ValueError(
            f"model.vocab_size={model_config.vocab_size} does not match the prepared "
            f"tokenizer vocabulary ({actual_vocab_size})"
        )
    if tokenizer.token_to_id("<pad>") != 0:
        raise ValueError("The model requires <pad> to have token ID 0")
    if getattr(model_config, "global_selection", None) != "sentence":
        return
    ids = set()
    for boundary in (".", " .", "!", " !", "?", " ?", "\n"):
        encoded = tokenizer.encode(boundary, add_special_tokens=False).ids
        if encoded:
            ids.add(encoded[-1])
    model_config.sentence_boundary_token_ids = sorted(ids)
