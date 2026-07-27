import torch

from microcolossus.config import ModelConfig
from microcolossus.model import DecoderOnlyTransformer


def test_model_forward_and_loss() -> None:
    config = ModelConfig(
        vocab_size=64,
        max_sequence_length=16,
        layers=2,
        heads=2,
        hidden_size=32,
        mlp_ratio=2,
    )
    model = DecoderOnlyTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids, input_ids)

    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_embedding_weights_are_tied() -> None:
    model = DecoderOnlyTransformer(ModelConfig())
    assert model.token_embedding.weight is model.lm_head.weight
