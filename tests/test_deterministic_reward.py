import torch

from Models.model_reward import DeterministicPPORewardModule


class _AnchorTokenizer:
    def __call__(self, text: str, **_kwargs):
        token_id = {"good": 1, "bad": 2}[text]
        return {"input_ids": torch.tensor([[token_id]])}


def test_deterministic_reward_compares_pooled_embeddings() -> None:
    embedding = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        embedding.weight.zero_()
        embedding.weight[1] = torch.tensor([1.0, 0.0])
        embedding.weight[2] = torch.tensor([0.0, 1.0])
        embedding.weight[3] = torch.tensor([0.8, 0.2])

    reward = DeterministicPPORewardModule(_AnchorTokenizer(), embedding)
    input_ids = torch.tensor([[3, 1], [2, 0]])
    attention_mask = torch.tensor([[1, 1], [1, 0]])

    reward.backbone(input_ids=input_ids, attention_mask=attention_mask)
    scores = reward.score(torch.zeros(2, 2, 1))

    assert scores[:, -1, 0].tolist() == [1.0, -1.0]
    assert "backbone.embedding.weight" in dict(reward.named_parameters())
    assert reward.backbone.embedding.weight.requires_grad is False
