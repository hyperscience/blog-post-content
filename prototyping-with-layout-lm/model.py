from dataclasses import dataclass

import torch
from torch import nn
from transformers import LayoutLMForTokenClassification

from ml_framework.zoo.unstructured_nlp.config import LAYOUT_LM_PRETRAINED_STR


@dataclass
class ModelPrediction:
    # (batch_size, MAX_WINDOW_SIZE, num_labels)
    logits: torch.Tensor = None
    # (batch_size, MAX_WINDOW_SIZE)
    predictions: torch.Tensor = None
    loss: torch.Tensor = None


class UnstructuredFieldIDModel(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.model = LayoutLMForTokenClassification.from_pretrained(
            LAYOUT_LM_PRETRAINED_STR, num_labels=num_labels
        )

    @staticmethod
    def post_process_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=2)

    def forward(
        self,
        token_ids: torch.Tensor,
        positions_normalized: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> ModelPrediction:
        # passing the labels will automatically compute Sparse Cross Entropy loss
        out_model = self.model(
            input_ids=token_ids,
            bbox=positions_normalized,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return ModelPrediction(
            logits=out_model.logits,
            predictions=self.post_process_logits(out_model.logits),
            loss=out_model.loss,
        )