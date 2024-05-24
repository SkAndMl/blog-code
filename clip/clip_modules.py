from torch import nn
import timm
import torch
import transformers as T
import torch.nn.functional as F
from typing import Tuple

class ImageEncoder(nn.Module):

    def __init__(self) -> None:

        super().__init__()
        self.model = timm.create_model(model_name="resnet50",
                                      pretrained=True,
                                      num_classes=0,
                                      global_pool='avg')

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        return self.model(image)
    

class TextEncoder(nn.Module):

    def __init__(self) -> None:

      super().__init__()
      self.model = T.DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

      return self.model(**{"input_ids": input_ids, "attention_mask": attention_mask}).last_hidden_state[:, 0, :]

class ProjectionLayer(nn.Module):

  def __init__(self,
               embedding_dim: int,
               projection_dim: int) -> None:

    super().__init__()

    self.projection_layer = nn.Linear(in_features=embedding_dim,
                                      out_features=projection_dim)
    self.gelu = nn.GELU()
    self.fc = nn.Linear(in_features=projection_dim,
                        out_features=projection_dim)
    self.dropout = nn.Dropout(p=0.1)
    self.layer_norm = nn.LayerNorm(normalized_shape=projection_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    projected = self.projection_layer(x)

    x = self.dropout(self.fc(self.gelu(projected))) + projected
    x = self.layer_norm(x)

    return x
  
class CLIP(nn.Module):

  def __init__(self,
               image_embedding_dim: int=2048,
               text_embedding_dim: int=768,
               projection_dim: int=256) -> None:

    super().__init__()

    self.image_encoder = ImageEncoder()
    self.text_encoder = TextEncoder()
    self.image_projection = ProjectionLayer(embedding_dim=image_embedding_dim,
                                            projection_dim=projection_dim)
    self.text_projection = ProjectionLayer(embedding_dim=text_embedding_dim,
                                           projection_dim=projection_dim)


  def forward(self, image: torch.Tensor, token: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor]:

    image_embedding = self.image_encoder(image) # B, image_embedding_dim
    text_embedding = self.text_encoder(token, attention_mask) # B, text_embedding_dim

    projected_image_embedding = self.image_projection(image_embedding) # B, projection_dim
    projected_text_embedding = self.text_projection(text_embedding) # B, projection_dim

    text_logits = projected_text_embedding @ projected_text_embedding.T # B, B
    image_logits = projected_image_embedding @ projected_image_embedding.T # B, B
    targets = F.softmax((text_logits+image_logits)/2, dim=-1)

    logits = projected_image_embedding @ projected_text_embedding.T # B, B
    image_loss = self._cross_entropy(logits, targets, reduction="none")
    text_loss = self._cross_entropy(logits.T, targets.T, reduction="none")
    loss = (image_loss+text_loss)/2
    return logits, loss.mean()


  def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str="none") -> torch.Tensor:

    log_softmax = nn.LogSoftmax()
    loss = -targets*log_softmax(logits) # B, B
    return loss if reduction=="none" else loss.mean()