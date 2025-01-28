import torch
from transformers import AutoModel, AutoTokenizer
import timm
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
import cv2
from PIL import Image
import pandas as pd
import torch.nn.functional as F

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
emb_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.float32)

class TextEncoder(nn.Module):

    def __init__(self) -> None:

      super().__init__()
      self.model = emb_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

      return self.model(**{"input_ids": input_ids, "attention_mask": attention_mask}).last_hidden_state[:, 0, :]
    

class ImageEncoder(nn.Module):

    def __init__(self) -> None:

        super().__init__()
        self.model = timm.create_model(model_name="resnet50",
                                      pretrained=True,
                                      num_classes=0,
                                      global_pool='avg')

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        return self.model(image)
    

def create_transforms():

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return transform

class CLIPDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transforms=None) -> None:

        self.df = df
        self.caption_tokenized = tokenizer(df['caption'].to_list(),
                                          max_length=512, 
                                          padding=True, 
                                          truncation=True, 
                                          return_tensors='pt')


        self.transforms = transforms
        if self.transforms is None:
            self.transforms = create_transforms()


    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        output = {}
        output["input_ids"] = torch.tensor(self.caption_tokenized["input_ids"][idx])
        output["attention_mask"] = torch.tensor(self.caption_tokenized["attention_mask"][idx])
        output["caption"] = self.df["caption"].iloc[idx]

        image = cv2.imread(self.df["image_path"].iloc[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(Image.fromarray(image))

        output['image'] = image.float()

        return output
    


# The embeddings that come out from the image and text encoder are in different shape (2048 and 768 respectively in this case).
# To bring them to the same dimension we introduce the projection layer.

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
               text_embedding_dim: int=1024,
               projection_dim: int=256) -> None:

    super().__init__()

    self.image_encoder = ImageEncoder()
    self.text_encoder = TextEncoder()
    self.image_projection = ProjectionLayer(embedding_dim=image_embedding_dim,
                                            projection_dim=projection_dim)
    self.text_projection = ProjectionLayer(embedding_dim=text_embedding_dim,
                                           projection_dim=projection_dim)


  def forward(self, image: torch.Tensor, token: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor]:

    with torch.no_grad():
      image_embedding = self.image_encoder(image) # B, image_embedding_dim
      text_embedding = self.text_encoder(token, attention_mask) # B, text_embedding_dim

    projected_image_embedding = self.image_projection(image_embedding) # B, projection_dim
    projected_text_embedding = self.text_projection(text_embedding) # B, projection_dim

    text_logits = projected_text_embedding @ projected_text_embedding.T # B, B
    image_logits = projected_image_embedding @ projected_image_embedding.T # B, B
    targets = F.softmax((text_logits+image_logits)/2, dim=-1)

    logits = projected_image_embedding @ projected_text_embedding.T # B, B

    # This compares the cross-modal logits (image-text) with the soft targets derived from intra-modal logits.
    image_loss = self._cross_entropy(logits, targets, reduction="none") 

    # This is similar to image_loss but flips the matrix dimensions to compute text-image alignment.
    text_loss = self._cross_entropy(logits.T, targets.T, reduction="none")
    
    loss = (image_loss+text_loss)/2
    return logits, loss.mean()


  def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str="none") -> torch.Tensor:

    log_softmax = nn.LogSoftmax() # Taking the log avoids numerical instability and is faster.
    loss = -targets*log_softmax(logits) # B, B
    return loss if reduction=="none" else loss.mean()
  

# Target : Indicates how much weight the model should give to the prediction for each class.
# log_softmax : Measures how confident the model is about its prediction for each class.

# If a target probability is high (closer to 1), its corresponding log probability contributes more to the loss.
# If a target probability is low (closer to 0), its corresponding log probability contributes less.

# Forces the predicted distribution to align with the target distribution.

# Penalizing incorrect predictions (log_softmax is low when predictions are incorrect).
# Scaling penalties based on the importance of each target (targets).