import torch
from torch.utils.data import DataLoader
import pandas as pd
from clip_modules import CLIP
from train import df
from data import CLIPDataset, tokenizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from PIL import Image
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIP()
clip.load_state_dict(state_dict=torch.load("clip.pt", map_location=device))
clip = clip.to(device)

@torch.inference_mode()
def create_image_embeddings(df: pd.DataFrame) -> torch.Tensor:

  clip.eval()

  ds = CLIPDataset(df=df)
  dl = DataLoader(dataset=ds, batch_size=64, shuffle=False)

  image_embeddings = []
  for batch in dl:
    image = batch['image']
    image = image.to(device)

    embeddings = clip.image_projection(clip.image_encoder(image)) # B, projection_dim
    image_embeddings.append(embeddings)

  return torch.cat(image_embeddings, dim=0)

image_embeddings = create_image_embeddings(df=df)

@torch.inference_mode()
def inference(query: str, df: pd.DataFrame, n_top_images: int=5):

  query_tokenized = tokenizer([query], padding=True, truncation=True, max_length=200)

  tokens, attention_mask = query_tokenized['input_ids'], query_tokenized['attention_mask']
  tokens, attention_mask = torch.tensor(tokens).to(device), torch.tensor(attention_mask).to(device)

  query_embedding = clip.text_projection(clip.text_encoder(tokens, attention_mask)) # 1, projection_dim

  logits = image_embeddings @ query_embedding.T # No of images, 1
  top_n_indices = torch.topk(logits.squeeze(), n_top_images).indices.tolist()
  top_image_paths = df.iloc[top_n_indices]['image_path'].to_list()
  top_image_captions = df.iloc[top_n_indices]['caption'].to_list()

  fig, axes = plt.subplots(nrows=3, ncols=math.ceil(n_top_images//3), figsize=(20, 8))
  for ax, img_path, caption in zip(axes.flatten(), top_image_paths, top_image_captions):
      img = Image.open(img_path)
      img = img.resize((300, 300))
      img = mpimg.pil_to_array(img)
      ax.imshow(img)
      # ax.set_title(caption)
      ax.axis('off')

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
   
   parser = argparse.ArgumentParser()
   parser.add_argument("-q", "--query", type=str, required=True)

   args = parser.parse_args()
   inference(query=args["query"], df=df, n_top_images=9)