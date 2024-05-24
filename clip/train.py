import pandas as pd
import re
import torch
from data import create_train_test_dl, CLIPDataset, tokenizer
from clip_modules import CLIP

with open("/kaggle/input/flickr8k/captions.txt", "r") as f:
    text = f.read()

data = {
    "image_path": [],
    "caption": []
}

pattern = r"^([^,]+),(.*)$"

for i, txt in enumerate(text.split("\n")[1:]):
    match_obj = re.match(pattern, txt)
    if match_obj and f"/kaggle/input/flickr8k/Images/{match_obj.group(1)}" not in data["image_path"]:
        data["image_path"].append(f"/kaggle/input/flickr8k/Images/{match_obj.group(1)}")
        data["caption"].append(match_obj.group(2))

df = pd.DataFrame(data=data)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dl, test_dl = create_train_test_dl(df=df)
clip = CLIP().to(device)
optimizer = torch.optim.AdamW(params=clip.parameters(), lr=1e-4)

def train_step() -> float:

  clip.train()
  total_loss = 0

  for batch in train_dl:
    image, token, attention_mask = batch['image'], batch['input_ids'], batch['attention_mask']
    image, token, attention_mask = image.to(device), token.to(device), attention_mask.to(device)

    _, loss = clip(image, token, attention_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss/len(train_dl)

@torch.inference_mode()
def eval_step() -> float:

  clip.eval()
  total_loss = 0

  for batch in test_dl:

    image, token, attention_mask = batch['image'], batch['input_ids'], batch['attention_mask']
    image, token, attention_mask = image.to(device), token.to(device), attention_mask.to(device)

    _, loss = clip(image, token, attention_mask)

    total_loss += loss.item()

  return total_loss/len(test_dl)

def train(epochs: int=2):

  for epoch in range(1, epochs+1):
    train_loss = train_step()
    test_loss = eval_step()

    print(f"({epoch}/{epochs}) train: {train_loss:.4f} test: {test_loss:.4f}")

    torch.save(clip.state_dict(), "clip.pt")


if __name__ == "__main__":
   train(2)