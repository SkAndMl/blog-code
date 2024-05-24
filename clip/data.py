import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Dict, Tuple
import transformers as T
import random


tokenizer = T.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


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
                                          padding=True,
                                          truncation=True,
                                          max_length=200)


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

def create_train_test_dl(df: pd.DataFrame, test_size: float=0.2) -> Tuple[DataLoader]:

  ids = list(range(df.shape[0]))
  test_ids = random.sample(ids, k=int(df.shape[0]*test_size))
  train_ids = [i for i in ids if i not in test_ids]

  train_df = df.iloc[train_ids, :].copy(deep=True)
  test_df = df.iloc[test_ids, :].copy(deep=True)

  train_df = train_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  train_ds, test_ds = CLIPDataset(df=train_df), CLIPDataset(df=test_df)
  train_dl = DataLoader(dataset=train_ds,
                        batch_size=16,
                        shuffle=True)
  test_dl = DataLoader(dataset=test_ds,
                       batch_size=16,
                       shuffle=False)
  return train_dl, test_dl