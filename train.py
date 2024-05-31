from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from timm.optim import AdaBelief, RAdam, Lookahead
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses._functional import soft_dice_score

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from matplotlib import pyplot as plt
import PIL.Image as Image


@dataclass
class Config:
    device = 1
    model_name = 'dataset_v2'
    out_path = f'/home/mpavlov/U-Net/TRAINout/{model_name}'
    train_size = 800

    data_path = Path('/home/mpavlov/U-Net/data')

    train = 'train'
    val = 'val'
    test = 'test'

    train_batch_size = 4
    val_batch_size = 1
    test_batch_size = 1

    max_epochs = 400


preproc = val_augs = A.Compose([A.Normalize(), ToTensorV2(transpose_mask=True)])
preproc1 = A.Compose([ A.Normalize(), ToTensorV2()])

train_augs = A.Compose(
    [A.RandomCrop(Config.train_size, Config.train_size, p=1), A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
     A.OneOf([A.ElasticTransform(p=.3), A.GaussianBlur(p=.3), A.GaussNoise(p=.3), A.OpticalDistortion(p=0.3),
              A.GridDistortion(p=.1), A.PiecewiseAffine(p=0.3), ], p=0.3), A.OneOf(
        [A.HueSaturationValue(15, 25, 0), A.CLAHE(clip_limit=2),
         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3), ], p=0.3), preproc])


class SeedDataset(Dataset):
    def __init__(self, data_path: Path, dir: str,
                 augs=None, class_values=(100, 132, 193, 108, 101, 63, 152, 215, 124, 23, 104, 231)) -> None:
        imgs = (data_path / 'Images' / dir).glob('*')
        self.imgs = sorted(list(imgs))

        masks = (data_path / 'Masks' / dir).glob('*')
        self.masks = sorted(list(masks))

        self.augs = augs
        self.class_values = class_values

    def __getitem__(self, i: int):
        #print(i)
        img = cv2.imread(self.imgs[i].as_posix())
        #print("here")
        mask = cv2.imread(self.masks[i].as_posix(), 0)
        mask = (mask[..., None] == self.class_values).astype(np.float32)
        if self.augs is not None:
            img, mask = self.augs(image=img, mask=mask).values()

        return img, mask

    def __len__(self) -> int:
        return len(self.imgs)


class SeedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_weights='imagenet', decoder_attention_type='scse', classes=12)

    def loss(self, pr, gt):
        #print(pr.size())
        #print(gt.size())
        return -soft_dice_score(pr.sigmoid(), gt, dims =(2,3)).mean() + F.binary_cross_entropy_with_logits(pr, gt)

    def metric4class(self, pr, gt):  
        return soft_dice_score(pr.sigmoid().round(), gt, dims=(2, 3)).mean(dim=0)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, gt = batch
        pr = self(img)
        loss = self.loss(pr, gt)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        pr = self(img)
        loss = self.loss(pr, gt)
        metrics = self.metric4class(pr, gt)
        self.log_dict(
            {"val_loss": loss, 'valid_iou': metrics.mean(), 'valid_iou_cl1': metrics[0], 'valid_iou_cl2': metrics[1], 'valid_iou_cl3': metrics[2], 
            'valid_iou_cl4': metrics[3], 'valid_iou_cl5': metrics[4], 'valid_iou_cl6': metrics[5], 'valid_iou_cl7': metrics[6], 'valid_iou_cl8': metrics[7],
            'valid_iou_cl9': metrics[8], 'valid_iou_cl10': metrics[9], 'valid_iou_cl11': metrics[10], 'valid_iou_cl12': metrics[11]},
            prog_bar=True)
        return loss

    def configure_optimizers(self):
        weight_decay = 1e-3
        radam = RAdam(self.parameters(), weight_decay=weight_decay)
        return Lookahead(radam)


if __name__ == "__main__":
    torch.set_num_threads(1)

    train_dataset = SeedDataset(
        data_path=Config.data_path,
        dir=Config.train,
        augs=train_augs
    )

    val_dataset = SeedDataset(
        data_path=Config.data_path,
        dir=Config.val,
        augs=val_augs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
    )

    model = SeedModel()

    path = Config.out_path

    callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}_{val_loss:.4f}',
        dirpath=path,
        save_top_k=5,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=Config.device,
        max_epochs=Config.max_epochs,
        default_root_dir=path,
        callbacks=[callback],
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    for img, mask in val_loader:
        pred = model(img)
        break
    

"""torch.set_num_threads(1)
model = SeedModel.load_from_checkpoint("/home/pavlov/my_dir/segmentation/TRAINout/dataset_v1/epoch=394_val_loss=-0.5955.ckpt")
model = model.eval()
image = cv2.imread("img.png")
image = preproc1(image = image)
image = image['image'].unsqueeze(0)
output = model(image)
print(output.data.size())
_, predicted = torch.max(output.data, 1)
print(predicted)
print(predicted.unique(return_counts=True))

mask = np.zeros((1728, 2592, 3))
for i in range (1728):
   for j in range (2592):
      arr = []
      for k in range (7):
         arr.append(output.data[0][k][i][j])
      el = max(arr)
      if (el < 0):
         mask[i][j] = [0, 0, 0]
      else:
         if (arr.index(el) == 0):
            mask[i][j] = [153, 0, 0]
         elif (arr.index(el) == 1):
            mask[i][j] = [153, 0, 204]
         elif (arr.index(el) == 2):
            mask[i][j] = [80, 80, 255]
         elif (arr.index(el) == 3):
            mask[i][j] = [255, 255, 0]
         elif (arr.index(el) == 4):
            mask[i][j] = [0, 0, 255]
         elif (arr.index(el) == 5):
            mask[i][j] = [102, 0, 102]
         elif (arr.index(el) == 6):
            mask[i][j] = [0, 255, 255]
   print(f"{i+1}/1728")
cv2.imwrite('newmy.png', mask)
"""
