import patoolib
import pytorch as torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.data import random_split
from torchvision.datasets.utils import download_url


class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        self.num_classes = 102

    def prepare_data(self):
        # source: https://figshare.com/articles/dataset/Caltech101_Image_Dataset/7007090
        url = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12855005/Caltech101ImageDataset.rar"
        # download
        download_url(url, self.data_dir)
        # extract
        patoolib.extract_archive(
            "Caltech101ImageDataset.rar", outdir=self.data_dir
        )

    def setup(self, stage=None):
        # build dataset
        caltect_dataset = ImageFolder("Caltech101")
        # split dataset
        self.train, self.val, self.test = random_split(
            caltect_dataset, [6500, 1000, 1645]
        )
        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


# ----------------


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=True)
        # layers are frozen by using eval()
        self.feature_extractor.eval()
        # freeze params
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        n_sizes = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_sizes, num_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)

        return x
