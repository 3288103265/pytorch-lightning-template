# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import importlib
import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from torch.utils import data
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 image_size=128,
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.test_batch_size = kwargs['test_batch_size']
        self.test_num_workers = kwargs['test_num_workers']
        self.val_batch_size = kwargs['val_batch_size']
        self.val_num_workers = kwargs['val_num_workers']
        self.image_size = image_size
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(
                train=True, image_size=self.image_size)
            self.valset = self.instancialize(
                train=False, image_size=self.image_size)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(
                train=False, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.val_batch_size, num_workers=self.val_num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        # class_args = inspect.signature(self.data_module.__init__).args[1:]
        # print(class_args)
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


if __name__ == '__main__':
    data_module = DInterface(
        dataset='coco_dataset', batch_size=7, val_batch_size=3,
        test_num_workers=1, test_batch_size=3, val_num_workers=1
    )
    data_module.setup(stage='fit')
    loader = data_module.val_dataloader()
    batch = next(iter(loader))
    for item in batch:
        print(item.shape)
