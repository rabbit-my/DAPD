import os
import random
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import write_json, read_json
from dassl.utils import listdir_nohidden, mkdir_if_missing


@DATASET_REGISTRY.register()
class Tiff_huaxi_modified_frame(DatasetBase):
    dataset_dir = "tiff_huaxi_modified_frame"

    def __init__(self, root_path):
        root = os.path.abspath(os.path.expanduser(root_path))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        mkdir_if_missing(self.dataset_dir)

        data = self.read_all_data(self.image_dir)

        super().__init__(train_x=data)

    @staticmethod
    def read_all_data(image_dir):
        categories = listdir_nohidden(image_dir)
        categories.sort()

        data = []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]

            for img in images:
                data.append(Datum(impath=img, label=label, classname=category))

        print(f"Loaded {len(data)} samples from {len(categories)} categories")
        return data
