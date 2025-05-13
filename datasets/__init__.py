from .xiangya_oct import TiffXiangyaFrameUpdate
from .huaxi_oct import Tiff_huaxi_modified_frame
from .mixed_center_oct import MixedcenterOCT


dataset_list = {
    "xiangya_oct": lambda root_path, shots=None: TiffXiangyaFrameUpdate(root_path),
    "huaxi_oct": lambda root_path, shots=None: Tiff_huaxi_modified_frame(root_path),
    "mixed_center_oct": MixedcenterOCT,  
}

def build_dataset(dataset, root_path, shots=None):
    if dataset in dataset_list:
        return dataset_list[dataset](root_path, shots)  

    raise ValueError(f"Unknown dataset '{dataset}'. Available datasets: {list(dataset_list.keys())}")
