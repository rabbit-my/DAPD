# Parameter-Efficient Adaptation of CLIP for Cervical OCT Diagnosis: Aligning Multi-Center Images via Text Consistency



## 📝 Abstract

This study proposes a novel method for adapting CLIP to **cervical OCT** diagnosis by leveraging textual consistency to address **cross-center image variations**. Our approach utilizes LoRA to fine-tune CLIP efficiently and incorporates a dual-alignment mechanism to enhance consistency between image and text features across centers. Experimental results on **three datasets** demonstrate that our method outperforms existing SOTA approaches in various metrics and cross-center generalization performance, showcasing the potential of combining cross-modal learning with OCT imaging to advance automated cervical disease detection. 

## ⚡️ Quick Start



### 📦 Installation

The following instructions are for Linux installation. We would like to recommend the **requirements** as follows.

```
conda create -n your_env python==3.9
conda activate your_env
pip install -r requirements.txt

# install CLIP
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install .
```
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the dassl environment. 

```
git clone https://github.com/KaiyangZhou/Dassl.pytorch
cd Dassl.pytorch-master
pip install .
```

### 📁 Dataset Prepare

Please create a folder named DATA and place the dataset inside it, organized in the following structure:

```
$DATA/
|–– mixed/
    |–– images
        |–– class1
            |–– ×××1.png
            |–– ×××2.png
            ...
        |–– class2
        ...
        |–– class5
|–– huaxi/
    ...
|–– xiangya/
    ...
```
You can replace **mixed**, **huaxi**, and **xiangya** with your own dataset.

**Load dataset** in `datasets/.`According to the paper, we select data from the mixed-center as the few-shot support set.
```
datasets/
|–– __init__.py
|–– huaxi_oct.py # Return the test set(External B dataset).
|–– mixed_center_oct.py  #Return the training set, validation set, and test set.
|–– xiangya.py # Return the test set (External A dataset).
|–– utils.py
```

Our code supports two options for dataset splits: using separate JSON files for training, validation, and testing, or performing random splits based on a given ratio. See `datasets/mixed_center_oct.py` for details.

### 🚀 Training & Evaluation

We written a runnable script using 32-shot as an example, with parameters that can be easily modified as needed：
```
python 32shot_run_shell.py
```
You can set the result saving location in the `32shot_run_shell.py` file by modifying `output_dir = ""`.

### Framework

<img src="https://github.com/user-attachments/assets/4d89936f-b93b-4317-a099-4a0d1b85b61f" alt="image" width="700"/>



### Prompt for Text Encoder

Here, we provide the prompts used in the paper for cervical OCT datasets. If you are using a different dataset, feel free to modify these text descriptions in our `prompt.py` file.

<img src="https://github.com/user-attachments/assets/9d869c94-b3ef-41f3-8f75-3a8af6d74511" alt="image" width="650"/>

### Mian Result

<img src="https://github.com/user-attachments/assets/2c497714-f92f-4b9f-aa11-0e53849d32d8" alt="image" width="700"/>

## 💬 Contact

If you have any questions or would like to collaborate, feel free to open an issue on GitHub:

👉 [Submit an Issue](https://github.com/rabbit-my/DAPD/issues)

⭐ Star this repo if you find it useful!


