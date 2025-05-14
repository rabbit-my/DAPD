# Parameter-Efficient Adaptation of CLIP for Cervical OCT Diagnosis: Aligning Multi-Center Images via Text Consistency



## 📝 Abstract

This study proposes a novel method for adapting CLIP to **cervical OCT** diagnosis by leveraging textual consistency to address **cross-center image variations**. Our approach utilizes LoRA to fine-tune CLIP efficiently and incorporates a dual-alignment mechanism to enhance consistency between image and text features across centers. Experimental results on **three datasets** demonstrate that our method outperforms existing SOTA approaches in various metrics and cross-center generalization performance, showcasing the potential of combining cross-modal learning with OCT imaging to advance automated cervical disease detection. 

## ⚡️ Quick Start



### 📦 Installation

The following instructions are for Linux installation. We would like to recommend the requirements as follows.

```
conda activate your_env
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
pip install -r requirements.txt  
```

### 📁 Dataset Prepare

Please create a folder named DATA and place the dataset inside it, organized in the following structure:

```
$DATA/
|–– mixed/
    |–– images
        |–– class1
        |–– class2
        ...
        |–– class5
|–– huaxi/
|–– xiangya/
```

### 🚀 Training & Evaluation

We created a runnable script using 32-shot as an example, with parameters that can be easily modified as needed：
```
python 32shot_run_shell.py
```
### Method

<img src="https://github.com/user-attachments/assets/4d89936f-b93b-4317-a099-4a0d1b85b61f" alt="image" width="700"/>



### Prompt for Text Encoder

<img src="https://github.com/user-attachments/assets/9d869c94-b3ef-41f3-8f75-3a8af6d74511" alt="image" width="700"/>

### Mian Result

<img src="https://github.com/user-attachments/assets/2c497714-f92f-4b9f-aa11-0e53849d32d8" alt="image" width="700"/>

## 💬 Contact

If you have any questions or would like to collaborate, feel free to open an issue on GitHub:

👉 [Submit an Issue](https://github.com/rabbit-my/DAPD/issues)

⭐ Star this repo if you find it useful!
