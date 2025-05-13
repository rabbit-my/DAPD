# Parameter-Efficient Adaptation of CLIP for Cervical OCT Diagnosis: Aligning Multi-Center Images via Text Consistency
## 📝 Intro

This study proposes a novel method for adapting CLIP to cervical OCT diagnosis by leveraging textual consistency to address **cross-center** image variations. Our approach utilizes LoRA to fine-tune CLIP efficiently and incorporates a dual-alignment mechanism to enhance consistency between image and text features across centers. Experimental results on three datasets demonstrate that our method outperforms existing SOTA approaches in various metrics and cross-center generalization performance, showcasing the potential of combining cross-modal learning with OCT imaging to advance automated cervical disease detection. 

## 🔧 Quick Start

### 📦 Installation

```
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```

### 📁 Dataset Prepare

```
$DATA/
|–– mixed/
|–– huaxi/
|–– xiangya/
```

### 🚀 Training & Evaluation

run a script：
```
python 32shot_run_shell.py
```
## Method
![image](https://github.com/user-attachments/assets/4d89936f-b93b-4317-a099-4a0d1b85b61f)



## Prompt in Text Encoder：
![image](https://github.com/user-attachments/assets/9d869c94-b3ef-41f3-8f75-3a8af6d74511)


## Mian Result

![image](https://github.com/user-attachments/assets/2c497714-f92f-4b9f-aa11-0e53849d32d8)

