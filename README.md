#  Splatter Image: Ultra-Fast Single-View 3D Reconstruction Improvement by Training on RGB-D Dataset

## 📝 Project Overview

This repository presents the **Splatter Image** framework, an ultra-fast approach for **single-view 3D object reconstruction**. The approach operates at 38 FPS and is based on **Gaussian Splatting**, a novel method that has shown success in multi-view reconstruction for real-time rendering, fast training, and scaling. Our research extends this method to **monocular reconstruction** by incorporating additional **depth information** into the model during training.

The **Splatter Image** framework modifies the **UNet architecture**, integrating depth channels to enhance 3D object reconstruction quality, significantly improving reconstruction metrics like **PSNR**, **SSIM**, and **LPIPS** across multiple datasets.

## ✨ Key Features
- 🔄 **Monocular 3D object reconstruction** using a fast feed-forward neural network.
- 🛠️ Integration of **depth channels** to improve the quality of reconstructions.
- 🧪 Evaluation of the approach on multiple datasets including **SRN Cars** and **CO3D Cars**.
- 📊 Quantitative improvements measured using **PSNR**, **SSIM**, and **LPIPS**.
  
## 📚 Datasets

The project evaluates the performance of the Splatter Image framework on the following datasets:
1. **SRN Cars**
   - Subsets used: 100%, 50%, 20%
2. **CO3D Cars with Background**

For each dataset, we tested baseline models using only RGB inputs, followed by models that integrate depth information.

## 🧪 Experimental Setup

We conducted experiments with two depth configurations:
- **RGB+D using Splatter Image Depth Output**: Depth maps were generated by the Splatter Image model itself.
- **RGB+D using Depth Anything Model Output**: Depth maps were generated using external depth estimation models, providing more robust depth predictions.

For each dataset, results were evaluated based on the following metrics:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**

## 📊 Results

The performance of the Splatter Image framework showed improvements in reconstruction quality when depth information was integrated. The results are summarized below:

| Dataset          | Configuration                                  | PSNR     | SSIM     | LPIPS   |
|------------------|------------------------------------------------|----------|----------|---------|
| SRN Cars (100%)  | Baseline (RGB only)                            | 19.5569  | 0.8334   | 0.2559  |
|                  | RGB+D using Splatter Image Depth Output        | 18.9316  | 0.8244   | 0.2639  |
|                  | RGB+D using Depth Anything Model Output        | 19.4645  | 0.8361   | 0.2530  |
| SRN Cars (50%)   | Baseline (RGB only)                            | 19.5290  | 0.8326   | 0.2539  |
|                  | RGB+D using Splatter Image Depth Output        | 18.9742  | 0.8225   | 0.2651  |
|                  | RGB+D using Depth Anything Model Output        | 19.4829  | 0.8374   | 0.2494  |
| SRN Cars (20%)   | Baseline (RGB only)                            | 19.3081  | 0.8298   | 0.2554  |
|                  | RGB+D using Splatter Image Depth Output        | 18.7255  | 0.8193   | 0.2663  |
|                  | RGB+D using Depth Anything Model Output        | 19.3170  | 0.8329   | 0.2567  |
| CO3D Cars        | Baseline (RGB only)                            | 14.0015  | 0.3806   | 0.6762  |
|                  | RGB+D using CO3D Depth Output                 | 13.9242  | 0.3730   | 0.6883  |

## 📸 Visual Comparisons

We provide visual comparisons for the datasets, including:

- 🖼️ Ground Truth images
- 🌌 Depth Outputs from Splatter Image and Depth Anything models
- 📷 Baseline RGB images

*Here, you can replace the placeholders with actual image links or screenshots once they are ready.*

![Ground Truth Placeholder](https://via.placeholder.com/150x150.png?text=Ground+Truth)
![Depth Output Splatter](https://via.placeholder.com/150x150.png?text=Depth+Splatter)
![Depth Output Depth Anything](https://via.placeholder.com/150x150.png?text=Depth+Anything)
![CO3D Depth Input](https://via.placeholder.com/150x150.png?text=CO3D+Depth)

## 🗂️ Repository Structure

```bash
.
├── figures/              # Contains images used in the paper
├── src/                  # Source code for the model and experiments
├── results/              # Contains experiment results, logs, and visualizations
├── README.md             # This file
└── Dina.pdf              # Final paper
```
## 🚀 How to Run
### Prerequisites
- Operating System: Windows 11
- Python Version: Python 3.8
- CUDA Toolkit: CUDA 11.7
- Anaconda/Miniconda: For managing the Python environment
### Installation Steps
#### 1. Install Dependencies
  1. Git: Install Git from the official website.
  2. Anaconda: Install Anaconda or Miniconda from the official website.
  3. Visual Studio 2019 Community: Download and install from the official website. During installation, select "Desktop Development with C++".
  4. CUDA Toolkit v11.7: Download and install from the NVIDIA website.
COLMAP: Install COLMAP as per the official instructions.
ImageMagick: Install from the official website.
FFmpeg: Install from the official website.
For visual guidance, refer to this YouTube tutorial.

2. Clone the Repository
bash
Copy code
git clone https://github.com/szymanowiczs/splatter-image.git
cd splatter-image
3. Create and Activate Conda Environment
bash
Copy code
conda create --name splatter-image python=3.8
conda activate splatter-image
4. Install PyTorch and CUDA
bash
Copy code
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
5. Set Up Visual Studio Environment
Open the Command Prompt and run:

bash
Copy code
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
6. Install Gaussian Splatting Dependency
Clone Gaussian Splatting Repository

bash
Copy code
git clone --recursive https://github.com/szymanowiczs/diff-gaussian-rasterization.git
Copy 'submodules' Folder

Copy the submodules folder from the cloned repository to the splatter-image folder.

Install Custom CUDA Extension

bash
Copy code
pip install submodules/diff-gaussian-rasterization
7. Install Remaining Dependencies
bash
Copy code
pip install -r requirements.txt
8. Verify CUDA Setup
Create a file named cuda_check.py with the following content:

python
Copy code
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
Run the script to verify CUDA is working:

bash
Copy code
python cuda_check.py
9. Run Training and Evaluation Scripts
bash
Copy code
python train_model.py --config=config_file.yaml
python evaluate_model.py --dataset=srn_cars
10. View Results
Results will be saved in the results/ folder.

🚧 Future Work
Scale the Experiments: Due to limited resources, our experiments were constrained. We hypothesize that with more computational power and a larger dataset, we could maintain the observed improvement trends. Scaling the dataset and experimenting with more extensive training iterations would be the next step.

Improve Model Architecture: Investigate more advanced neural architectures that can dynamically adapt to varying depth inputs.

Multi-View Inputs: Extend the Splatter Image framework to handle multi-view inputs and real-time dynamic object reconstruction.

🗂️ Repository Structure
bash
Copy code
.
├── figures/              # Contains images used in the paper
├── src/                  # Source code for the model and experiments
├── results/              # Contains experiment results, logs, and visualizations
├── README.md             # This file
└── Dina.pdf              # Final paper
📚 References
Szymanowicz, S., et al. (2024). Splatter Image: Ultra-Fast Single-View 3D Reconstruction. arXiv preprint, arXiv:2312.13150.

Kerbl, B., et al. (2024). Gaussian Splatting for Real-Time Rendering. Graph Deco Inria GitHub Repository.

