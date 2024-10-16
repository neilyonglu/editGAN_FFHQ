# EditGAN for FFHQ 256x256

<div align="center">
Official code and tool release for: 


**EditGAN: High-Precision Semantic Image Editing**
**NeurIPS 2021**
</div>

### Requirements

- Python 3.8.

- The code is tested with CUDA 12.6.

- All results are based on NVIDIA GeForce RTX 4080 GPU with 16GB RAM. 

- Set up python virtual environment (anaconda) steps from scratch:
```
conda create -n editgan python=3.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
- On Windows, the compilation requires Microsoft Visual Studio to be in PATH. We recommend installing Visual Studio Community Edition and adding it into PATH using "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" and "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64".
    > If after installing the "Build Tools for Visual Studio 2022" and doing all that was recommended in the other answers, you still can't find the the file in the location mentioned (no Build folder inside Auxiliary) make sure you **Install "Desktop Development With C++ Workload"**, because vcvarsall.bat is part of C++ workload. (In VS, go Tools menu -> Get Tools and Features -> Install the Desktop Development With C++ workload)
- CUDA setting: Follow this link below https://blog.csdn.net/sinat_34770838/article/details/136946280 or https://qqmanlin.medium.com/cuda-%E8%88%87-cudnn-%E5%AE%89%E8%A3%9D-e982d92162af


### Preparing your dataset - FFHQ

- **Step 1:** Follow these steps to download and preprocess [FFHQ dataset](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) (1024x1024) (89.1 GB)
  - Folder structure as below:

    ```
    ffhq_dataset/
    ├── ffhq-dataset-v2.json // not necessary file
    └── images1024x1024/
        ├── LICENSE.txt
        ├── 00000/
            ├── 00000.png
            ├── 00001.png
            └── ...  
        ├── 01000/
            ├── 01000.png
            ├── 01001.png
            └── ...
        ├── 02000/
            ...
        └── 69000/

    ```
- **Step 2:** From 1024x1024 resolution change to 256x256

    ```
    python resolution_change.py
    ```
    > [!NOTE]
    > Please check the **source_dir** (1024x1024) and **target_dir** (256x256)


### Training steps

Here, we provide step-by-step instructions to create a new EditGAN model. We use our fully released *Face* class as an example.

- **Step 0:** Train StyleGAN2.

  - Download StyleGAN2 training images from FFHQ.

  - Train your own StyleGAN2 model using the official [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) code. Note the specific "stylegan2ada_checkpoint" fields in
    `experiments/datasetgan_face.json ; experiments/encoder_face.json ; experiments/tool_face.json`.
  
  

- **Step 1:** Train StyleGAN2 Encoder. 

  - Specify location of StyleGAN2 checkpoint in the "stylegan_checkpoint" field in `experiments/encoder_face.json`.

  - Check exp_dir, category, im_size etc.

  - Specify path with training images (FFHQ) downloaded in **Step 0** in the "training_data_path" field in `experiments/encoder_face.json`.

  - Run `python train_encoder.py --exp experiments/encoder_face.json`.

    

- **Step 2:** Train DatasetGAN.

  - Specify "stylegan_checkpoint" field in `experiments/datasetgan_car.json`.

  - Download DatasetGAN training images and annotations from [drive](https://drive.google.com/drive/u/1/folders/17vn2vQOF1PQETb1ZgQZV6PlYCkSzSRSa) and fill in "annotation_mask_path" in `experiments/datasetgan_car.json`.

  - Embed DatasetGAN training images in latent space using

    ```
    python train_encoder.py --exp experiments/encoder_car.json --resume *encoder checkppoint* --testing_path data/annotation_car_32_clean --latent_sv_folder model_encoder/car_batch_8_loss_sampling_train_stylegan2/training_embedding --test True
    ```

    and complete "optimized_latent_path" in `experiments/datasetgan_car.json`.

  - Train DatasetGAN (interpreter branch for segmentation) via

    ```
    python train_interpreter.py --exp experiments/datasetgan_car.json
    ```

- **Step 3:** Run the app.

  - Download DatasetGAN test images and annotations from [drive](https://drive.google.com/drive/u/1/folders/1DxHzs5XNn1gLJ_6vAVctdl__nNZerxue). 

  - Embed DatasetGAN test images in latent space via

    ```
    python train_encoder.py --exp experiments/encoder_car.json --resume *encoder checkppoint* --testing_path *testing image path* --latent_sv_folder model_encoder/car_batch_8_loss_sampling_train_stylegan2/training_embedding --test True
    ```

  - Specify the "stylegan_checkpoint", "encoder_checkpoint", "classfier_checkpoint", "datasetgan_testimage_embedding_path" fields in `experiments/tool_car.json`.

  - Run the app via `python run_app.py`.

### Inference

  - Download all checkpoints from [checkpoints](https://drive.google.com/drive/folders/1neucNSPp23CeoZs7n5JxrlaCi_rLhwAj?usp=sharing) and put them into a **./checkpoint** folder:

  - **./checkpoint/stylegan_pretrain**: Download the pre-trained checkpoint from [StyleGAN2](https://github.com/NVlabs/stylegan2) and convert the tensorflow checkpoint to pytorch. We also released the converted checkpoint for your convenience. 
  - **./checkpoint/encoder_pretrain**: Pre-trained encoder.
  - **./checkpoint/encoder_pretrain/testing_embedding**: Test image embeddings.
  - **./checkpoint/encoder_pretrain/training_embedding**: Training image embeddings.
  - **./checkpoint/datasetgan_pretrain**: Pre-trained DatasetGAN (segmentation branch).

- Run the app using `python run_app.py`.

- The app is then deployed on the web browser at `locolhost:8888`.