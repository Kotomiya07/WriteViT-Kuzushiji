 # WriteViT: Handwritten Text Generation with Vision Transformer

<p align="center">
  <img src="./Figures/architecture.png" alt="Model Architecture" width="800"/>
</p>

<p align="center">
  <b>
    <a href="https://arxiv.org/abs/2505.13235">ArXiv</a>
    |
    <a href="https://github.com/DAIR-Group/WriteViT">Code</a>
    |
    <a href="https://colab.research.google.com/drive/15Lswqr-aQwI-fF6yRoGYt-2pxSlC2L-R#scrollTo=abWDlzrTFa_h">
      Demo
    </a>
  </b>
</p>

<p align="center">
  <a href="https://github.com/DAIR-Group/WriteViT">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repo-181717.svg?logo=github&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2505.13235">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.13235-b31b1b.svg">
  </a>
  <a href="https://colab.research.google.com/drive/15Lswqr-aQwI-fF6yRoGYt-2pxSlC2L-R#scrollTo=abWDlzrTFa_h">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

</p>

  
## Software environment

- Python 3.7
- PyTorch >=1.4

## Setup & Training
Please refer to `INSTALL.md` for installation instructions of required libraries.

To visualize generated handwriting during training, you can modify the settings in `params.py`.



Download Dataset files and model from [dataset and checkpoint](https://drive.google.com/drive/folders/1ZgYS6-6l6fjKY75RJipONBByujIgf-uE?usp=sharing)

Quick setup with terminal:

```bash
git clone https://github.com/DAIR-Group/WriteViT.git
cd WriteViT
pip install --upgrade --no-cache-dir gdown
gdown --id 1D_aT7CKEufR87pbfK-fF4wCr3cca6jAg && unzip ckpt.zip && rm ckpt.zip
```

To train the model

```
python train.py
```

If you want to use ```wandb``` please install it and change your auth_key in the ```train.py``` file. 

You can also modify different hyperparameters in  ```params.py``` file.

The dataset is organized as a dictionary containing lists of writer samples 

```python
{
'train': [{writer_1:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_2:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
'test': [{writer_3:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_4:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
}
```
 <!-- ## Run Demo using Docker
```
 docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/ankankbhunia-hwt:latest python app.py
 ``` -->

## Handwriting generation results

 <p align="center">
<img src=Figures/Generation.png width="1000"/>
</p>


## Handwriting reconstruction results
 

 <p align="center">
<img src=Figures/Reconstruction.png width="1000"/>
</p>

## Acknowledgements

A large portion of codes in this repo is based on:[Handwriting-Transformers](https://github.com/ankanbhunia/Handwriting-Transformers) by Ankan Bhunia et al.

We thank the authors for open-sourcing their work, which has been instrumental in developing this project.

 
