# IDEA: Image Description Enhanced CLIP-Adapter for Image Classification

Official implementation of 'IDEA: Image Description Enhanced CLIP-Adapter for Image Classification'.

## Abstract

CLIP (Contrastive Language-Image Pre-training) has attained great success in pattern recognition and computer vision. Transferring CLIP to downstream tasks (e.g., zero- or few-shot classification) is a hot topic in multimodal learning. However, current studies focus on single‐modality adaptation and fail to capture target‐relevant fine‐grained features. In this paper, we propose an Image Description Enhanced CLIP‑Adapter (IDEA), a multimodal adapter that effectively boosts CLIP’s performance on few‑shot classification tasks. This adapter leverages the textual descriptions in the training set to enhance the model’s ability to capture fine‑grained features. Meanwhile, IDEA is a training-free method for CLIP, and it can be comparable to or even exceeds state-of-the-art models on multiple tasks. Furthermore, we introduce Trainable-IDEA (T-IDEA), which extends IDEA by adding two lightweight learnable components (i.e., a projector and a learnable latent space), further enhancing the model's performance and achieving SOTA results on 11 datasets. As one important contribution, we employ the LLaMA model and design a comprehensive pipeline to generate textual descriptions for images of 11 datasets, resulting in a total of 1,637,795 image-text pairs, named "IMD-11". Our code and data are released at [GitHub](https://github.com/FourierAI/IDEA).

![IDEA](model.png)

## Highlights

- A new training-free adapter termed as IDEA is proposed for CLIP.
- As a first attempt, IDEA explores complement and correlation information among image-text pairs.
- IDEA achieves state-of-the-art performance in extensive experiments.
- A new multimodal dataset is released to the community.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/FourierAI/IDEA
cd IDEA
```

2. Creat an environment and install dependencies:
```bash
conda create -n IDEA python=3.7
conda activate IDEA

pip install -r requirements.txt
```

## Datasets

1. Follow [Datasets.md](DATASETS.md) to install 11 datasets referring to [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch).

2. Download prompts for the datasets from [GoogleDrive](https://drive.google.com/drive/folders/1MuhtKy-_eAmoKpo2umqFgxTtLNnd6JoZ) and extract the files under `$prompts/`.
The directory structure should look like:
```
prompts/
|–– caltech/
|–– dtd/
|–– eurosat/
...
```

## Running

```bash
CUDA_VISIBLE_DEVICES=0 python IDEA.py --config [path-to-config.yaml]
```
For example, run the following command to train IDEA on Caltech101 shot1:
```bash
CUDA_VISIBLE_DEVICES=0 python IDEA.py --config configs/ByDatasets/caltech/caltech101_shot1.yaml
```

## Contributors
[Zhipeng Ye](https://github.com/FourierAI), Feng Jiang, Qiufeng Wang, Kaizhu Huang, [Jiaqi Huang](https://github.com/7BELIEVEYOU)

## Acknowledgement
This repo benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter). Thanks for their wonderful works.

## Citation

```bash
@misc{ye2025ideaimagedescriptionenhanced,
      title={IDEA: Image Description Enhanced CLIP-Adapter}, 
      author={Zhipeng Ye and Feng Jiang and Qiufeng Wang and Kaizhu Huang and Jiaqi Huang},
      year={2025},
      eprint={2501.08816},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.08816}, 
}
```

## License

[MIT License](LICENSE)