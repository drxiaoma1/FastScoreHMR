# FastScoreHMR: Fast-Few-Step Score-Guided Human Mesh Recovery

Code repository for the paper:
**3D Human Recovery via FAST-FEW-STEP Score-Guided Diffusion**\
[Mingkang Ma](https://github.com/drxiaoma1)

> **A Note on Attribution & Foundation**
>
> This project is built upon and extends the work of the original project, **[ScoreHMR](https://github.com/statho/ScoreHMR)**, created by Anastasis Stathopoulos, Ligong Han, Dimitris Metaxas. We are grateful to the original authors for their work.
>
> Significant portions of this documentation, particularly regarding environment setup, data preparation, and installation, are adapted from or directly sourced from the original project's [README](https://github.com/statho/ScoreHMR/blob/master/README.md). All such reused content remains under the terms of its original [MIT License](https://github.com/statho/ScoreHMR/blob/master/LICENSE.md).

## Installation and Setup
First, clone the repository and submodules. Then, set up a new conda environment and install all dependencies, as follows:
```bash
git clone --recursive https://github.com/drxiaoma1/FastScoreHMR.git
cd FastScoreHMR
source install_environment.sh
```

Download the pretrained model weights, and annotations for the datasets by running the following:
```bash
source download_data.sh
```
This will download all necessary data files, and place them in `data/`. Alternatively, you can download them from [here](https://drive.google.com/file/d/1W53UMg8kee3HGRTNd2aNhMUew_kj36OH/view?usp=sharing) and [here](https://drive.google.com/file/d/1f-D3xhQPMC9rwtaCVNoxtD4BQh4oQbY9/view?usp=sharing). Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de). Please go to the corresponding website and register to get access to the downloads section. Download the model, create a folder `data/smpl`, rename `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl`, and place it in `data/smpl/`.

To reproduce the experimental results of **FastScoreHMR**, you need to download the **best model** we trained as mentioned in the paper. Download it from [here](https://drive.google.com/file/d/1wpWW7lGybdBoU3IMp-z_QG2EPkmT25_a/view?usp=drive_link) and place the model in the `data/model_weights/score_hmr` directory. 

Finally, if you wish to run the evaluation and/or training code, you will need to download the images/videos for the datasets. The instructions are mostly common with the description in [here](https://github.com/nkolot/ProHMR/blob/master/dataset_preprocessing/README.md). We provide the annotations for all datasets, so you will only need to download the images/videos. Edit the `IMG_DIR` in `score_hmr/configs/datasets.yml` accordingly.


## Evaluation
The evaluation code is `eval/eval_keypoint_fitting_shortcut.py`. 

The evaluation code uses cached HMR 2.0 predictions, which can be downloaded from [here](https://drive.google.com/file/d/1m9lv9uDYosIVZ-u0R3GCy1J1wHYNVUMP/view?usp=sharing) or by running:
```bash
source download_hmr2_preds.sh
```
We also provide example code for saving the HMR 2.0 predictions in the appropriate format in `data_preprocessing/cache_hmr2_preds.py`.

Evaluation code example:
```bash
python eval/eval_keypoint_fitting_shortcut.py --dataset 3DPW-TEST --shuffle --use_default_ckpt
```
Running the above command will compute the MPJPE and Reconstruction Error before and after single-frame model fitting with FastScoreHMR on the test set of 3DPW.

## Training
The training code uses cached image features. First, extract the PARE image features for the training datasets:
```
python data_preprocessing/cache_pare_preds.py
```
Then, start training using the following command:
```
python train.py --name <name_of_experiment>
```
Checkpoints and logs will be saved to `logs/`.


## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [ScoreHMR](https://github.com/statho/ScoreHMR)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [PARE](https://github.com/mkocabas/PARE)
- [SLAHMR](https://github.com/vye16/slahmr)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex

```
