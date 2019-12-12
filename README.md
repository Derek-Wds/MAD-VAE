# MAD-VAE: Manifold Awareness adversarial Defense Variational Autoencoder

Dingsu Wang, Frederick Morlock

<font size="3">This repo contains the codes for the research of **MAD-VAE: Manifold Awareness adversarial Defense Variational Autoencoder**, which is a adversarial defense model based on Defense-VAE.</font>

## How to use this repo
### Repo structure

* The details of our research can be find in the `.pdf` file under this repo.
* We provide the pretrained parameters for all of our models. In the `pretrained_model` folder contains pretrained params for classifiers and our models, while in the `experiments/pretrained` folder are the pretraiend params for the test classifiers mentioned in the Defense-VAE paper.
* The plots for our experiments can be found in the `plots` folder.
```
├── LICENSE
├── MAD-VAE.pdf
├── MAD_VAE.py
├── README.md
├── experiments
│   ├── __init__.py
│   ├── test
│   │   ├── __init__.py
│   │   ├── attacks.py
│   │   ├── pretrained
│   │   ├── test_models.py
│   │   └── train_test_models.py
│   ├── test.py
│   ├── test_black.py
│   ├── test_confusion.py
│   └── test_generate_data.py
├── plots
├── plotting
│   ├── UMAP\ Test.ipynb
│   ├── adv_plot.py
│   ├── defense_plot.py
│   ├── latent_plot.py
│   ├── mnist_plot.py
│   └── plotting.py
├── pretrained_model
├── requirements.txt
├── train.py
├── train_classification.py
├── train_cluster.py
├── train_combined.py
└── utils
    ├── __init__.py
    ├── adversarial.py
    ├── classifier.py
    ├── dataset.py
    ├── generate_data.py
    ├── loss_function.py
    └── scheduler.py

```

### Run our codes
* Our codes are based on **Python3**, make sure it is successfully installed on your machines. Since we are using **GPUs** for training, please make sure you have GPU driver (cuda, cudnn) installed and function well.
* Clone our repo from Github by running:
  ```bash
  git clone git@github.com:Derek-Wds/MAD-VAE.git
  cd MAD-VAE
  ```
* Install all the dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```

### Training
* Generate the training data by running:
  ```bash
  cd utils
  python generate_data.py
  ```
  Since generating training data while training usually takes few days (especially the CW attack), it is more efficient to generate data first before training.

  If you find it takes a lot of time to generate data, we provide the training data we have at the link [here](https://drive.google.com/drive/folders/1SVGPW6_Vm9cqXT2MpzULv-xfG8PXwMHw?usp=sharing).

* Train the vanilla model by running following under the main directory:
  ```bash
  python train.py --batch_size=512 --epochs=5 --log_dir="v_log" --gpu_num=2
  ```
  `log_dir` argument is for the Tensorboard log files, while the `gpu_num` argument specifies the number of GPUs you want to use for training. Our scripts supports multi-GPU training up to 4 GPUs.

  Other arguments for the training process can be found in each training files. We would **NOT SUGGEST** to modify arguments such as `h_dim`, `z_dim`, `image_channels`. `image_size` and `num_classes` unless you know what you are doing and know how to modify the model structures correspondingly.

  Training methods for other models are roughly the same by running `train_classification.py`, `train_cluster.py` and `train_combined.py` respectively.

* Visualize the training process by tensorboard:
  ```bash
  tensorboard --logdir v_log --port 9090
  ```
  Then the tensorboard will be available at `localhost:9090`


### Testing
* Testing code is available in the `experiments` directory.
  * `test.py` runs whitebox attacks against a pretrained MAD-VAE – outputs results to files in the `experiments` directory
  * `test_black.py` runs blackbox attacks against a pretrained MAD-VAE – outputs results to files in the `experiments` directory
  * `test_confusion.py` prints the LaTeX code for a test-data confusion matrix
  * `valid_generate_data.py` generates validation data in the data directory at the root of the project.


### Plotting
* We provide the plotting scripts for adversarial images and model output, as well as t-SNE and UMAP dimension reduction algorithms. All these can be found in the `plotting` directory.

## Example Output
* **FGSM** adversarial examples\
![FGSM attack](/plots/fgsm_img.png)
* **CW** adversarial examples\
![CW attack](/plots/cw_img.png)
* Model with Proximity and Distance Loss output\
  **FGSM** attack output
![FGSM proxi out](/plots/fgsm_proxi_dist_img.png)\
  **CW** attack output
![CW proxi out](/plots/cw_proxi_dist_img.png)

## Citation
If you find our ideas are helpful to your research, we would appreciate if you would cite our work by:
```
@misc{madvae2019,
  author = {Dingsu, Wang and Frederick, Morlock},
  title = {MAD-VAE},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Derek-Wds/MAD-VAE}}
}
```
## Credit
This work would not be done without the insights and code from the work [Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks](https://github.com/aamir-mustafa/pcl-adversarial-defense) and [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335).