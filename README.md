# MAD-VAE
* The details of our research can be find in the `.pdf` file under this repo.
* We provide the pretrained parameters for all of our models. In the `pretrained_model` folder contains pretrained params for classifiers and our models, while in the `experiments/pretrained` folder are the pretraiend params for the test classifiers mentioned in the Defense-VAE paper.
* The plots for our experiments can be found in the `plots` folder.

## How to use this repo
### Repo structure
```
├── experiments
│   ├── adversarial.py
│   ├── classifier.py
│   ├── __init__.py
│   ├── test
│   │   ├── attacks.py
│   │   ├── __init__.py
│   │   ├── plotting.py
│   │   ├── pretrained
│   │   ├── test_models.py
│   │   └── train_test_models.py
│   ├── test_black.py
│   ├── test_confusion.py
│   ├── test_generate_data.py
│   ├── test.py
│   └── UMAP Test.ipynb
├── LICENSE
├── MAD-VAE.pdf
├── MAD_VAE.py
├── plots
├── pretrained_model
│   ├── classification
│   ├── combined
│   ├── proxi_dist
│   └── vanilla
├── README.md
├── requirements.txt
├── train_classification.py
├── train_cluster.py
├── train_combined.py
├── train.py
└── utils
    ├── dataset.py
    ├── generate_data.py
    ├── __init__.py
    ├── loss_function.py
    └── scheduler.py

```

### Run our codes
* Our codes are based on **Python3**, make sure it is successfully installed on your machines. Since we are using **GPUs** for training, please make sure you have GPU driver (cuda, cudnn) installed and function well.
* Install all the dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```
* Generate the training data by running:
  ```bash
  cd utils
  python generate_data.py
  ```
  Since generating training data while training usually takes few days (especially the CW attack), it is more efficient to generate data first before training.
* Train the vanilla model by running:
  ```bash
  python train.py --batch_size=512 --epochs=5 --log_dir="v_log" --gpu_num=2
  ```
  Training other models are roughly the same by running `train_classification.py`, `train_cluster.py` and `train_combined.py` respectively.
* Visualize the training process by tensorboard:
  ```bash
  tensorboard --logdir v_log --port 9090
  ```
  Then the tensoboard will be available at `localhost:9090`

## Example Output
* **FGSM** adversarial examples\
![FGSM attack](/plots/fgsm_img.png)
* **CW** adversarial examples\
![CW attack](/plots/cw_img.png)
* Model with Proximity and Distance Loss output\
![FGSM proxi out](/plots/fgsm_proxi_dist_img.png)\
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
This work would not be done without the insights and code from the work [Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks](https://github.com/aamir-mustafa/pcl-adversarial-defense).