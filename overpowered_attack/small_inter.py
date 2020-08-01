from torch.utils.tensorboard import SummaryWriter
from models import MADVAE, Classifier
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import os

R = 20
L = 50000
I = int(L * 0.8)

BATCH_SIZE = 5
LARGE_BATCH = 50

THR = (784*0.0051)**2 / 784
CRIT = nn.CrossEntropyLoss(reduce=False)
LARGE_NUM = 10000
RANGE = torch.arange(0, BATCH_SIZE * R).long()
SAMPLES_PER_ITER = 100
ROBUSTNESS_NORM = 0.5
ADAM_LR = 0.05
SGD_LR = 10.0
IMAGE_DIM = 784
LATENT_DIM = 28


def attack(model,
           cla,
           data,
           labels,
           num_steps=L,
           batch_num=None,
           ind=None,
           suffix=""):
    data = data.view(-1, IMAGE_DIM)
    assert os.path.isfile(f"encodings{suffix}/saved_latents_{batch_num}")
    zh = torch.load(f"encodings{suffix}/saved_latents_{batch_num}")
    zh_ = zh[ind * BATCH_SIZE:(ind + 1) * BATCH_SIZE, ...]

    zhat = zh_.repeat(R, 1)
    targets = data.repeat(R, 1)
    zhat.requires_grad_()

    not_dones_mask = torch.ones(zhat.shape[0]).cuda()
    LAM = 1000 * torch.ones_like(not_dones_mask)
    LAM.requires_grad_()

    opt = optim.Adam([zhat], lr=ADAM_LR)
    lam_opt = optim.SGD([LAM], lr=10000.0)

    lr_maker = StepLR(opt, step_size=I)

    for i in range(num_steps):
        opt.zero_grad()

        # Image Recovery Loss
        gen = model.img_decode(zhat.clone())
        gen = gen.view(-1, IMAGE_DIM)
        loss_mat = ((gen - targets)**2).mean(-1)
        loss_mat = loss_mat * (loss_mat > THR / 2).float() - (loss_mat <=
                                                              THR / 2).float()
        total_loss = loss_mat.clone()
        ttf = targets.view(R * BATCH_SIZE, 1, 28, 28)
        gtf = gen.view(ttf.shape)
        loss_extra = 0

        # Min-max CW loss
        for j in range(SAMPLES_PER_ITER):
            r = torch.randn_like(gtf)
            norm_r = torch.sqrt(r.view(-1, IMAGE_DIM).pow(2).sum(-1)).view(
                -1, 1, 1, 1)
            cla_res = cla.main(gtf + ROBUSTNESS_NORM * r / norm_r)

            cla_res_second_best = cla_res.clone()
            cla_res_second_best[:, labels.repeat(R)] = -LARGE_NUM
            true_classes = cla_res_second_best.argmax(-1)
            loss_new = cla_res[RANGE, labels.
                               repeat(R)] - cla_res[RANGE, true_classes]
            loss_extra = loss_extra + loss_new

        loss_extra = loss_extra / SAMPLES_PER_ITER
        total_loss = loss_extra.mean() + total_loss * LAM
        #new_loss = torch.log(torch.exp(loss_extra).sum())*LAM
        #total_loss += new_loss

        if i % 50 == 0:
            print("Iteration %d | Distance Loss %f | Adversarial Loss %f" %
                  (i, loss_mat.mean(), loss_extra.mean()))

        cla_mat = torch.stack(loss_extra.chunk(R, 0), 0)
        distance_mat = torch.stack(loss_mat.chunk(R, 0), 0)
        not_dones_mask = 1 - (distance_mat <= THR).float() * (cla_mat <=
                                                              -1).float()
        not_dones_mask = not_dones_mask.min(dim=0)[0].repeat(R)
        not_dones_mask = not_dones_mask.view(-1, 1)

        gen = model.img_decode(zhat)
        gen = gen.view(-1, IMAGE_DIM)
        image_mat = torch.stack(gen.chunk(R, 0), 0)
        im_range = torch.range(0, BATCH_SIZE - 1).long()

        ind = (-cla_mat - LARGE_NUM * (distance_mat > THR).float()).argmax(
            0)  # Pick argmin of cla_mat
        loss_at_best = cla_mat[ind, im_range]
        dists_at_best = distance_mat[ind, im_range]

        if not_dones_mask.mean() < 0.1 or i == num_steps - 1:
            zh_mat = torch.stack(zhat.chunk(R, 0), 0)
            best_ims = image_mat[ind, im_range, :]
            best_zhs = zh_mat[ind, im_range, :]
            return best_ims.clone().detach(), zhat.clone().detach()
        elif i % 50 == 0:
            print("----")
            print("Norms", dists_at_best)
            print("Losses", loss_at_best)
            #print("----")
            #print("Maximum loss (of best images)", loss_at_best.max())
            #print("Mean loss (of best images)", loss_at_best.mean())
            #print("----")
            print("Success rate: ", not_dones_mask.mean())
            print("Lambda: ", LAM)

        # this is the loss for the primal variable z
        ((total_loss * not_dones_mask).mean() /
         not_dones_mask.mean()).backward(retain_graph=True)
        opt.step()

        # Lambda step
        lam_opt.zero_grad()
        # this is the loss for the lagrange multiplier/dual variable lambda
        (-(total_loss * not_dones_mask).mean() /
         not_dones_mask.mean()).backward()
        lam_opt.step()
        #LAM.data = torch.max(LAM, 0)[0]

        lr_maker.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Training epoch numbers')
    parser.add_argument('--h_dim',
                        type=int,
                        default=4096,
                        help='Hidden dimensions')
    parser.add_argument('--z_dim',
                        type=int,
                        default=128,
                        help='Latent dimensions for images')
    parser.add_argument('--image_channels',
                        type=int,
                        default=1,
                        help='Image channels')
    parser.add_argument('--image_size',
                        type=int,
                        default=28,
                        help='Image size (default to be squared images)')
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='Number of image classes')
    parser.add_argument('--log_dir',
                        type=str,
                        default='c_logs',
                        help='Logs directory')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for the Adam optimizer')
    parser.add_argument('--closs_weight',
                        type=float,
                        default=0.1,
                        help='Weight for classification loss functions')
    parser.add_argument('--data_root',
                        type=str,
                        default='data',
                        help='Data directory')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_model',
                        help='Pretrained model directory')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='If use GPU for training')
    parser.add_argument('--gpu_num',
                        type=int,
                        default=2,
                        choices=range(0, 5),
                        help='GPU numbers available for parallel training')
    parser.add_argument('--method', type=int, default=0, choices=range(2))
    args = parser.parse_args()

    methods = ['combined', 'vanilla']
    method = methods[args.method]
    ROBUSTNESS_NORM = 3.5

    classifier = Classifier(args)
    classifier_pt = torch.load('classifier.pt')
    classifier.load_state_dict(classifier_pt)
    classifier.eval()

    transform = transforms.Compose(
        [transforms.CenterCrop(args.image_size),
        transforms.ToTensor()])
    testset = datasets.MNIST('./data',
                            train=False,
                            download=True,
                            transform=transform)
    dataloader = torch.utils.data.DataLoader(testset,
                                            batch_size=LARGE_BATCH,
                                            shuffle=False,
                                            num_workers=1)

    print(f"Executing method {method} with norm {ROBUSTNESS_NORM}")
    method_ = f"{method}_"

    model = MADVAE(args)
    model_pt = torch.load(
        f'../MAD-VAE/MAD-VAE/pretrained_model/{method}/params.pt')
    model.load_state_dict(model_pt)
    model.eval()

    if torch.cuda.is_available():
        print("Using CUDA")
        model = model.cuda()
        classifier = classifier.cuda()

    enum = enumerate(dataloader)

    # Detect current progress
    i = 0
    while True:
        try:
            k = (i + 1) * (LARGE_BATCH // BATCH_SIZE)
            if not (os.path.isfile(
                    f"nu_intermediates_{method_}{ROBUSTNESS_NORM}/batch_{k}_attack"
            ) and os.path.isfile(
                    f"nu_intermediates_{method_}{ROBUSTNESS_NORM}/batch_{k}_orig")
                    ):
                break
            next(enum)
            i += 1
        except StopIteration as e:
            break

    print(f"Resuming execution from batch {i}...")

    for i, (data_, label_) in enum:
        if torch.cuda.is_available():
            data_, label_ = data_.cuda(), label_.cuda()

        if i * (LARGE_BATCH // BATCH_SIZE) >= 200:
            break

        for j in range(LARGE_BATCH // BATCH_SIZE):
            data = data_[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, ...]
            label = label_[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, ...]
            intermediates, _ = attack(model,
                                    classifier,
                                    data,
                                    label,
                                    num_steps=10000,
                                    batch_num=i,
                                    ind=j,
                                    suffix=f'_{method}')
            k = i * (LARGE_BATCH // BATCH_SIZE) + j
            os.makedirs(f"nu_intermediates_{method_}{ROBUSTNESS_NORM}",
                        exist_ok=True)
            torch.save(
                intermediates,
                f"nu_intermediates_{method_}{ROBUSTNESS_NORM}/batch_{k}_attack")
            torch.save(
                data,
                f"nu_intermediates_{method_}{ROBUSTNESS_NORM}/batch_{k}_orig")
            