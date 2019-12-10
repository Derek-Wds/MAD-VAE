import torch
from test.attacks import *
from advertorch.attacks import *

# function for construct adversarial images
def add_adv(model, image, label, adv):
    # fast gradient sign method
    if adv == 'fgsm':
        # _, adv_image = fgsm_attack(model, image, label)
        fgsm = GradientSignAttack(model)
        adv_image = fgsm(image, label)
    # iterative fast gradient sign method
    elif adv == 'i-fgsm':
        # _, adv_image = ifgsm_attck(model, image, label)
        ifgsm = LinfBasicIterativeAttack(model)
        adv_image = ifgsm(image, label)
    # iterative least likely sign method
    elif adv == 'iterll':
        _, adv_image = iterll_attack(model, image, label)
    # random fast gradient sign method
    elif adv == 'r-fgsm':
        #_, adv_image = rfgsm_attck(model, image, label)
        alpha = 0.05
        data = torch.clamp(image + alpha * torch.empty(image.shape).normal_(mean=0,std=1).cuda(), min=0, max=1)
        fgsm = GradientSignAttack(model, eps=0.3-alpha)
        adv_image = fgsm(data, label)
    # momentum iterative fast gradient sign method
    elif adv == 'mi-fgsm':
        mifgsm = MomentumIterativeAttack(model)
        adv_image = mifgsm(image, label)
    # projected gradient sign method
    elif adv == 'pgd':
        #_, adv_image = pgd_attack(model, image, label)
        pgd = PGDAttack(model)
        adv_image = pgd(image, label)
    # deepfool attack method
    elif adv == 'deepfool':
        _, adv_image = deepfool_attack(model, image, label)
    # Carlini-Wagner attack
    elif adv == 'cw':
        # _, adv_image = cw_attack(model, image, label)
        cw = CarliniWagnerL2Attack(model, 10, confidence=10, max_iterations=1500)
        adv_image = cw(image, label)
    # simba attack
    elif adv == 'simba':
        _, adv_image = simba_attack(model, image, label)
    elif adv == 'single':
        single = SinglePixelAttack(model)
        adv_image = single(image, label)
    else:
        _, adv_image = image
        print('Did not perform attack on the images!')
    # if attack fails, return original
    if adv_image is None:
        adv_image = image
    if torch.cuda.is_available():
        image = image.cuda()
        adv_image = adv_image.cuda()

    return image, adv_image