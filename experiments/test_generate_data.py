import sys, os, argparse
sys.path.insert(0, os.path.abspath('..'))
from utils.generate_data import *

# argument parser
def parse_args():
    desc = "MAD-VAE for adversarial defense"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Training epoch numbers')
    parser.add_argument('--h_dim', type=int, default=4096, help='Hidden dimensions')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimensions for images')
    parser.add_argument('--image_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--image_size', type=int, default=28, help='Image size (default to be squared images)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of image classes')
    parser.add_argument('--log_dir', type=str, default='v_logs', help='Logs directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--data_root', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='Pretrained model directory')
    parser.add_argument('--use_gpu', type=bool, default=True, help='If use GPU for training')
    parser.add_argument('--gpu_num', type=int, default=2, choices=range(0,5), help='GPU numbers available for parallel training')

    return parser.parse_args()

if __name__ == "__main__":
    # get arguments
    args = parse_args()
    # init and load model
    classifier = Classifier(args)
    classifier.load_state_dict(torch.load('pretrained_model/classifier_mnist.pt'))
    classifier.eval()
    classifier = classifier.cuda()

    # init dataset
    transform  = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=1)
    # adversarial methods
    adv_list = ['fgsm', 'r-fgsm', 'cw', 'mi-fgsm', 'pgd', 'single']

    for adv in adv_list:
        xs = []
        ys = []
        advs = []
        for image, label in dataloader:
            image = image.cuda()
            label = label.cuda()

            output, adv_out = add_adv(classifier, image, label, adv)

            xs.append(image.cpu().detach().numpy())
            ys.append(label.cpu().detach().numpy())
            advs.append(adv_out.cpu().detach().numpy())
        adv_x = np.concatenate(advs, axis=0)
        xt = np.concatenate(xs, axis=0)
        yt = np.concatenate(ys, axis=0)

        np.save('../testdata/' + adv + '_advs_mnist.npy', adv_x)
        np.save('../testdata/' + adv + '_xs_mnist.npy', xt)
        np.save('../testdata/' + adv + '_ys_mnist.npy', yt)