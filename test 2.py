import argparse
from model_cnn import VAE_CNN
from model_fc import VAE_FC
from linearschedule import LinearSchedule
from datasampler import DataSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from IPython.display import Image
from pathlib import Path


from datetime import date
from attacks import Attacks

import os
import csv
import pandas as pd


# def save_image(x, path='real_image.png'):
#     save_image(x, path)

def loss_fn_fc(recon_x, x, mu, logvar, weight):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # print("bce shape ",BCE.shape)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = weight * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE + KLD


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def flatten(x):
    return to_var(x.view(x.size(0), -1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="use gpu? 1 yes 0 no kinda useless", type=int, default=1)
    parser.add_argument("--model", help="choose model architecture from: cnn fc", type=str)
    
    # parameter for fc network
    parser.add_argument("--imgsize", help="img size", type=int, default=784)
    parser.add_argument("--zdim", help="z dim", type=int, default=20)
    parser.add_argument("--hdim", help="hdim of the fc network ", type=int, default=400)

    # parameter for cnn network    
    parser.add_argument("--convlayers", help="tuple list: [(n_channels, kernel_size, stride, padding),(n_channels,...]", type=list, default=[(16, 3, 2, 1)])
    parser.add_argument("--fclayers", help="hdim of the fc network", type=list, default=[100])
    parser.add_argument("--inputsize", help="size of a single dimension of the image", type=int, default=28)
    # load model
    parser.add_argument("--loadmodel", help="load a pretrained model weight?", type=bool, default=False)
    parser.add_argument("--model_path", help="path to model weight", type=str)

    # test parameter

    parser.add_argument("--metric_test", help="which metric to report test result on?",
                        choices=["original",'PGD', 'FGSM', 'Graphic', 'VAEA', 'LatentA'], nargs="+",type=str,default="original")
    parser.add_argument("--loss", help="p for lp loss",
                        choices=['0', '2', 'infinite'], default='infinite')

    parser.add_argument("--bs", help="batch size?", type=int, default=128)

    parser.add_argument("--dataset", help="cifar or mnist?", choices=["mnist", "cifar"], default="mnist")

    # hyper parameters
    parser.add_argument("--KLweight", help="kl weight for vae loss", type=int, default=-0.5)



    args = parser.parse_args()

    if args.model == "cnn":
        model = VAE_CNN(args.convlayers, args.fclayers, args.inputsize, args.zdim, args.KLweight)
    elif args.model == "fc":
        model = VAE_FC(image_size=args.imgsize,h_dim=args.hdim, z_dim=args.zdim)
    # this assume u give the right path. assume google drive already connected.
    # e_best =-1
    # today = date.today()
    # d3 = today.strftime("%y/%m/%d")
    DS = DataSampler()
    # model_save_name = f'vae_cnn_cur_{e_best}.pt' some model u want to load
    # path = f"/content/drive/MyDrive/model_weights/{model_save_name}"
    if args.loadmodel:
        model.load_state_dict(torch.load(args.model_path))
    if args.dataset == "mnist":

        test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
        
        first_x, first_labels = next(iter(test_loader))
        quick_targets_dataset = DS.quick_sample_udata(first_x, first_labels, 10)
    elif args.dataset == "cifar":

        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    today = date.today()
    d = today.strftime("%y/%m/%d")
    fixed_x, _ = next(iter(test_loader))
    
    if args.model == "fc":
        fixed_x = flatten(fixed_x)
    def test():
        # set the evaluation mode
        model.eval()
        n = len(args.metric_test)
        # test loss for the data
        test_loss = [0]*n

        # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
        #with torch.no_grad():
        for i, (x, label) in enumerate(test_loader):
            # reshape the data
            # x = x.view(-1, 28 * 28)
            x = x.to(device)
            if args.model == "fc":

                x = flatten(x)
            for j, m in enumerate(args.metric_test):
                if m == "original":
                # forward pass
                    x_recon, mu, var = model(x)

                    if args.model == "cnn":
                        loss = model.loss_fn(x_recon, x, mu, var, args.KLweight)
                    elif args.model == "fc":
                        loss = loss_fn_fc(x_recon, x, mu, var, args.KLweight)

                    test_loss[j] += loss.item()
                elif m == "FGSM":

                    if args.model == "fc":
                        x_flat = flatten(x)
                        # print(images.shape) #128*28^2
                        # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        adv = Attacks.fgsm_untargeted(model, x_flat, label=7, eps=0.15, clip_min=0,
                                                    clip_max=1.0)
                        # model.train()
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar, args.KLweight)

                    if args.model == "cnn":
                        adv = Attacks.fgsm_untargeted(model, x, label=7,  eps=0.15, is_fc=False,
                                                    clip_min=0,
                                                    clip_max=1.0)  # do something here.
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var, args.KLweight)

                    test_loss[j] += loss.item()
                elif m == "PGD":
                    if args.model == "fc":

                        x_flat = flatten(x)
                        # print(images.shape) #128*28^2
                        # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        adv = Attacks.pgd_untargeted(model, x_flat, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0,
                                             clip_max=1.0)
                        # model.train()
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar,args.KLweight)

                    if args.model == "cnn":
                        adv = Attacks.pgd_untargeted(model, x, label=7, k=10, eps=0.08, eps_step=0.05,is_fc =False,
                                                    clip_min=0,
                                                    clip_max=1.0) # do something here.
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var, args.KLweight)
                elif m == "GNA":
                    if args.model == "fc":

                        # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        adv = Attacks.GNA(model, x,clip_min=0, clip_max=1.0, _sigma = 0.1,_mean=0.1,rand=True)
                        x_flat = flatten(x)
                        # model.train()
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar,args.KLweight)

                    if args.model == "cnn":
                        adv = Attacks.GNA(model, x,clip_min=0, clip_max=1.0, _sigma = 0.1,_mean=0.1,rand=True)
                        
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var, args.KLweight)
                        
                elif m == "GTA":
                    if args.model == "fc":

                        # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        adv =Attacks.GTA(model, x, rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]))
                        x_flat = flatten(x)
                        # model.train()
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar,args.KLweight)

                    if args.model == "cnn":
                        adv = Attacks.GTA(model, x, rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]))
                        
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var, args.KLweight)

                    test_loss[j] += loss.item()
                elif m == "VAEA":
                    if args.model == "fc":

                        # create adv images.perform step on the reconstructed image generated by adv images.
                        norm = args.loss
                        model.eval()
                        
                        (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=label)
                        A = Attacks(model)
                        adv = A.VAEA(model, x, target, chosen_norm=norm, steps=10, eps=0.08, eps_norm=norm)
                        x_flat = flatten(x)
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar,args.KLweight)

                    if args.model == "cnn":
                        adv = Attacks.GNA(model, x,clip_min=0, clip_max=1.0, _sigma = 0.1,_mean=0.1,rand=True)
                        
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var, args.KLweight)
        return test_loss


    test_loss = test()

    test_loader_lenght =  len(test_loader)
    test_loss = [loss / test_loader_lenght for loss in test_loss]
    #test_loss /= len(test_loader)
    exp_nr = 0
    print(f'Test Loss: {test_loss}')
    
    path_para = f'/content/drive/MyDrive/model_para/test_par.csv'   
    header = ['Experiment Date','Exp Nr','Test Loss']
    parameters = [f'{d}',exp_nr,test_loss]
    
    for attr, value in args.__dict__.items():
        #parameters.append(f'{attr}_{value}')
        parameters.append(f'{value}')
        header.append(attr)
        

    if not (os.path.exists(path_para)):
        with open(path_para,'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header)
    else:
        file_df = pd.read_csv(path_para)
        n = file_df.iloc[-1][1]
        exp_nr = int(n) + 1
        parameters[1] = exp_nr
        

        
    with open(path_para,'a', newline='') as file:
        reader = csv.DictReader(file)
        
        writer = csv.writer(file, delimiter=',')
        if not csv.Sniffer().has_header(path_para):
          writer.writerow(header) # Seems not to work
        writer.writerow(parameters)

    recon_x, _, _ = model(fixed_x)
    pic_path=f'/content/drive/MyDrive/recon_images/vae_{args.model}_final_{d}_exp_{exp_nr}.png'
    Path(os.path.dirname(pic_path)).mkdir(parents=True, exist_ok=True)

    save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(),
               pic_path)

    # print("Hello World!")


if __name__ == "__main__":
    main()