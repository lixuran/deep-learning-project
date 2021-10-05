
from  model_cnn import VAE_CNN
from model_fc import VAE_FC
from linearschedule import LinearSchedule
import segment_tree
from replay_buffer_v2 import ReplayBuffer, PrioritizedReplayBuffer
from attacks import Attacks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from IPython.display import Image

import argparse
from datetime import date
import numpy as np
import os
import random

from datasampler import DataSampler
import csv
from critic import critic_FC

np.random.seed(2020)
torch.manual_seed(2020)
random.seed(2020)

# def save_image(x, path='real_image.png'):
#     save_image(x, path)
def loss_fn_fc(recon_x, x, mu, logvar,weight):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    #print("bce shape ",BCE.shape)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD =weight * torch.mean(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD
def loss_fn_fc_each(recon_x, x, mu, logvar,weight):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False,reduce=False)

    #print("bce shape ",BCE.shape)
    BCE = torch.sum(BCE,1,keepdim=True)
    #print("bce shape 2 ", BCE.shape)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD =weight * torch.mean(1 + logvar - mu**2 -  logvar.exp(),1,keepdim=True)
    #print(KLD.shape)
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
    #parser.add_argument("--save_dir",help="save model parameters and result",type=str)
    parser.add_argument("--zdim", help="z dim", type=int, default=20)
    # parameter for fc network
    parser.add_argument("--hdim", help="hdim of the fc network ", type=int, default=400)
    # parameter for cnn network
    parser.add_argument("--convlayers", help="hdim of the fc network", type=list, default=[(16, 3, 2, 1)])
    parser.add_argument("--fclayers", help="hdim of the fc network", type=list, default=[100])
    parser.add_argument("--inputsize", help="size of a single dimension of the image", type=int, default=28)
    #load model
    parser.add_argument("--loadmodel", help="load a pretrained model weight?", type=bool, default=False)
    parser.add_argument("--model_path", help="path to model weight", type=str)
    # training methods
    parser.add_argument("--useminimax", help="use minimax algorithm to train?", type=bool,default=False)
    parser.add_argument("--adv", help="what single adversarial to choose for minimax?",
                        choices = ['PGD', 'FGSM', 'GTA','GNA', 'VAEA','LatentA','Graphic','RANDOM','TEST'], default='FGSM')
    parser.add_argument("--loss", help="p for lp loss",
                        choices=['0', '2', 'infinite'], default='infinite') #todo: change the loss function
    parser.add_argument("--usereplaybuffer", help="use replay buffer?", type=bool, default=False)
    parser.add_argument("--usemultiadv", help="use multiple adversarial?",
                        type= bool, default=False)
    parser.add_argument("--multiadv", help="use multiple adversarial?",
                        choices = ['PGD', 'FGSM', 'GNA','GTA'], nargs = "+")
    parser.add_argument("--opt", help="optimizer?",
                        choices=['adam', 'SGD'],default="adam")
    parser.add_argument("--lr", help="learning rate?",type= float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--bs", help="batch size?", type=int, default=128)
    parser.add_argument("--epochs", help="iteration?", type=int, default=10)
    parser.add_argument("--dataset", help="cifar or mnist?", choices = ["mnist", "cifar"] ,default="mnist")
    parser.add_argument("--valid_size", help="validation set split", type = float, default=0.2)
    #hyper parameters
    parser.add_argument("--KLweight", help="kl weight for vae loss", type=float, default=-0.5)
    parser.add_argument("--metric_valid", help="which metric to report validation performance on?",
                        choices=["original",'PGD', 'FGSM', 'Graphic','GTA','GNA'], nargs="+")

    #eps, eps_step, k, label, clip_min,clip_max
    parser.add_argument("--eps", help="attack change bound", type=float, default=0.08)
    parser.add_argument("--eps_step", help="pgd step size", type=float, default=0.05)
    parser.add_argument("--k", help="iteration for pgd", type=int, default=10)
    parser.add_argument("--label", help="label for targetd attack", type=int, default=7)
    parser.add_argument("--clip_min", help="input bound", type=float, default=0)
    parser.add_argument("--clip_max", help="input bound", type=float, default=1)

    parser.add_argument("--buffer_size", help="you know the drill XD", type=int, default=6400)
    parser.add_argument("--learning_starts", help="some words blablabla", type=int, default=5)
    parser.add_argument("--train_freq", help="update frequency", type=int, default=10)
    parser.add_argument("--prioritized_replay_alpha", help="alpha", type=float, default=0.6)
    parser.add_argument("--prioritized_replay_eps", help="eps for priority update", type=float, default=1e-6)
    parser.add_argument("--prioritized_replay_beta0", help="beta init", type=float, default=0.4)
    parser.add_argument("--prioritized_replay_beta1", help="beta init", type=float, default=0.6)
    parser.add_argument("--bs_replaybuffer", help="batch size of sample from replay buffer", type=int, default=128)
    parser.add_argument("--prioritized_replay_beta_iters", help="input bound", type=int, default=None)
    parser.add_argument("--prioritized_replay", help="use normal replay buffer or prioritised", type=bool, default=True)
    parser.add_argument("--use_weight", help="use weight during priority calculation?", type=bool, default=False)
    parser.add_argument("--probs", help="probability of each attack of multi attack", type=list, default=[])
    parser.add_argument("--use_critic", help="use critic for multi adv training?", type=bool, default=False)
    parser.add_argument("--best_size", help="best size for critic", type=int, default=128)

    # Init variables
    class empty():
        def __init__(self):
            pass

    init = empty()
    init.adv = ['FGSM','PGD','GTA','GNA','VAEA','LatentA','RANDOM','TEST']
    locN = 0 #index of picture

    args = parser.parse_args()
    train_size = 1-args.valid_size
    if args.model == "cnn":
        model = VAE_CNN(args.convlayers,args.fclayers,args.inputsize,args.zdim,args.KLweight)
    elif args.model =="fc":
        model = VAE_FC(image_size=args.inputsize*args.inputsize,h_dim=args.hdim,z_dim = args.zdim)

    # this assume u give the right path. assume google drive already connected.
    # e_best =-1
    # today = date.today()
    # d3 = today.strftime("%y/%m/%d")

    # model_save_name = f'vae_cnn_cur_{e_best}.pt' some model u want to load
    # path = f"/content/drive/MyDrive/model_weights/{model_save_name}"
    DS = DataSampler()


    if args.loadmodel:
        model.load_state_dict(torch.load(args.model_path))
    if args.dataset == "mnist":

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

        dataset_len = len(train_dataset)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

        first_x, first_labels = next(iter(train_loader))
        quick_targets_dataset = DS.quick_sample_udata(first_x, first_labels, 10)

    elif args.dataset == "cifar":

        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        dataset_len = len(train_dataset)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    A = Attacks(model)
    best_valid_loss = float('inf')
    today = date.today()
    d = today.strftime("%y%m%d")
    fileindex =0
    while os.path.exists(f"/content/drive/MyDrive/model_weights/{d}_{fileindex}"):
        fileindex+=1
    os.mkdir(f"/content/drive/MyDrive/model_weights/{d}_{fileindex}")
    exp_name =f'{d}_{fileindex}'


    # Training Parameter
    path_para = f'/content/drive/MyDrive/model_para/exp_par.csv'

    parameters = [f'{exp_name}']
    header = ['Experiment Name']
    for attr, value in args.__dict__.items():
        #parameters.append(f'{attr}_{value}')
        parameters.append(f'{value}')
        header.append(attr)

    if not (os.path.exists(path_para)):
        with open(path_para,'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header)
    with open(path_para,'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        if not csv.Sniffer().has_header(path_para):
          writer.writerow(header) # Seems not to work
        writer.writerow(parameters)


    fixed_x, _ = next(iter(train_loader))
    #save_image(fixed_x)
    if args.model == "fc":
        fixed_x = flatten(fixed_x)

    buffer_size = args.buffer_size  # todo: this number is not optimal
    learning_starts = args.learning_starts  # learning_starts batches are to be stored in the replay buffer before sampling starts
    train_freq = args.train_freq
    prioritized_replay_alpha = args.prioritized_replay_alpha
    prioritized_replay_eps = args.prioritized_replay_eps  # todo: change this
    prioritized_replay_beta0 = args.prioritized_replay_beta0
    epochs = args.epochs
    total_timesteps = epochs * len(train_loader)
    bs = args.bs
    prioritized_replay_beta_iters = args.prioritized_replay_beta_iters

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0,
                                       final_p=args.prioritized_replay_beta1)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
        raise Exception("implement normal replay buffer")
    def train(e):
        # set the train mode
        model.train()

        # loss of the epoch
        train_loss = 0


        # update the gradients to zero
        optimizer.zero_grad()
        if args.useminimax == False:
        # forward pass
            for i, (x, _) in enumerate(train_loader):
                # print("x shape",x.shape)
                # reshape the data into [batch_size, 784]
                # x = x.view(-1, 28 * 28)
                x = x.to(device)
                if args.model == "fc":
                    x = flatten(x)
                x_recon, mu, var = model(x) # var is probably in reality the log of variance mu is the mean
                optimizer.zero_grad()

                # forward pass
                if args.model == "fc":
                    loss = loss_fn_fc(x_recon, x, mu, var,args.KLweight)
                elif args.model == "cnn":
                # reconstruction loss
                    loss = model.loss_fn(x_recon, x, mu, var)
                # backward pass
                loss.backward()
                train_loss += loss.item()

                # update the weights
                optimizer.step()

        elif args.usereplaybuffer == False:
            #print("ggg")
            if args.adv == "Graphic":
                pass
            elif args.adv in init.adv:
                #print("ggg")
                for idx, (x, y) in enumerate(train_loader):
                    #print("ggg2")
                    if args.model == "fc":
                        x_flat = flatten(x)

                    # print(images.shape) #128*28^2
                    # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        if args.adv == "FGSM" :
                            adv = A.fgsm_untargeted( x=x_flat, label=args.label, eps=args.eps, is_fc=True,
                                                         clip_min=args.clip_min,
                                                         clip_max=args.clip_max)
                        elif args.adv == "PGD":
                            adv = A.pgd_untargeted( x=x_flat, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,
                                                    clip_min=args.clip_min,
                                                   clip_max=args.clip_max)

                        elif args.adv == "GNA":
                            #print('GNA Attack')
                            adv = A.GNA(x,  clip_min=args.clip_min, clip_max=args.clip_max,_sigma=0.1,_mean=0.1,rand=True,is_fc=True)

                            #Test code for picture
                            locN = 0
                            pic_name = f'GNA{exp_name}'
                            if random.random()<0.002:
                                while os.path.exists(f'/content/drive/MyDrive/test_folder/{exp_name}_{locN}.png'):
                                    locN = locN + 1
                                save_image(adv.view(adv.size(0), 1, 28, 28).data.cpu(),
                                       f'/content/drive/MyDrive/test_folder/{exp_name}{locN}.png')

                        elif args.adv == "GTA":
                            adv = A.GTA(x,rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),rand=True,is_fc=True)
                                                       #Test code for picture
                            locN = 0
                            pic_name = f'GTA{exp_name}'
                            if random.random()<0.002:
                                while os.path.exists(f'/content/drive/MyDrive/test_folder/{exp_name}_{locN}.png'):
                                    locN = locN + 1
                                save_image(adv.view(adv.size(0), 1, 28, 28).data.cpu(),
                                       f'/content/drive/MyDrive/test_folder/{exp_name}{locN}.png')

                        elif args.adv == "VAEA":
                            (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                            adv = A.VAEA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)
                        elif args.adv == "LatentA":
                            (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                            adv = A.LatentA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)
                        elif args.adv == "RANDOM":
                            #Draw out of the random states
                            r =  random.random()
                            #p_c = 1.0/len(init.adv)
                            p_c = 1.0/4.0
                            #print(p_c)
                            #print(r)

                            #Use probability and perform random attack
                            if  0 <= r < p_c: #FGSM Attack
                                adv = A.fgsm_untargeted( x=x_flat, label=args.label, eps=args.eps, is_fc=True,
                                    clip_min=args.clip_min,
                                    clip_max=args.clip_max)
                                #print('FGSMattack')

                            elif  p_c <= r < 2*p_c: #PGDAttack
                                adv = A.pgd_untargeted( x=x_flat, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,
                                                clip_min=args.clip_min,
                                               clip_max=args.clip_max)

                                #print('PGDattack')
                            elif  2*p_c <= r < 3*p_c: #GNA  Attack
                                adv = A.GNA(x,  clip_min=args.clip_min, clip_max=args.clip_max,_sigma=0.1,_mean=0.1,rand=True,is_fc=True)
                                #print('GNAAttack')
                            elif  3*p_c <= r <= 4*p_c: #GTA Attack
                                 adv = A.GTA(x,rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),rand=True,is_fc=True)
                                 #print('GTAAttack') #Doesn't work yet
                            elif  4*p_c < r < 5*p_c: #VAEA Attack
                                (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                                adv = A.VAEA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)
                                # Implememt VAEA  here
                                 #print(' VAE Attack')
                            elif  5*p_c < r < 6*p_c: # LatentA Attack
                                (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                                adv = A.LatentA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)

                                # Implememt LatentAhere
                                 #print('Latent Attack')
                            else:
                                print('something went wrong')
                        elif args.adv == "TEST":
                            print('TEST Attack')
                            adv = A.GNA(x,  clip_min=args.clip_min, clip_max=args.clip_max,_sigma=0.1,_mean=0.1,rand=False,is_fc=True)
                            adv3 = A.GNA(x,  clip_min=args.clip_min, clip_max=args.clip_max,_sigma=0.1,_mean=0.1,rand=True,is_fc=True)
                            #adv2 = A.GTA(x,rotate=30,translate=([1,1]),scale = 1.2,rand=False,is_fc=True)
                            adv4 = A.GTA(x,rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),rand=True,is_fc=True)
                            #Test code for picture
                            locN = 0
                            pic_name = f'TEST{exp_name}'
                            if random.random()<0.01:
                                while os.path.exists(f'/content/drive/MyDrive/test_folder/{exp_name}_{locN}.png'):
                                    locN = locN + 1
                                save_image(adv.view(adv.size(0), 1, 28, 28).data.cpu(),
                                       f'/content/drive/MyDrive/test_folder/{exp_name}_GNA{locN}.png')
                                save_image(adv3.view(adv3.size(0), 1, 28, 28).data.cpu(),
                                       f'/content/drive/MyDrive/test_folder/{exp_name}_GNA_RAND{locN}.png')
                                save_image(x.view(x.size(0), 1, 28, 28).data.cpu(),
                                       f'/content/drive/MyDrive/test_folder/{exp_name}_OR{locN}_o.png')
                                save_image(adv4.view(adv4.size(0), 1, 28, 28).data.cpu(),
                                       f'/content/drive/MyDrive/test_folder/{exp_name}_GTA_RAND{locN}.png')


                    elif args.model == "cnn":
                        #print("wtf")
                        model.eval()
                        if args.adv == "FGSM":
                            adv = A.fgsm_untargeted( x, label=args.label, eps=args.eps, is_fc=False,
                                                         clip_min=args.clip_min,
                                                         clip_max=args.clip_max)
                        elif args.adv == "PGD":
                            adv = A.pgd_untargeted( x, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,is_fc =False,
                                                    clip_min=args.clip_min,
                                                    clip_max=args.clip_max)
                        elif args.adv == "GNA":
                            adv = A.GNA(x,clip_min=args.clip_min, clip_max=args.clip_max,_sigma=0.1,_mean=0.1,rand=True,is_fc=False)
                        elif args.adv == "GTA":
                            adv = A.GTA(x,rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),rand=True,is_fc=False)



                        elif args.adv == "VAEA":
                            (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                            adv = A.VAEA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)

                        elif args.adv == "LatentA":
                            (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                            adv = A.LatentA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)
                        elif args.adv == "RANDOM":
                            #Draw out of the random states
                            r =  random.random()

                            p_c = 1.0/4.0
                            #print(p_c)
                            #print(r)
                            #Use probability and perform random attack
                            if  0 <= r < p_c: #FGSM Attack
                                adv = A.fgsm_untargeted( x, label=args.label, eps=args.eps, is_fc=False,
                                                         clip_min=args.clip_min,
                                                         clip_max=args.clip_max)
                                #print('FGSMattack')

                            elif  p_c <= r < 2*p_c: #PGDAttack
                                adv = A.pgd_untargeted( x, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,is_fc =False,
                                                    clip_min=args.clip_min,
                                                    clip_max=args.clip_max)

                                #print('PGDattack')
                            elif  2*p_c <= r < 3*p_c: #GNA  Attack
                             #print('GNAAttack')

                             adv = A.GNA(x,clip_min=args.clip_min, clip_max=args.clip_max,_sigma=0.1,_mean=0.1,rand=True,is_fc=False)

                            elif  3*p_c <= r <= 4*p_c: #GTA Attack
                             #print('GTAAttack')
                             adv = A.GTA(x,rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),rand=True,is_fc=False)
                            elif  4*p_c < r < 5*p_c: #VAEA Attack
                                (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                                adv = A.VAEA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)

                            elif  5*p_c < r < 6*p_c: # LatentA Attack
                                (target, _) = DS.sample_adv_untargeted_quick(quick_targets_dataset, source=x, label=y)
                                adv = A.LatentA(x, target, args.loss, steps=10, eps=0.08, eps_norm=args.loss)
                            else:
                                print('something went wrong, use FGSM attack')
                                adv = A.fgsm_untargeted( x, label=args.label, eps=args.eps, is_fc=False,
                                                         clip_min=args.clip_min,
                                                         clip_max=args.clip_max)


                    model.train()
                    # print("adv.shape ",adv.shape)

                    recon_images, mu, logvar = model(adv)
                    # recon_images = recon_images.clamp_(min=args.clip_min, max=args.clip_max)
                    # print(recon_images.shape)
                    adv = adv.detach()
                    if args.model == "fc":
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar,args.KLweight)
                    elif args.model == "cnn":
                        loss = model.loss_fn(recon_images, x, mu, logvar)
                    # loss2 = loss_fn(recon_images, adv, mu, logvar)
                    # print(loss)
                    # print(loss2)
                    # very important to clear the grad both in inner and outer loop.

                    # backward pass
                    loss.backward()
                    train_loss += loss.item()
                    #print(loss.item())
                    # update the weights
                    optimizer.step()

                    # if idx % 100 == 0:
                    #     print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, args.epochs, loss.item() / args.bs))
                    #
                    #     recon_x, _, _ = model(fixed_x)
                    #     save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(),
                    #                f'reconstructed_ad_training2/recon_image_{epoch}_{idx}.png')

        elif not args.usemultiadv:
            # adversarial training using minimax. with replay buffer single loss
            # train(vae3,xs, advs, recon_xs, losses,mus,vars, weights)
            def train_pgd( opt, xs, advs,  losses,  weights,is_fc,use_weight):
                model.eval()
                if args.adv == "FGSM":
                    adv_new = A.fgsm_untargeted(xs, label=args.label, eps=args.eps, is_fc=is_fc,
                                            clip_min=args.clip_min,
                                            clip_max=args.clip_max)
                elif args.adv == "PGD":
                    adv_new = A.pgd_untargeted( xs, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,
                                            is_fc=is_fc,clip_min=args.clip_min, clip_max=args.clip_max)
                elif args.adv == "GTA":

                    adv_new=A.GTA(xs, rotate=30, translate=([0.01, 0.01]), scale=([0.8, 1.2]), rand=True, is_fc=is_fc)

                elif args.adv == "GNA":
                    # print('GNA Attack')
                    adv_new = A.GNA(xs, clip_min=args.clip_min, clip_max=args.clip_max, _sigma=0.1, _mean=0.1, rand=True,
                                is_fc=is_fc)

                model.train()
                recon_images, mu_new, logvar_new = model(adv_new)
                if(is_fc):
                    losses_new = loss_fn_fc_each(recon_images, xs, mu_new, logvar_new,args.KLweight)
                    if use_weight:
                        losses_new = weights*losses_new/torch.max(weights)
                else: # assume is cnn
                    losses_new = model.loss_fn_each(recon_images, xs, mu_new, logvar_new)
                    if use_weight:
                        losses_new = weights * losses_new/torch.max(weights)
                opt.zero_grad()
                loss_new = torch.sum(losses_new)
                loss_new.backward()
                opt.step()
                #print("lossinpgd",losses.shape)
                #print(losses_new.shape)
                #todo: weights normalization and application.
                return torch.abs(losses_new - losses),losses_new


            for idx, (images, _) in enumerate(train_loader):

                is_fc = args.model =="fc"
                if is_fc:
                    images = flatten(images)
                # print(images.shape) #128*28^2
                # build adv images.perform step on ad images.

                model.eval()
                if args.adv == "FGSM":
                    adv = A.fgsm_untargeted(images, label=args.label, eps=args.eps, is_fc=is_fc,
                                            clip_min=args.clip_min,
                                            clip_max=args.clip_max)
                elif args.adv =="PGD":
                    adv = A.pgd_untargeted(images, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,
                                           is_fc=is_fc,
                                           clip_min=args.clip_min,
                                           clip_max=args.clip_max)
                elif args.adv == "GTA":

                    adv = A.GTA(images, rotate=30, translate=([0.01, 0.01]), scale=([0.8, 1.2]), rand=True, is_fc=is_fc)

                elif args.adv == "GNA":
                    # print('GNA Attack')
                    adv = A.GNA(images, clip_min=args.clip_min, clip_max=args.clip_max, _sigma=0.1, _mean=0.1,
                                    rand=True,
                                    is_fc=is_fc)
                elif args.adv=="Graphic":
                    raise Exception("implement graphic plz")#todo: this would break the code

                model.train()
                # print("adv.shape ",adv.shape)

                recon_images, mu, logvar = model(adv)
                adv = adv.detach()
                if is_fc:
                    losses = loss_fn_fc_each(recon_images, images, mu, logvar,args.KLweight)

                elif args.model =="cnn":
                    losses = model.loss_fn_each(recon_images, images, mu, logvar)

                replay_buffer.add(images.detach().numpy(), adv.numpy(), recon_images.detach().numpy(),
                                  losses.detach().numpy(),mu.detach().numpy(), logvar.detach().numpy(),"")
                optimizer.zero_grad()  # this might need to change.
                loss = torch.sum(losses)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # if idx % 100 == 0:
                #     print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / bs))
                #
                #     recon_x, _, _ = model(fixed_x)
                #     save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(),
                #                f'reconstructed_per/recon_image_{epoch}_{idx}.png')
                t = e * len(train_loader) + idx
                if t >= learning_starts and t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if args.prioritized_replay:
                        experience = replay_buffer.sample(args.bs_replaybuffer, beta=beta_schedule.value(t))
                        (xs, advs, recon_xs, losses, mus, vars,att_type, weights, batch_idxes) = experience
                    else:
                        # to do
                        xs, advs, recon_xs, losses, mus, vars,att_type, weights, batch_idxes = replay_buffer.sample(
                            args.bs)
                        #weights, batch_idxes = np.ones_like(rewards), None todo
                    xs      =  torch.tensor(xs).to(device)
                    advs    =  torch.tensor(advs).to(device)
                    losses  =  torch.tensor(losses).to(device)
                    weights =  torch.tensor(weights).to(device)
                    #print("b index ",batch_idxes)
                    td_errors,losses_new = train_pgd( optimizer, xs, advs, losses,  weights,is_fc,args.use_weight)
                    if args.prioritized_replay:
                        new_priorities = np.abs(td_errors.detach().numpy()) + prioritized_replay_eps

                        #print(new_priorities.shape)
                        replay_buffer.update_priorities(batch_idxes, new_priorities,losses_new.detach().numpy())

        else :
            # adversarial training using minimax. with replay buffer multi loss with critic
            # train(vae3,xs, advs, recon_xs, losses,mus,vars, weights)
            is_fc = args.model == "fc"
            if args.use_critic:
                critic = critic_FC()
                #raise Exception("implement critic")
            else:
                critic = critic_FC()
            opt_critic =  torch.optim.Adam(critic.parameters(), lr=args.lr)
            #args.best_size
            def train_critic(critic, xs, losses, is_fc, att_type, losses_new =None):
                #print("shit")
                losses_clone = losses.clone().detach()
                pred_losses = critic.forward(xs,att_type)
                # print(pred_losses.shape)
                # print(losses.shape)
                loss_fc_mse = torch.nn.MSELoss()
                real_loss = loss_fc_mse(pred_losses ,losses_clone)
                opt_critic.zero_grad()
                real_loss.backward()
                opt_critic.step()
            def select_best(pred_losses,best_size):
                arr = pred_losses.detach().numpy()
                #print("shit")
                return arr.argsort()[-best_size:]
            def train_pgd(opt, xs, advs, losses, weights, is_fc, use_weight,att_type):
                model.eval()


                if args.use_critic:
                    #print("shit")
                    pred_losses=critic.forward(xs,att_type)
                    new_idxes = select_best(pred_losses,args.best_size)
                    # print(new_idxes.shape)
                    new_idxes=new_idxes.reshape(args.best_size)
                    #print("new_idex shape 1 ",new_idxes.shape)
                    #replace the old ones.
                    xs = xs[new_idxes]
                    # print(xs.shape)
                    losses=losses[new_idxes]
                    weights = weights[new_idxes]
                    att_type = att_type[new_idxes] #todo: probably make this outside
                else:
                    new_idxes=-1
                adv_new = xs.clone()
                #print(att_type)
                if np.any(np.array(att_type=="FGSM").reshape(att_type.shape[0])):
                    adv_new[att_type=="FGSM"] = A.fgsm_untargeted(xs[att_type=="FGSM"], label=args.label, eps=args.eps, is_fc=is_fc,
                                                clip_min=args.clip_min,
                                                clip_max=args.clip_max)
                    #print("fgsm")
                if np.any(np.array(att_type=="PGD").reshape(att_type.shape[0])):
                    adv_new[att_type=="PGD"] = A.pgd_untargeted(xs[att_type=="PGD"], label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,
                                               is_fc=is_fc, clip_min=args.clip_min, clip_max=args.clip_max)

                    #print("pgd")
                if np.any(np.array(att_type=="GTA").reshape(att_type.shape[0])):
                    adv_new[att_type=="GTA"] = A.GTA(xs[att_type=="GTA"], rotate=30, translate=([0.01, 0.01]), scale=([0.8, 1.2]), rand=True, is_fc=is_fc)

                    #print("GTA")
                if np.any(np.array(att_type == "GNA").reshape(att_type.shape[0])):
                    adv_new[att_type=="GNA"] = A.GNA(xs[att_type=="GNA"], clip_min=args.clip_min, clip_max=args.clip_max, _sigma=0.1, _mean=0.1,
                                rand=True,
                                is_fc=is_fc)
                    #print("GNA")
                model.train()

                #print(adv_new.shape)
                recon_images, mu_new, logvar_new = model(adv_new)
                if (is_fc):
                    losses_new = loss_fn_fc_each(recon_images, xs, mu_new, logvar_new, args.KLweight)
                    if use_weight:
                        losses_new = weights * losses_new / torch.max(weights)
                else:  # assume is cnn
                    losses_new = model.loss_fn_each(recon_images, xs, mu_new, logvar_new)
                    if use_weight:
                        losses_new = weights * losses_new / torch.max(weights)

                opt.zero_grad()
                loss_new = torch.sum(losses_new)
                loss_new.backward()
                opt.step()
                # print("lossinpgd",losses.shape)
                # print(losses_new.shape)
                # todo: weights normalization and application.
                #todo: could add a attack class based annealing weight.
                # bias = 1/torch.tensor(args.probs)
                # add_bias = torch.zeros(losses_new.shape)
                # add_bias[att_type == "FGSM"] = bias[0]
                # add_bias[att_type == "PGD"] = bias[1]
                # added_bias = added_bias*1.0* (args.epochs-e)/args.epochs
                #todo: weighted bias for td error and just loss?
                if args.use_critic:
                    train_critic(critic,xs,losses,is_fc,att_type,losses_new=losses_new)
                    #TD_errors = critic.pred(xs,losses,att_type)
                    TD_errors=torch.abs(losses_new - losses)
                else:
                    TD_errors=losses_new - losses
                return torch.abs(TD_errors),losses_new,new_idxes

            n = len(args.multiadv)
            prob = args.probs if args.probs !=[] else [1.0/n]*n


            for idx, (images, _) in enumerate(train_loader):
                choice = np.random.choice(n,1,p=prob)[0]
                # print(choice,prob)
                if is_fc:
                    images = flatten(images)
                # print(images.shape) #128*28^2
                # build adv images.perform step on ad images.

                model.eval()

                if args.multiadv[choice] == "FGSM":
                    adv = A.fgsm_untargeted(images, label=args.label, eps=args.eps, is_fc=False,
                                            clip_min=args.clip_min,
                                            clip_max=args.clip_max)
                elif args.multiadv[choice] == "PGD":
                    adv = A.pgd_untargeted(images, label=args.label, k=args.k, eps=args.eps, eps_step=args.eps_step,
                                           is_fc=False,
                                           clip_min=args.clip_min,
                                           clip_max=args.clip_max)
                elif args.multiadv[choice] == "GTA":

                    adv = A.GTA(images, rotate=30, translate=([0.01, 0.01]), scale=([0.8, 1.2]), rand=True, is_fc=is_fc)

                elif args.multiadv[choice] == "GNA":
                    # print('GNA Attack')
                    adv = A.GNA(images, clip_min=args.clip_min, clip_max=args.clip_max, _sigma=0.1, _mean=0.1,
                                    rand=True,
                                    is_fc=is_fc)
                elif args.adv == "Graphic":
                    raise Exception("implement graphic plz")  # todo: this would break the code

                model.train()
                # print("adv.shape ",adv.shape)

                recon_images, mu, logvar = model(adv)
                adv = adv.detach()
                if is_fc:
                    losses = loss_fn_fc_each(recon_images, images, mu, logvar, args.KLweight)

                elif args.model == "cnn":
                    losses = model.loss_fn_each(recon_images, images, mu, logvar)

                replay_buffer.add(images.detach().numpy(), adv.numpy(), recon_images.detach().numpy(),
                                  losses.detach().numpy(), mu.detach().numpy(), logvar.detach().numpy(),args.multiadv[choice])
                optimizer.zero_grad()  # this might need to change.

                if args.use_critic:
                    train_critic(critic,images,losses,is_fc,np.array([args.multiadv[choice]]*args.bs))
                loss = torch.sum(losses)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # if idx % 100 == 0:
                #     print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / bs))
                #
                #     recon_x, _, _ = model(fixed_x)
                #     save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(),
                #                f'reconstructed_per/recon_image_{epoch}_{idx}.png')
                t = e * len(train_loader) + idx
                if t >= learning_starts and t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if args.prioritized_replay:
                        experience = replay_buffer.sample(args.bs_replaybuffer, beta=beta_schedule.value(t))
                        (xs, advs, recon_xs, losses, mus, vars,att_type, weights, batch_idxes) = experience
                    else:
                        # to do
                        xs, advs, recon_xs, losses, mus, vars,att_type, weights, batch_idxes = replay_buffer.sample(
                            args.bs)
                        # weights, batch_idxes = np.ones_like(rewards), None todo
                    xs      = torch.tensor(xs).to(device)
                    advs    = torch.tensor(advs).to(device)
                    losses  = torch.tensor(losses).to(device)
                    weights = torch.tensor(weights).to(device)
                    # print("b index ",batch_idxes)
                    td_errors,losses_new,new_idxes = train_pgd(optimizer, xs, advs, losses, weights, is_fc, args.use_weight,att_type=att_type)
                    if args.use_critic:
                        # print("new_idex shape 2 ",new_idxes.shape)
                        # print("batch idex shape 1 ",batch_idxes.shape)
                        batch_idxes=[batch_idxes[i] for i in new_idxes]
                    if args.prioritized_replay:
                        new_priorities = np.abs(td_errors.detach().numpy()) + prioritized_replay_eps
                        # print(new_priorities.shape)
                        replay_buffer.update_priorities(batch_idxes, new_priorities,losses_new.detach().numpy())
        # else:
        #     pass

        return train_loss

    def validate(e):
        # set the evaluation mode
        model.eval()



        n = len(args.metric_valid)
        # test loss for the data
        test_loss = [0] * n

        # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing

        for i, (x, x_label) in enumerate(val_loader):
            # reshape the data
            # x = x.view(-1, 28 * 28)
            x = x.to(device)
            if args.model == "fc":
                x = flatten(x)
            for j, m in enumerate(args.metric_valid):
                #print(j,"validation ",m)
                if m == "original":
                    # forward pass
                    x_recon, mu, var = model(x)

                    if args.model == "cnn":
                        loss = model.loss_fn(x_recon, x, mu, var)
                    elif args.model == "fc":
                        loss = loss_fn_fc(x_recon, x, mu, var, args.KLweight)

                    #test_loss[j] += loss.item()
                elif m == "FGSM":

                    if args.model == "fc":
                        x_flat = flatten(x)
                        # print(images.shape) #128*28^2
                        # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        adv = A.fgsm_untargeted( x_flat, label=7, eps=0.15, clip_min=0,
                                                     clip_max=1.0)
                        # model.train()
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar, args.KLweight)

                    if args.model == "cnn":
                        adv = A.fgsm_untargeted( x, label=7, eps=0.15, is_fc=False,
                                                     clip_min=0,
                                                     clip_max=1.0)  # do something here.
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var)

                    #test_loss[j] += loss.item()
                elif m == "PGD":
                    if args.model == "fc":
                        x_flat = flatten(x)
                        # print(images.shape) #128*28^2
                        # create adv images.perform step on the reconstructed image generated by adv images.

                        model.eval()
                        adv = A.pgd_untargeted( x_flat, label=7, k=10, eps=0.08, eps_step=0.05,
                                                    clip_min=0,
                                                    clip_max=1.0)
                        # model.train()
                        # print("adv.shape ",adv.shape)

                        recon_images, mu, logvar = model(adv)
                        # print(recon_images.shape)
                        adv = adv.detach()
                        loss = loss_fn_fc(recon_images, x_flat, mu, logvar, args.KLweight)

                    if args.model == "cnn":
                        adv = A.pgd_untargeted( x, label=7, k=10, eps=0.08, eps_step=0.05, is_fc=False,
                                                    clip_min=0,
                                                    clip_max=1.0)  # do something here.
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var)


                elif m == "GTA":
                    if args.model == "fc":
                        adv = A.GTA(x, rotate=30, translate=([0.01, 0.01]), scale=([0.8, 1.2]), rand=True, is_fc=True)
                        x_recon, mu, var = model(adv)
                        loss = loss_fn_fc(x_recon, x, mu, var,args.KLweight)
                    elif args.model == "cnn":
                        adv = A.GTA(x, rotate=30, translate=([0.01, 0.01]), scale=([0.8, 1.2]), rand=True, is_fc=False)
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var)

                elif m == "GNA":
                    if args.model == "fc":
                    # print('GNA Attack')
                        adv = A.GNA(x, clip_min=args.clip_min, clip_max=args.clip_max, _sigma=0.1, _mean=0.1,
                                    rand=True,
                                    is_fc=True)
                        x_recon, mu, var = model(adv)
                        loss = loss_fn_fc(x_recon, x, mu, var,args.KLweight)
                    elif args.model == "cnn":
                        adv = A.GNA(x, clip_min=args.clip_min, clip_max=args.clip_max, _sigma=0.1, _mean=0.1,
                                    rand=True,
                                    is_fc=False)
                        x_recon, mu, var = model(adv)
                        loss = model.loss_fn(x_recon, x, mu, var)
                test_loss[j] += loss.item()
        return test_loss
    train_losses=[]
    val_losses = []
    if not os.path.exists(f"/content/drive/MyDrive/recon_images/{d}_{fileindex}"):
        os.mkdir(f"/content/drive/MyDrive/recon_images/{d}_{fileindex}")
    print(f"Model number: {d}_{fileindex}")
    for e in range(args.epochs):

        train_loss = train(e)
        val_loss = validate(e)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_loss /= len(train_dataset)
        for i in range(len(val_loss)):
            val_loss[i] /= len(valid_dataset)
        #assume the first validation loss is used for early stopping.
        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {val_loss}')

        if best_valid_loss > val_loss[0]:
            # assuming connected to google drive
            if not os.path.exists(f"/content/drive/MyDrive/model_weights/{d}_{fileindex}"):
                os.mkdir(f"/content/drive/MyDrive/model_weights/{d}_{fileindex}")

            model_save_name = f'vae_cur_{args.model}_{e}_{d}.pt'
            path = f"/content/drive/MyDrive/model_weights/{d}_{fileindex}/{model_save_name}"
            torch.save(model.state_dict(), path)
            best_valid_loss = val_loss[0]
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > 3:
            break
        recon_x, _, _ = model(fixed_x)

        save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(),
                   f'/content/drive/MyDrive/recon_images/{d}_{fileindex}/vae_{args.model}_{e}_{d}.png')
    train_losses=np.asarray(train_losses)
    val_losses  =np.asarray(val_losses)
    if not os.path.exists(f"/content/drive/MyDrive/model_results/{d}_{fileindex}"):
        os.mkdir(         f"/content/drive/MyDrive/model_results/{d}_{fileindex}")
    if not os.path.exists(f"/content/drive/MyDrive/model_para/{d}_{fileindex}"):
        os.mkdir(f"/content/drive/MyDrive/model_para/{d}_{fileindex}")
    np.savetxt(f"/content/drive/MyDrive/model_results/{d}_{fileindex}/{model_save_name}_train", train_losses, delimiter=',')
    np.savetxt(f"/content/drive/MyDrive/model_results/{d}_{fileindex}/{model_save_name}_val", val_losses,
               delimiter=',')
    #print("Hello World!")

if __name__ == "__main__":
    main()