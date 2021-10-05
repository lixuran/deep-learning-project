# -*- coding: utf-8 -*-
"""vae_fc_mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UXMimxx71UsQchCbOmfGDlaUbGJCGaPd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from IPython.display import Image

from google.colab import drive
drive.mount('/content/drive')

!cp drive/MyDrive/"Colab Notebooks"/replay_buffer_v2.py .

!cp drive/MyDrive/"Colab Notebooks"/segment_tree.py .

import segment_tree
from replay_buffer_v2 import ReplayBuffer, PrioritizedReplayBuffer

import numpy as np

np.random.seed(2020)
torch.manual_seed(2020)

import os
path = os.getcwd()
print(path)
os.mkdir(path+"/reconstructed")
os.mkdir(path+"/reconstructed_ad_training")
os.mkdir(path+"/reconstructed_per")
os.mkdir(path+"/reconstructed_per_multi")

bs = 128
epochs = 10

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

"""# 新段落"""

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def flatten(x):
    return to_var(x.view(x.size(0), -1))

def save_image(x, path='real_image.png'):
    torchvision.utils.save_image(x, path)

# Load Data
dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

# Fixed input for debugging
fixed_x, _ = next(iter(data_loader))
save_image(fixed_x)
fixed_x = flatten(fixed_x)

Image('real_image.png')

class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

vae = VAE()
if torch.cuda.is_available():
    vae.cuda()
vae

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    #print("bce shape ",BCE.shape)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD

epochs = 10

for epoch in range(epochs):
    for idx, (images, _) in enumerate(data_loader):
        images = flatten(images)
        #print(images.shape)
        recon_images, mu, logvar = vae(images)
        #print(recon_images.shape)
        loss = loss_fn(recon_images, images, mu, logvar)
        #print(loss)
        #print()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx%100 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs))
    
            recon_x, _, _ = vae(fixed_x)
            save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed/recon_image_{epoch}_{idx}.png')

sample = Variable(torch.randn(128, 20))
recon_x = vae.decoder(sample)
# recon_x, _, _ = vae(fixed_x)

save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), 'sample_image.png')
Image('sample_image.png')

images = []
for file in sorted([file for file in path('reconstructed').glob('*.png')]):
    images.append(imageio.imread(file))
imageio.mimsave('recon_image.gif', images)
Image(filename="recon_image.gif")

def fgsm_(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_ = flatten(input_)
    input_.requires_grad_()

    # run the model and obtain the loss
    recon_x = model(input_)[0]
    #print(recon_x.shape)
    #print(input_.shape)
    
    
    model.zero_grad()
    #print("test backward 0")
    loss = F.binary_cross_entropy(input_,recon_x.detach(), size_average=False)
    #print("test backward 1")
    #loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    #print("test backward")

    #perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()
    #print("test out")
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out

def fgsm_targeted(model, x, target, eps, **kwargs):
    return fgsm_(model, x, target, eps, targeted=True, **kwargs)

def fgsm_untargeted(model, x, label, eps, **kwargs):
    return fgsm_(model, x, label, eps, targeted=False, **kwargs)

# k: iterations, x:original image, target: class label, 
# eps:  boundary, eps_step: step size, 
# clip_min: range for gray scale pixel value 0 to 1
def pgd_(model, x, target, k, eps, eps_step, targeted=True, clip_min=None, clip_max=None):
    x = flatten(x)
    x_min = x - eps
    x_max = x + eps
    
    # Randomize the starting point x.
    x = x + eps * (2 * torch.rand_like(x) - 1)
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    #assert np.prod(x.detach().numpy()>=0) and np.prod(x.detach().numpy()<=0)
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None) 
        # as we want to apply the attack as defined
        x = fgsm_(model, x, target, eps_step, targeted)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
        #assert np.prod(x.detach().numpy()>=0) and np.prod(x.detach().numpy()<=0)
        #if desired clip the ouput back to the image domain 
        #note: the clamping is changed so its done in every loop to avoid stepping out of bound, 
        #todo: not sure if this makes the algorithm less effective
        if (clip_min is not None) or (clip_max is not None):
          x.clamp_(min=clip_min, max=clip_max)
    return x

def pgd_targeted(model, x, target, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, target, k, eps, eps_step, targeted=True, **kwargs)

def pgd_untargeted(model, x, label, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, label, k, eps, eps_step, targeted=False,clip_min=clip_min,clip_max=clip_max, **kwargs)

"""# Attacks of Tobias Elmiger:


*   Gaussian Noise Attack: 
  * Function: GNA(model, x,  clip_min=None, clip_max=None,sigma=0.1,mean=0)
*   Geometric Transformation Attack

  (Random or fixed value is possible):
    * GTA(vae, original, rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),random=True) #Random
    * GTA(vae, original, rotate=30,translate=([0,0]),scale = 1.2,random=False) #fixed

"""

#Define Gaussian noise function
def GNA(model, x,  clip_min=None, clip_max=None,sigma=0.1,mean=0):
    """Gaussian Noise Attack:
    @Param
      sigma: standard deviation of gaussian noise
      mean: mean of gaussion noise"""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_ = flatten(input_)
    input_.requires_grad_()

    
    #Calculate the gaussian noise and add it to the output
    out = torch.add(input_, mean) + torch.randn(input_.size())*sigma

    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


#Define Geometric Transformation
# 1 Geometric robustness of deep networks Kanbak, Moosavi et al., CVPR 2018.
def GTA(model, x,rotate=0,translate=([0,0]),scale = None,random=False):
    """Geometric Transformation Attack:
    @Param
      rotate: rotational angle of image in degree : range (+-) if random = true, otherwise fixed
      translate : pixel to translate image, fixed in pixels [x,y] if random = false, otherwise range in percentage [x%,y%](+-) 
      scale: range for random (percentage) [min,max], value for fixed"""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()

    #Random/ fixed value geometric transformation
    if random:
      trans = transforms.Compose([
      transforms.RandomAffine(rotate, translate=translate, scale=scale)
      ])
      out = trans(input_)
    else:
      out = transforms.functional.affine(input_,angle = rotate,translate=translate,scale=scale,shear=0)

    return out

vae.eval()
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)
# try out our attacks
original = torch.unsqueeze(test_dataset[0][0], dim=0)

adv = fgsm_untargeted(vae, original, label=7, eps=0.25, clip_min=0, clip_max=1.0)
#adv = fgsm_targeted(model, original, target=3, eps=0.2, clip_min=0, clip_max=1.0)

#adv = pgd_untargeted(model, original, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)
#adv = pgd_targeted(model, original, target=8, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)

#show(original, adv, model_to_prob)

# Attacks of Tobias Elmiger
#Run Gaussian noise Attack
#adv = GNAvae, original,clip_min=0, clip_max=1.0, sigma = 0.3)

#Run Geometric Transformation attack: Either random or fixed is possible
#adv = GTA(vae, original, rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),random=True) #Random
#adv = GTA(vae, original, rotate=30,translate=([0,0]),scale = 1.2,random=False) #fixed

original.shape

test_dataset[0][0].shape

import matplotlib.pyplot as plt

adv2= adv.reshape(1,28,28)
adv2 = adv2.detach().numpy().reshape(28, 28)
input_ = adv.clone().detach_()
input_ = flatten(input_)
recon_x = vae(input_)[0]
loss_adv =  F.binary_cross_entropy(input_,recon_x.detach(), reduction="mean")
# loss of FGSM attack image
print(loss_adv)

plt.imshow(adv2, cmap='gray')

plt.imshow(recon_x.detach().numpy().reshape(28,28), cmap='gray')

ori2 = test_dataset[0][0].reshape(28,28)
plt.imshow(ori2, cmap='gray')

vae.eval()
input_ = original.clone().detach_()
# ... and make sure we are differentiating toward that variable
input_ = flatten(input_)
recon_x_ori = vae(input_)[0]
loss_original =  F.binary_cross_entropy(input_,recon_x_ori.detach(), reduction="mean")
#original binary cross entropy loss of the sample image. 
#note: why is the loss so small compared with the loss reported during training?
# is it this loss *28^2 * 128?
print(loss_original)

plt.imshow(recon_x_ori.detach().numpy().reshape(28,28), cmap='gray')

adv = pgd_untargeted(vae, original, label=7, k=50, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)

adv2= adv.reshape(1,28,28)
adv2 = adv2.detach().numpy().reshape(28, 28)
input_ = adv.clone().detach_()
# ... and make sure we are differentiating toward that variable
input_ = flatten(input_)
recon_x = vae(input_)[0]
loss_adv =  F.binary_cross_entropy(input_,recon_x.detach(), reduction="mean")
# print loss of pgd untargetd attack on sample image
print(loss_adv)

plt.imshow(adv2, cmap='gray')
# adversarial image of pgd untargeted on mnist. visible background noise. would be
# less noticable if attack on cifar-10.

plt.imshow(recon_x.detach().numpy().reshape(28,28), cmap='gray')

# adversarial training using minimax. without replay buffer
#next step: save the adv images. to check.
epochs = 10
vae2 = VAE()
#vae2 = vae # bruh the starting point is important???? no its not niceeeeeeee
        # but could potentially be helpful.
optimizer2 = torch.optim.Adam(vae2.parameters(), lr=1e-3)
for epoch in range(epochs):
    for idx, (images, _) in enumerate(data_loader):
        images = flatten(images)
        #print(images.shape) #128*28^2
        #create adv images.perform step on the reconstructed image generated by adv images.
       
        vae2.eval()
        adv = pgd_untargeted(vae2, images, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)
        vae2.train()
        #print("adv.shape ",adv.shape)
       
        recon_images, mu, logvar = vae2(adv)
        #print(recon_images.shape)
        adv=adv.detach()
        loss = loss_fn(recon_images, images, mu, logvar)
        #loss2 = loss_fn(recon_images, adv, mu, logvar)
        #print(loss)
        #print(loss2)
        #very important to clear the grad both in inner and outer loop.
        optimizer2.zero_grad() 
        loss.backward()
        optimizer2.step()
        
        if idx%100 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs))
    
            recon_x, _, _ = vae2(fixed_x)
            save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed_ad_training2/recon_image_{epoch}_{idx}.png')

# adversarial training using minimax. with replay buffer single loss 
buffer_size = 6400 # todo: this number is not optimal
learning_starts =5   # learning_starts batches are to be stored in the replay buffer before sampling starts 
train_freq = 1
prioritized_replay_alpha = 0.6
prioritized_replay_eps  = 1e-6 #todo: change this
prioritized_replay_beta0 = 0.4
epochs = 20
total_timesteps = epochs*len(data_loader)
bs =128
if prioritized_replay:
    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None:
        prioritized_replay_beta_iters = total_timesteps
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,initial_p=prioritized_replay_beta0,final_p=1.0)
else:
    replay_buffer = ReplayBuffer(buffer_size)
    beta_schedule = None



vae3 = VAE()
optimizer3 = torch.optim.Adam(vae3.parameters(), lr=1e-3)

#train(vae3,xs, advs, recon_xs, losses,mus,vars, weights)
def train_pgd(model,opt,xs,advs,recon_xs,losses,mus,vars,weights):
    model.eval()
    adv = pgd_untargeted(model, xs, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)
    model.train()
    recon_images, mu, logvar = model(adv)
    loss_new = loss_fn(recon_images, xs, mu, logvar)
    opt.zero_grad() 
    loss_new.backward()
    opt.step()
    print("loss new shape",loss_new.shape)
    return loss_new - losses
for epoch in range(epochs):
    for idx, (images, _) in enumerate(data_loader):
        images = flatten(images)
        #print(images.shape) #128*28^2
        #build adv images.perform step on ad images.
        vae3.eval()
        adv = pgd_untargeted(vae3, images, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)
        vae3.train()
        #print("adv.shape ",adv.shape)
        
        recon_images, mu, logvar = vae3(adv)
        adv=adv.detach()
        loss = loss_fn(recon_images, images, mu, logvar)
        replay_buffer.add(images, adv, recon_images, loss)
        optimizer3.zero_grad() # this might need to change.
        loss.backward()
        optimizer3.step()
        
        if idx%100 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs))
    
            recon_x, _, _ = vae3(fixed_x)
            save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed_per/recon_image_{epoch}_{idx}.png')
        t = epoch*len(data_loader) +idx
        if t >= learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (xs, advs, recon_xs, losses,mus,vars,weights,idxes) = experience
            else:
                #to do
                xs, advs, recon_xs, losses,mus,vars,weights,batch_idxes = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = train_pgd(vae3,optimizer3,xs, advs, recon_xs, losses,mus,vars, weights)
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)


#todo: define train td_errors =abs( loss(vae3(ad),xs) - loss ) ?

os.mkdir(path+"/reconstructed_per_multi")

# adversarial training using minimax. with replay buffer, multiple loss 
# todo: make it so that u can specify how many adversarial there is.
#os.mkdir(path+"/reconstructed_per_multi")
buffer_size_pgd   = 6400 # todo: this number is not optimal
buffer_size_fgsm   = 6400
learning_starts_pgd = 5   # learning_starts batches are to be stored in the replay buffer before sampling starts 
train_freq_pgd   = 1
learning_starts_fgsm = 5   # learning_starts batches are to be stored in the replay buffer before sampling starts 
train_freq_fgsm   = 1
prioritized_replay_alpha = 0.6
prioritized_replay_eps  = 1e-6 
prioritized_replay_beta0 = 0.4
epochs = 20
total_timesteps = epochs*len(data_loader)
bs =128
def create_buffer(buffer_size,prioritized_replay_alpha,prioritized_replay_beta0,prioritized_replay,total_timesteps,final_p=1.0,prioritized_replay_beta_iters=None)
  if prioritized_replay:
      replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
      if prioritized_replay_beta_iters is None:
          prioritized_replay_beta_iters = total_timesteps
      beta_schedule = LinearSchedule(prioritized_replay_beta_iters,initial_p=prioritized_replay_beta0,final_p=final_p)
  else:
      replay_buffer = ReplayBuffer(buffer_size)
      beta_schedule = None
  return replay_buffer,beta_schedule

replay_buffer_1,beta_schedule_1 = create_buffer(buffer_size,prioritized_replay_alpha,prioritized_replay_beta0,prioritized_replay,total_timesteps,final_p=1.0)
replay_buffer_2,beta_schedule_2 = create_buffer(buffer_size,prioritized_replay_alpha,prioritized_replay_beta0,prioritized_replay,total_timesteps,final_p=1.0)

vae3 = VAE()
optimizer3 = torch.optim.Adam(vae3.parameters(), lr=1e-3)

#train(vae3,xs, advs, recon_xs, losses,mus,vars, weights)

#todo: use huber loss and reduce mean??
def train_pgd(model,opt,xs,advs,recon_xs,losses,mus,vars,weights):
    model.eval()
    # todo : eps eps_step and k should be iterated.
    adv = pgd_untargeted(model, xs, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)
    model.train()
    recon_images, mu, logvar = model(adv)
    loss_new = loss_fn(recon_images, xs, mu, logvar)
    opt.zero_grad() 
    loss_new.backward()
    opt.step()
    print("pgd loss new shape",loss_new.shape)
    return loss_new - losses
def train_FGSM(model,opt,xs,advs,recon_xs,losses,mus,vars,weights):
    model.eval()
    # todo : eps should be iterated.
    adv = fgsm_untargeted(model, xs, label=7, eps=0.15, clip_min=0, clip_max=1.0)
    model.train()
    recon_images, mu, logvar = model(adv)
    loss_new = loss_fn(recon_images, xs, mu, logvar)
    opt.zero_grad() 
    loss_new.backward()
    opt.step()
    print("fgsm loss new shape",loss_new.shape)
    return loss_new - losses
for epoch in range(epochs):
    for idx, (images, _) in enumerate(data_loader):
        images = flatten(images)
        #print(images.shape) #128*28^2
        #build adv images.perform step on ad images.
        vae3.eval()
        #todo: dicede which attack to use.
        adv = pgd_untargeted(vae3, images, label=7, k=10, eps=0.08, eps_step=0.05, clip_min=0, clip_max=1.0)
        vae3.train()
        #print("adv.shape ",adv.shape)
        
        recon_images, mu, logvar = vae3(adv)
        adv=adv.detach()
        loss = loss_fn(recon_images, images, mu, logvar)
        replay_buffer.add(images, adv, recon_images, loss)
        optimizer3.zero_grad() # this might need to change.
        loss.backward()
        optimizer3.step()
        
        if idx%100 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs))
    
            recon_x, _, _ = vae3(fixed_x)
            save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed_per_multi/recon_image_{epoch}_{idx}.png')
        t = epoch*len(data_loader) +idx
        if t >= learning_starts_pgd and t % train_freq_pgd == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer_1.sample(batch_size, beta=beta_schedule_1.value(t))
                (xs, advs, recon_xs, losses,mus,vars,weights,idxes) = experience
            else:
                #to do
                xs, advs, recon_xs, losses,mus,vars,weights,batch_idxes = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = train_pgd(vae3,optimizer3,xs, advs, recon_xs, losses,mus,vars, weights)
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)


#todo: define train td_errors =abs( loss(vae3(ad),xs) - loss ) ?

len(data_loader)