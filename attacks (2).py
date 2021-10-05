import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torchvision import transforms
import random



#Fix seeds
random.seed(2020)


def flatten(x):
    return to_var(x.view(x.size(0), -1))
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
class Attacks(object):
    def __init__(self,model):
        self.model = model
    def fgsm_(self, x, target, eps, targeted=True,is_fc= True, clip_min=None, clip_max=None):
        """Internal process for all FGSM and PGD attacks."""
        # create a copy of the input, remove all previous associations to the compute graph...

        input_ = x.clone().detach_()
 
        # ... and make sure we are differentiating toward that variable
        if is_fc:
            input_ = flatten(input_)

        input_.requires_grad_()
        # print(is_fc)
        # print(input_.shape)
        # run the model and obtain the loss
        recon_x = self.model(input_)[0]
        # print(recon_x.shape)
        # print(input_.shape)

        self.model.zero_grad()
        # print("test backward 0")
        loss = F.binary_cross_entropy(input_, recon_x.detach(), size_average=False)
        # print("test backward 1")
        # loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # print("test backward")

        # perfrom either targeted or untargeted attack
        if targeted:
            out = input_ - eps * input_.grad.sign()
        else:
            out = input_ + eps * input_.grad.sign()
        # print("test out")
        # if desired clip the ouput back to the image domain
        if (clip_min is not None) or (clip_max is not None):
            out.clamp_(min=clip_min, max=clip_max)
        return out


    def fgsm_targeted(self, x, target, eps,is_fc=True, **kwargs):
        return self.fgsm_( x, target, eps, targeted=True,is_fc=is_fc, **kwargs)


    def fgsm_untargeted(self, x, label, eps,    is_fc=True, **kwargs):
        return self.fgsm_( x, label, eps, targeted=False, is_fc=is_fc,**kwargs)


    # k: iterations, x:original image, target: class label,
    # eps:  boundary, eps_step: step size,
    # clip_min: range for gray scale pixel value 0 to 1
    def pgd_(self, x, target, k, eps, eps_step, targeted=True,is_fc=True, clip_min=None, clip_max=None):

        if is_fc:
            x = flatten(x)
        x_min = x - eps
        x_max = x + eps

        # Randomize the starting point x.
        x = x + eps * (2 * torch.rand_like(x) - 1)
        if (clip_min is not None) or (clip_max is not None):
            x.clamp_(min=clip_min, max=clip_max)
        # assert np.prod(x.detach().numpy()>=0) and np.prod(x.detach().numpy()<=0)
        for i in range(k):
            # FGSM step
            # We don't clamp here (arguments clip_min=None, clip_max=None)
            # as we want to apply the attack as defined
            x = self.fgsm_(x, target, eps_step, targeted,is_fc)
            # Projection Step
            x = torch.max(x_min, x)
            x = torch.min(x_max, x)
            # assert np.prod(x.detach().numpy()>=0) and np.prod(x.detach().numpy()<=0)
            # if desired clip the ouput back to the image domain
            # note: the clamping is changed so its done in every loop to avoid stepping out of bound,
            # todo: not sure if this makes the algorithm less effective
            if (clip_min is not None) or (clip_max is not None):
                x.clamp_(min=clip_min, max=clip_max)
        return x


    def pgd_targeted(self, x, target, k, eps, eps_step,is_fc=True, clip_min=None, clip_max=None, **kwargs):
        return self.pgd_(x, target, k, eps, eps_step, targeted=True,is_fc=is_fc, **kwargs)


    def pgd_untargeted(self, x, label, k, eps, eps_step,is_fc=True, clip_min=None, clip_max=None, **kwargs):
        return self.pgd_(x, label, k, eps, eps_step, targeted=False,is_fc=is_fc, clip_min=clip_min, clip_max=clip_max, **kwargs)

    # Define Gaussian noise function
    def GNA(self, x,  clip_min=None, clip_max=None,_sigma=0.1,_mean=0.1,rand=False,is_fc=True):
        """Gaussian Noise Attack:
        @Param
          sigma: standard deviation of gaussian noise
          mean: mean of gaussion noise
          random: If random = true, sdt_dev is in range from 0 to sigma and mean in range of +- mean"""    
        # create a copy of the input, remove all previous associations to the compute graph...
        input_ = x.clone().detach_()
        # ... and make sure we are differentiating toward that variable
        if is_fc:
            input_ = flatten(input_)
        input_.requires_grad_()
        recon_x = self.model(input_)[0]
        # print(recon_x.shape)
        #print(input_.shape)

        self.model.zero_grad()
        # print("test backward 0")
        loss = F.binary_cross_entropy(input_, recon_x.detach(), size_average=False)
        # print("test backward 1")
        # loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        
        if rand:
          sigma = random.random()*_sigma
          mean = 2*random.random()*_mean-_mean
        else:
          sigma = _sigma
          mean = _mean
        #Calculate the gaussian noise and add it to the output
        out = torch.add(input_, mean) + torch.randn(input_.size())*sigma
    
        #if desired clip the ouput back to the image domain
        if (clip_min is not None) or (clip_max is not None):
            out.clamp_(min=clip_min, max=clip_max)
        #print(out.shape)
        return out
    

    #Define Geometric Transformation
    # 1 Geometric robustness of deep networks Kanbak, Moosavi et al., CVPR 2018.
    def GTA(self,x,rotate=30,translate=([0.01,0.01]),scale = ([0.8,1.2]),rand=True,is_fc=True):
        """Geometric Transformation Attack:
        @Param
          rotate: rotational angle of image in degree : range (+-) if random = true, otherwise fixed
          translate : pixel to translate image, fixed in pixels [x,y] if random = false, otherwise range in percentage [x%,y%](+-) 
          scale: range for random (percentage) [min,max], value for fixed"""    
        # create a copy of the input, remove all previous associations to the compute graph...
        input_ = x.clone().detach_()
        if is_fc:
            input_ = flatten(input_)
            
        input_.requires_grad_()
        recon_x = self.model(input_)[0]
        # print(recon_x.shape)
        # print(input_.shape)

        self.model.zero_grad()
        # print("test backward 0")
        loss = F.binary_cross_entropy(input_, recon_x.detach(), size_average=False)
        # print("test backward 1")
        # loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        trans = transforms.Compose([
              transforms.RandomAffine(rotate, translate=translate, scale=scale)
              ])
    
        #Random/ fixed value geometric transformation
        if rand:
            if not is_fc:
           
              out = trans(input_)
            else:
                #trans = transforms.RandomRotation(degrees = rotate, resample=False, expand=False, center=None, fill=None)
                #out = trans.forward(torch.reshape(input_,(128,1,28,28)))
                #print(f'inputshape{input_.shape}')
                out = trans(torch.reshape(input_,(input_.shape[0],1,28,28)))
                out = torch.reshape(out,(input_.shape[0],784))
                #print(out.shape)
                #print(out.shape)
        else:
          out = transforms.functional.affine(input_,angle = rotate,translate=translate,scale=scale,shear=0)
    
        return out
            
    def VAEA(self, x, target, chosen_norm, steps, eps, eps_norm, step_size=0.01, clamp=(0,1), delta= 0.1):
    
        # create a copy of the input, remove all previous associations to the compute graph...
        x_star = x.clone().detach_()
        target_ = target.clone().detach_()

        # ... and make sure we are differentiating toward that variable
        x_star = flatten(x_star)
        x_origin = x_star
        x_star.requires_grad_()
        x_star.retain_grad()
        target_ = flatten(target_)
        target_.requires_grad_(False)
        
        # run the model and obtain the loss
        recon_t = self.model(target_)[0]
        recon_t = recon_t.detach()
        
        output = x_star
        # use PGD to create adverserial image
        for step in range(steps):
            x_star = output.clone().detach_()
            x_star.requires_grad_()
            
            (recon_x_star, mu, logvar)  = self.model(x_star)
            
            BCE = F.binary_cross_entropy(recon_x_star, recon_t, size_average=False)
            KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
            loss_2 = torch.nn.BCELoss() #Changed (TE)
            print(f'x_star:{x_star.shape} x:{x.shape} KLD: {KLD.shape} BCE: {BCE.shape}')
            loss = delta * loss_2(x_star, x) + KLD + BCE
    
            loss.backward()
            #print(x_star)
            #print(x_star.grad)

            #calculate gradient
            num_channels = x_star.shape[1]
            if chosen_norm == 'infinite':
                gradient = x_star.grad.sign() * step_size
            else:
                # acctepted norms:  https://pytorch.org/docs/stable/generated/torch.norm.html
                gradient = x_star.grad * step_size / x_star.grad.view(x_star.shape[0], -1)\
                .norm('None')\
                .view(-1, num_channels, 1, 1)
            
            # Apply gradient
            output = x_star + gradient
            
            # projection
            if eps_norm == 'infinite':
                output = torch.max(torch.min(output, x + eps), x - eps)
            else:
                diff = output - x
                mask = diff.view(diff.shape[0], -1).norm('None', dim=1) <= eps
                scaling_factor = diff.view(diff.shape[0], -1).norm('None', dim=1)
                scaling_factor[mask] = eps
                diff *= eps / scaling_factor.view(-1, 1, 1, 1)
                output = x + diff
            
        return output.detach()
    
    def LatentA(self,  x, target, chosen_norm, steps, eps, eps_norm,step_size=0.01, clamp=(0,1), delta= 0.1):
        output = x #Think this should be initialized (Tobias)
        x_star = x.clone().detach_()
        target_ = target.clone().detach_()
        # ... and make sure we are differentiating toward that variable
        x_star = flatten(x_star)
        x_star.requires_grad_()
        target_ = flatten(target_)
        target_.requires_grad_(False)
        
        num_channels = x_star.shape[1]
        
        # get latent representation
        z_t = self.model.latent(target_)[0]
        z_t.requires_grad_(False)
        # use PGD to create adverserial image
        for step in range(steps):
            x_star = output.clone().detach_()
            x_star.requires_grad_()
            enc_x_star = self.model.latent(x_star)[0]
            
            # calculate loss depending on latent representation
            loss_2 = torch.nn.BCELoss() 
            loss = delta* loss_2(x, x_star) + loss_2(enc_x_star, z_t)
            loss.backward()
            
            with torch.no_grad():
                #calculate gradient
                if chosen_norm == 'infinite':
                    gradients = x_star.grad.sign() * step_size
                else:
                    # acctepted norms:  https://pytorch.org/docs/stable/generated/torch.norm.html
                    gradient = x_star.grad * step_size / x_star.grad.view(x_star.shape[0], -1)\
                    .norm(chosen_norm)\
                    .view(-1, num_channels, 1, 1)
            
            # Apply gradient
            output += gradient
            
            # projection
            if eps_norm == 'infinite':
                output = torch.max(torch.min(output, x + eps), x - eps)
            else:
                diff = x_star - x
                mask = diff.view(diff.shape[0], -1).norm(eps_norm, dim=1) <= eps
                scaling_factor = diff.view(diff.shape[0], -1).norm(eps_norm, dim=1)
                scaling_factor[mask] = eps
                
                diff *= eps / scaling_factor.view(-1, 1, 1, 1)
                output = x + diff

        return output.detach()