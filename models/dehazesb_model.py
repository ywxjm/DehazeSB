import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from omegaconf import OmegaConf
# from models.ldm.util import instantiate_from_config
from models.ssim_loss import ssim_criterion
from models.utils.dcp_generate import dark_channel_generate
from models.utils.haze_util import synthesize_fog
import lpips
from models.utils.hfp import extract_high_freq_component, sobel_operator
import torch.nn.functional as F
from models.clip_guide.prompt_initial import Prompts, TextEncoder
from models.clip_guide.CLIP import clip
from models.clip_guide import clip_score
from models import vision_aided_loss
# from models.networks import TimeEmbedding, get_timestep_embedding
# from DSCNet import TimeEmbedding, get_timestep_embedding
model, preprocess = clip.load('ViT-B/32', device=torch.device("cpu"))  # ViT-B/32
model.to('cuda')

for para in model.parameters():
    para.requires_grad = False

def _get_hf_map(rgb):
    # convert to 0-1
    rgb = (rgb + 1.0) / 2.0
    rgb = rgb.to(torch.float32)  # can not on torch.float16
    assert rgb.dim() == 4, "Input must be 4D tensor"
    high_freq_image = extract_high_freq_component(rgb, cutoff=30)
    sobel_edges = sobel_operator(rgb)
    return high_freq_image, sobel_edges

class dehazesbModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for SB model
        """

        parser.add_argument('--length_prompt', type=int, default=16)
        parser.add_argument('--mode', type=str, default="sbunetdcp")
        parser.add_argument('--use_prompt', type=bool, default=True)
        parser.add_argument('--lambda_hfp', type=float, default=0.5, help='weight for GAN loss：DCP')
        parser.add_argument('--lambda_dcp', type=float, default=0.5, help='weight for GAN loss：DCP')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=1.0, help='weight for SB loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        ### additional ssim regualrization, only useful when if_ssim =True
        parser.add_argument('--if_ssim',type=util.str2bool,default=True,help='whether or not use additional ssim regularization')
        parser.add_argument('--lambda_ssim',type=float,default=0.5,help='weight for ssim loss')
        parser.add_argument('--ssim_idt',type=util.str2bool,default=True, help='if use ssim for identity mapping, only set it to be true if using ssim regularization')
        parser.add_argument('--gan_idt_weight', type=int, default=0,
                            help='if use ssim for identity mapping, only set it to be true if using ssim regularization')
        
        ### additional DSCNet parameters only useful when if_DSC_G = True
        parser.add_argument('--DSC_kernel',type=int,default=9,help='kernel size for DSCNet')
        parser.add_argument('--DSC_extend',type=float,default=1.0,help='extend scope for DSCNet')
        parser.add_argument('--DSC_offset',type=util.str2bool,default=True,help='whether or not use DSConv')
        parser.add_argument('--DSC_number',type=int,default=32,help='DSCNet basic conv channel')
        parser.add_argument('--DSC_n_blocks',type=int,default=9,help='number of middle resnet block')
        parser.add_argument('--DSE_padding_type',type=str, default='reflect',choices=['zero', 'replicate', 'reflect'],help='padding method in middle resnet blocks')

        ####    
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB']

        if self.opt.if_ssim ==True:
            self.loss_names += ['SSIM']
            self.loss_names += ['CLIP']
            self.loss_names += ['HFP']
        if self.opt.ssim_idt ==True and self.isTrain:
            self.loss_names +=['SSIM_Y']
            self.loss_names +=['HFP_Y']
        ###
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps): 
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name) 
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')] 

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y'] 
            self.visual_names += ['idt_B']
        ### default: 
        if self.isTrain:
            self.model_names = ['G', 'F', 'D','E', 'T']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netT = networks.define_G(input_nc=4, output_nc=1, ngf=64, netG='unet_trans_256', norm='instance',
                                      gpu_ids=self.gpu_ids)
        self.net_lpips = lpips.LPIPS(net='vgg')
        self.net_lpips.cuda()
        self.net_lpips.requires_grad_(False)
        self.crit_L1 = torch.nn.L1Loss()
        self.text_encoder = TextEncoder(model).cuda()
        self.L_clip = clip_score.L_clip_from_feature(only_hazy=True)
        self.L_clip_MSE = clip_score.L_clip_MSE()
        # self.time_embed = TimeEmbedding(128, 256)
        if self.isTrain:
            # 马尔科夫鉴别器
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_1 = vision_aided_loss.Discriminator(cv_type='clip', loss_type="multilevel_sigmoid", device="cuda").to(self.device)
            self.netD_1.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device) 
            self.criterionNCE = []
            self.gan_idt = torch.nn.L1Loss().to(self.device)

  
            ### add potential ssim_loss
            if self.opt.if_ssim ==True:
                self.criterionSSIM=ssim_criterion()
                print(f'ssim loss create successfully')

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device)) ## loss for discriminator ###

            self.criterionIdt = torch.nn.L1Loss().to(self.device)  ### whether can modify this one?  ## there is no idt loss
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.parameter_disc = list(self.netD.parameters()) + list(self.netD_1.parameters())
            self.optimizer_D = torch.optim.Adam(self.parameter_disc, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_T)
        if self.opt.use_prompt:
            learn_prompt = Prompts(self.opt.prompt_pretrain_dir).cuda()
            embedding_prompt = learn_prompt.embedding_prompt
            embedding_prompt.requires_grad = False
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * self.opt.length_prompt)]])
            self.text_features = self.text_encoder(embedding_prompt, tokenized_prompts)
            for name, param in learn_prompt.named_parameters():
                param.requires_grad_(False)
            
    def data_dependent_initialize(self, data,data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1) #data.A (b,c,256,256)
        self.set_input(data,data2)
        self.real_A = self.real_A[:bs_per_gpu] 
        self.real_B = self.real_B[:bs_per_gpu]  
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()  
            if self.opt.lambda_NCE > 0.0:  ## F network only work when using NCE_loss
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self): #optimize for a random time step t ###
        # forward
        self.forward()  
        self.netG.train()  
        self.netE.train()
        self.netD_1.train()
        self.netD.train()
        self.netF.train()
        self.netT.train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netD_1, True)
        ##
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        
        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        self.set_requires_grad(self.netD_1, False)
        
        self.optimizer_G.zero_grad()
        self.optimizer_T.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:  ### there is no loss for netF
            self.optimizer_F.step()       
        
    def set_input(self, input,input2=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # real_A form data
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # real_B from data
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device) # real_A form data2
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device) # # real_B from data2
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        '''
         for a given step T, random generated self.real_A_noisy(noise from real_A), self.real_A_noisy2(noise from real_A2)
         self.XtB (noise from real_B), self.fake_B2(GEN(self.real_A_noisy2)), self.fake_B ( GEN(self.real_A_noisy)),
         self.idt_B (GEN(self.real_B))
         Noising the image step by step by interpolation
        
        '''
        if self.opt.phase == 'train':
            tau = self.opt.tau  
            T = self.opt.num_timesteps # 5
            incs = np.array([0] + [1/(i+1) for i in range(T-1)]) 
            times = np.cumsum(incs) 
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda() 
            self.times = times 
            bs =  self.real_A.size(0) 
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long() 
            self.time_idx = time_idx 
            self.timestep     = times[time_idx] 
            
            ## the noising step ###
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.time_idx.int().item()+1): 
                    if t > 0:
                        delta = times[t] - times[t-1] 
                        denom = times[-1] - times[t-1] 
                        inter = (delta / denom).reshape(-1,1,1,1) 
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1) 
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)  
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long() 
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device) 
                    Xt_1, time_embed     = self.netG(Xt, time_idx, z)
                    # netG为生成器，从x_t_i直接生成x1

                    Xt2       = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_12, time_embed    = self.netG(Xt2, time_idx, z)

                    if self.opt.nce_idt:
                        XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                        time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                        time     = times[time_idx]
                        z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                        Xt_1B, time_embed = self.netG(XtB, time_idx, z)
                if self.opt.nce_idt:
                    self.XtB = XtB.detach()

                self.real_A_noisy = Xt.detach()  
                self.real_A_noisy2 = Xt2.detach()
                self.time_embed = time_embed
 
                        
            ## two noise tensor for forward pass ###
            if self.opt.nce_idt==True:
                z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
            else:
                z_in    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)

            z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
            """Run forward pass"""

            self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
            self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy

            if self.opt.flip_equivariance:
                self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
                if self.flipped_for_equivariance:
                    self.real = torch.flip(self.real, [3])
                    self.realt = torch.flip(self.realt, [3])
            
            ### generated fake images using noisy input self.realt


            self.fake, time_ = self.netG(self.realt,self.time_idx,z_in)
            self.fake_B2, time_ =  self.netG(self.real_A_noisy2,self.time_idx,z_in2)
            self.fake_B = self.fake[:self.real_A.size(0)] 
            if self.opt.nce_idt:
                self.idt_B = self.fake[self.real_A.size(0):]


        if self.opt.phase == 'test' or self.opt.validation_phase == True:  ### use this part when testing or validation during training 
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs =  self.real_A.size(0)
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []

            ##
            self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps): 
                    
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1, time_embed     = self.netG(Xt, time_idx, z)
                    
                    setattr(self, "fake_"+str(t+1), Xt_1)  
                    
    def compute_D_loss(self): ## discriminator to make real_B to be real, fake_B to be fake, Use GANloss
        """Calculate GAN loss for the discriminator"""
        bs =  self.real_A.size(0)
        
        fake = self.fake_B.detach() 
        std = torch.rand(size=[1]).item() * self.opt.std  
        
        pred_fake = self.netD(fake,self.time_idx) 
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        self.pred_real = self.netD(self.real_B,self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        

        self.time_embed = self.time_embed[:,None, :, None]
        self.loss_D1_fake = self.netD_1(fake + self.time_embed, for_real=False).mean()
        self.loss_D1_real = self.netD_1(self.real_B + self.time_embed, for_real=True).mean()
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D = self.loss_D1

        return self.loss_D

    def compute_E_loss(self): ## Custmize loss for netE
        
        bs =  self.real_A.size(0)
        
        """Calculate GAN loss for the discriminator"""
        
        XtXt_1 = torch.cat([self.real_A_noisy,self.fake_B.detach()], dim=1) 
        XtXt_2 = torch.cat([self.real_A_noisy2,self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        
        return self.loss_E
    
    def compute_G_loss(self):

        bs =  self.real_A.size(0)
        tau = self.opt.tau
        
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std
        ### clip loss
        if self.opt.use_prompt:

            clip_loss = self.L_clip(self.fake_B, self.real_A, self.text_features)
            self.loss_CLIP = clip_loss


        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake,self.time_idx) 
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        ### idt loss
        self.loss_idtGAN = self.gan_idt(self.real_B, self.idt_B) * self.opt.gan_idt_weight

        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            
            bs = self.opt.batch_size  

            ET_XY  = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY
            self.loss_SB += self.opt.tau*torch.mean((self.real_A_noisy-self.fake_B)**2)

        if self.opt.lambda_NCE > 0.0:  ## patchNCE_loss
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B) 
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            self.loss_NCE_Y = 0.0
            loss_NCE_both = self.loss_NCE

        ### add additional ssim resularization for generator to pererve high-level feature consistency
        if self.opt.if_ssim == True and self.opt.lambda_ssim >0:
            self.loss_SSIM = self.criterionSSIM(self.real_A,fake)
        else:
            self.loss_SSIM = 0.0

        if self.opt.if_ssim == True and self.opt.lambda_ssim >0 and self.opt.ssim_idt == True:
            self.loss_SSIM_Y = self.criterionSSIM(self.idt_B,self.real_B)
            self.loss_SSIM = (self.loss_SSIM + self.loss_SSIM_Y) /2


        fft_pred, sobel_pred = _get_hf_map(self.fake_B)
        fft_original, sobel_original = _get_hf_map(self.real_A)

        fft_loss = F.mse_loss(fft_pred, fft_original, reduction='mean')
        sobel_loss = F.mse_loss(sobel_pred, sobel_original, reduction='mean')
        hfp_loss = (fft_loss + sobel_loss) / 2

        self.loss_HFP = hfp_loss

        fft_pred_idt_B, sobel_pred_idt_B = _get_hf_map(self.idt_B)
        fft_original_real_B, sobel_original_real_B = _get_hf_map(self.real_B)

        fft_loss_idt_B = F.mse_loss(fft_pred_idt_B, fft_original_real_B, reduction='mean')
        sobel_loss_idt_B = F.mse_loss(sobel_pred_idt_B, sobel_original_real_B, reduction='mean')
        hfp_loss_idt_B = (fft_loss_idt_B + sobel_loss_idt_B) / 2
        self.loss_HFP_Y = hfp_loss_idt_B
        self.loss_HFP_all = (self.loss_HFP + self.loss_HFP_Y) / 2



        self.real_A_0_1 = (self.real_A + 1) / 2.
        J_dcp_real_A, t_dcp_real_A, A_dcp_real_A = dark_channel_generate(self.real_A_0_1)
        refine_T, out_T = self.netT(torch.cat((self.real_A, t_dcp_real_A), dim=1))

        shape = refine_T.shape
        dcp_A_scale = A_dcp_real_A
        map_A = (dcp_A_scale).reshape((shape[0], 3, 1, 1)).repeat(1, 1, shape[2], shape[3])

        refine_T_map = refine_T.repeat(1, 3, 1, 1)
        self.rec_dcp_real_A = synthesize_fog(self.fake_B, refine_T_map, map_A)

        self.loss_DCP = self.crit_L1(self.rec_dcp_real_A, self.real_A) * 1
        self.loss_DCP += self.net_lpips(self.rec_dcp_real_A, self.real_A).mean() * 1

        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + self.opt.lambda_NCE * loss_NCE_both+ \
                      self.loss_SSIM * 0.5 + self.loss_CLIP + self.loss_HFP_all * 0.5 \
                      + self.loss_DCP * 0.5
        return self.loss_G


    def calculate_NCE_loss(self, src, tgt): ### work like the feature loss?
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        feat_k = self.netG(src, self.time_idx*0,z,self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
