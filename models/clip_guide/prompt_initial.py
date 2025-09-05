import os
import sys

sys.path.append('/code/models/clip_guide/')
import torch
import torch.nn as nn
from CLIP import clip
import clip_score
from collections import OrderedDict
# from options.base_options import BaseOptions
import argparse
import torch.nn.functional as F
import dataloader_prompt_add
import wandb, datetime

model, preprocess = clip.load('ViT-B/32', device=torch.device("cpu"))  # ViT-B/32
model.to('cuda')

for para in model.parameters():
    para.requires_grad = False
# opt = BaseOptions.praser
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Prompts(nn.Module):
    def __init__(self, initials=None):
        super(Prompts, self).__init__()
        print("The initial prompts are:", initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials, list):
            text = clip.tokenize(initials).cuda()
            # print(text)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials, str):
            prompt_path = initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # name = k[7:]
                name = k # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = False
        else:
            self.embedding_prompt = torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding(
                [" ".join(["X"] * opt.length_prompt),
                 " ".join(["X"] * opt.length_prompt)]).requires_grad_())).cuda()

    def forward(self, tensor, flag=1):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * opt.length_prompt)]])
        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)

        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            if flag == 0:
                similarity = (100.0 * image_features @ (text_features / nor).T)  # .softmax(dim=-1)
                if (i == 0):
                    probs = similarity
                else:
                    probs = torch.cat([probs, similarity], dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features / nor).T).softmax(dim=-1)  # /nor
                if (i == 0):
                    probs = similarity[:, 0]
                else:
                    probs = torch.cat([probs, similarity[:, 0]], dim=0)
        return probs


### 训练prompt
def train(opt):
    os.makedirs(opt.prompt_snapshots_folder, exist_ok=True)
    wandb.init(project=opt.project, name=datetime.datetime.now().strftime("%Y年-%m月-%d日-%H时-%M分"))
    train_loss = {}
    ## 创建列表全部为X的字符串
    learn_prompt = Prompts([" ".join(["X"] * (opt.length_prompt)), " ".join(["X"] * (opt.length_prompt))]).cuda()
    ### 第一个为negativeprompt 第二个为positiveprompt， 即clear
    text_encoder = TextEncoder(model)
    text_encoder.to('cuda')
    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()



    prompt_train_dataset_1 = dataloader_prompt_add.lowlight_loader(opt.lowlight_images_path,
                                                                   opt.normallight_images_path)
    prompt_train_loader_1 = torch.utils.data.DataLoader(prompt_train_dataset_1, batch_size=opt.prompt_batch_size,
                                                        shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(), lr=opt.prompt_lr,
                                        weight_decay=opt.weight_decay)

    # text_encoder = TextEncoder(model)
    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()
    # 初始化参数
    best_prompt = learn_prompt
    min_prompt_loss = 100


    total_iteration = 0

    for epoch in range(opt.num_epochs):   ### opt.nmu_epoch 2000

        for iteration, item in enumerate(prompt_train_loader_1):
            img_lowlight, label = item
            img_lowlight = img_lowlight.cuda()
            label = label.cuda()
            output = learn_prompt(img_lowlight, 0)
            loss = F.cross_entropy(output, label)
            prompt_optimizer.zero_grad()
            loss.backward()
            prompt_optimizer.step()
            train_loss['loss'] = loss.detach().item()


            if ((total_iteration + 1) % opt.prompt_display_iter) == 0:  ### 20
                if loss < min_prompt_loss:
                    min_prompt_loss = loss
                    best_prompt = learn_prompt
                    best_prompt_iter = total_iteration + 1
                    torch.save(learn_prompt.state_dict(),
                               opt.prompt_snapshots_folder + "best_prompt_round" + '.pth')

                    print("prompt current learning rate: ", prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", total_iteration + 1, ":", loss.item())
                    print("output", output.softmax(dim=-1), "label", label)
                    print("cross_entropy_loss", loss)


            if ((total_iteration + 1) % opt.prompt_snapshot_iter) == 0:   ### 100
                torch.save(learn_prompt.state_dict(),
                           opt.prompt_snapshots_folder + "iter_" + str(total_iteration + 1) + '.pth')

            total_iteration += 1
            wandb.log(train_loss)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--project', type=str,
                        default="train_prompt_2")
    parser.add_argument('--clip_pth', type=str,
                        default="/model/lanyunwei/diffusers/vit_b_32_.pt/ViT-B-32.pt")
    parser.add_argument('-b', '--lowlight_images_path', type=str, default="/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky/train_hazy/")
    parser.add_argument('--overlight_images_path', type=str, default=None)
    parser.add_argument('-r', '--normallight_images_path', type=str, default='/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky/train_clear/')
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--thre_train', type=float, default=90)
    parser.add_argument('--thre_prompt', type=float, default=60)
    parser.add_argument('--reconstruction_train_lr', type=float, default=0.00005)  # 0.0001
    parser.add_argument('--train_lr', type=float, default=0.00002)  # 0.00002#0.00005#0.0001
    parser.add_argument('--prompt_lr', type=float, default=0.000005)  # 0.00001#0.00008
    parser.add_argument('--T_max', type=float, default=100)
    parser.add_argument('--eta_min', type=float, default=5e-6)  # 1e-6
    parser.add_argument('--weight_decay', type=float, default=0.001)  # 0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=2000)  # 3000
    parser.add_argument('--num_reconstruction_iters', type=int, default=0)  # 1000
    parser.add_argument('--num_clip_pretrained_iters', type=int, default=0)  # 8000
    parser.add_argument('--noTV_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--prompt_batch_size', type=int, default=1)  # 32
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=20)
    parser.add_argument('--snapshot_iter', type=int, default=20)
    parser.add_argument('--prompt_display_iter', type=int, default=2)
    parser.add_argument('--prompt_snapshot_iter', type=int, default=2000)
    # parser.add_argument('--train_snapshots_folder', type=str,
    #                     default="./" + task_name + "/" + "snapshots_train_" + task_name + "/")
    parser.add_argument('--prompt_snapshots_folder', type=str,
                        default="/data/lanyunwei/my_experiment_results/CUNSB_RFIE/model/init_prompt_save_dir_2/")
    parser.add_argument('--load_pretrain', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--pretrain_dir', type=str,
                        default='./pretrained_models/init_pretrained_models/init_enhancement_model.pth')
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--prompt_pretrain_dir', type=str,
                        default='./pretrained_models/init_pretrained_models/init_prompt_pair.pth')

    opt = parser.parse_args()

    train(opt)

