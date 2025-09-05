import torch
import torch.nn as nn
import glob, os
from PIL import Image

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):  # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """

        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter(torch.ones(p.size()))

        if I.is_cuda:
            N = N.cuda()

        # print(N.shape)
        # print(I.shape)
        # print('-----------')

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b

def get_dark_channel(input, win_size=5, ):
    _, _, H, W = input.shape
    _, _, H, W = input.shape
    maxpool = nn.MaxPool3d((3, win_size, win_size), stride=1, padding=(0, win_size // 2, win_size // 2))
    dc = maxpool(0 - input[:, :, :, :])

    return -dc

def get_atmosphere_dark(I, dark_ch, p=0.01):
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

    return A

def dark_channel_generate(x, omega=0.95):
    guided_filter = GuidedFilter(r=15, eps=1e-3)
    if x.shape[1] > 1:
            # rgb2gray
        guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
    else:
        guidance = x

    # guidance = (guidance + 1) / 2
    guidance = torch.unsqueeze(guidance, dim=1)
    # imgPatch = (x + 1) / 2
    imgPatch = x

    num, chl, height, width = imgPatch.shape

    # dark_img and A with the range of [0,1]
    dark_img = get_dark_channel(imgPatch)
    A = get_atmosphere_dark(imgPatch, dark_img)
    A = A.unsqueeze(-1).unsqueeze(-1)
    map_A = A.repeat(1, 1, height, width)
    # make sure channel of trans_raw == 1
    trans_raw = 1 - omega * get_dark_channel(imgPatch / map_A)

    # get initial results
    T_DCP = guided_filter(guidance, trans_raw)
    J_DCP = (imgPatch - map_A) / T_DCP.repeat(1, 3, 1, 1) + map_A

    return J_DCP, T_DCP, torch.squeeze(A)


if __name__ == '__main__':

    from torchvision import transforms
    img_pth = '/data/lanyunwei/my_experiment_results/dehaze_demo_results/hazy/I-HAZE/hazy_test/'
    inp_files = []
    result_dir = '//data/lanyunwei/my_experiment_results/dehaze_demo_results/dcp/dcp_ihaze1024/'
    os.makedirs(result_dir, exist_ok=True)
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    for extension in image_extensions:
        # 使用glob模块获取所有匹配的图片文件
        inp_files += glob.glob(os.path.join(img_pth, extension))

    for img in inp_files:
        img_name = img.split('/')[-1]
        img = Image.open(img).convert('RGB')
        resized_img = img.resize((1024, 1024), Image.LANCZOS)
        img = transforms.ToTensor()(resized_img).unsqueeze(0).to('cuda')
        J_dcp_real_A, t_dcp_real_A, A_dcp_real_A = dark_channel_generate(img)
        J_dcp_real_A = torch.clamp(J_dcp_real_A, 0, 1)
        J = transforms.ToPILImage()(J_dcp_real_A[0])
        J.save(os.path.join(result_dir, img_name))


