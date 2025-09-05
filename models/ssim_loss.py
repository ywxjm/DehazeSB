from pytorch_msssim import SSIM
import torch
import torch.nn as nn

###
class ssim_criterion(nn.Module):
    def __init__(self, data_range=1,size_average=True, channel=3,nonnegative_ssim=True):
        super(ssim_criterion,self).__init__()
        self.ssim_module = SSIM(data_range=data_range, size_average=size_average,channel=3,nonnegative_ssim=nonnegative_ssim)

    def normalization(self, input: torch.Tensor) -> torch.Tensor: 
        """
        Normalize a tensor with range(-1,1) to (0,1)
        input: (B,C,H,W)  tensor
        output: (B,C,H,W) tensor
        """
        device = input.device
        data_type = input.dtype

        if isinstance(input, torch.Tensor):
            output = (input + 1) / 2 #(-1,1) -> (0,1)
            output = output.to(dtype=data_type, device=device)
            assert output.shape == input.shape
        else:
            raise ValueError('Input must be a torch.Tensor')
        return output
        
    def forward(self,input:torch.Tensor , target: torch.Tensor):
        if input.shape == target.shape:
            input = self.normalization(input) ## normalize to range 0-1 
            target = self.normalization(target)
            ssim_loss = 1 - self.ssim_module(input,target)
        else:
            raise ValueError(f'shape unaligned: input shape:{input.shape},target shape:{target.shape}')
        return ssim_loss
    
if __name__ =='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_A = torch.randn(3, 3, 256, 256, device=device, dtype=torch.float32)
    test_B = torch.randn(3, 3, 256, 256, device=device, dtype=torch.float32)
    ssim_loss = ssim_criterion()
    result = ssim_loss(test_A,test_B)
    print(f'{result},{type(result)},{result.device}')
    
