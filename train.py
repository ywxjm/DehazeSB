import glob
import time, torchvision
import torch, os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
from data.unpaired_dataset import ValidationSet
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

if __name__ == '__main__':


    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.    wandb.init(project=opt.project, name=datetime.datetime.now().strftime("%Y年-%m月-%d日-%H时-%M分"))
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    total_iters = 0                # the total number of training iterations

    ###
    if opt.if_validation ==True: # track validation result during training
        vali_data = ValidationSet(opt)
        validation_dataloader = DataLoader(vali_data,batch_size=opt.validation_batch,shuffle=False,drop_last=False) ##validation
    ###
    times = []
    train_loss = {}
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch


        dataset.set_epoch(epoch)
        for i, (data,data2) in tqdm(enumerate(zip(dataset,dataset2))):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size  ### how many images has been processed ###
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0: ### intialize the network start from opt.epoch_count ###
                model.data_dependent_initialize(data,data2)  
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data,data2)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d' % (epoch, opt.n_epochs + opt.n_epochs_decay))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        if opt.if_validation ==True and opt.validation_phase == True:
            metrics = {}
            result_dir = os.path.join(opt.validate_model_dir, f"epoch{epoch}_iter{total_iters}")
            os.makedirs(result_dir, exist_ok=True)
            model.eval() ## set to eval state
            opt.phase = 'test' # modify the phase for validation
            with torch.no_grad():
                for i,data in tqdm(enumerate(validation_dataloader)): #data is a dict contain 2 tensor(b,c,h,w) and 2 image list
                    model.set_input(data)
                    model.forward() ## call forward function to get related interval value

                    true_target = model.real_B
                    fake_target_1 = getattr(model, f'fake_1')
                    fake_1_dir_71hazy = os.path.join(result_dir, '71hazy', 'fake_1/')
                    os.makedirs(fake_1_dir_71hazy, exist_ok=True)
                    val_fake_1_PIL = torchvision.transforms.ToPILImage()(fake_target_1[0] * 0.5 + 0.5)
                    val_fake_1_PIL.save(os.path.join(fake_1_dir_71hazy, f"{model.image_paths[0].split('/')[-1]}"))

            model.train() # set back to training  
            opt.phase = 'train' # set phase back to training   


