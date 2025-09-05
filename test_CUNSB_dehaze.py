
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from torchvision import transforms
import torchvision

from data.unpaired_dataset import ValidationSet, ValidationHaze2020
from data.unpaired_dataset import ValidationURHI
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    if opt.dataset_name == 'haze2020':
        val_ohaze = ValidationHaze2020(opt)
    elif opt.dataset_name == 'URHI':
        val_ohaze = ValidationURHI(opt)

    OHAZE_dataloader = DataLoader(val_ohaze, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    model = create_model(opt)      # create a model given opt.model and other options
    test_dir = os.path.join('/output/CUNSB_dehaze/', '{}/{}/'.format(opt.checkpoint_dir.split('/')[-1   ],opt.checkpoint_name), '{}'.format(opt.dataset_name))
    os.makedirs( test_dir, exist_ok=True)
    print('creating test directory to store generation', test_dir)

    for i, (data,data2) in tqdm(enumerate(zip(OHAZE_dataloader, OHAZE_dataloader))):
        if i == 0:
            model.data_dependent_initialize(data,data2)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()

        model.set_input(data,data2)  # unpack data from data loader
        model.test()           # run inference

        fake_target_1 = getattr(model, f'fake_1')

        val_fake_1_PIL = torchvision.transforms.ToPILImage()(fake_target_1[0] * 0.5 + 0.5)
        fake_1_dir = os.path.join(test_dir, 'fake_1')
        os.makedirs(fake_1_dir, exist_ok=True)
        val_fake_1_PIL.save(os.path.join(fake_1_dir, f"{model.image_paths[0].split('/')[-1]}"))



