from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        ### evaluation metrics
        parser.add_argument('--metrics_save_dir', type=str,default='',help='the place to directory where store the psnr and ssim metrics')
        parser.add_argument('--metrics_dic_name',type=str,default='default_metrics.txt',help='the name to store the output metrics')
        parser.add_argument('--target_truth_path',type=str,default='',help='the path to directory where store the ground truth target')
        parser.add_argument('--checkpoint_dir', type=str, default='',
                            help='the path to directory where store the ground truth target')
        parser.add_argument('--checkpoint_name', type=str, default='',
                            help='the path to directory where store the ground truth target')
        parser.add_argument('--dataset_name', type=str, default='',
                            help='the path to directory where store the ground truth target')
        parser.add_argument('--metric_name', type=str, default='',
                            help='the path to directory where store the ground truth target')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
    

