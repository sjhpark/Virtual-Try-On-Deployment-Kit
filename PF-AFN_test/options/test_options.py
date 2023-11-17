from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='checkpoints/PFAFN/warp_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='checkpoints/PFAFN/gen_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--sparsity', type=float, default=0.33, help='sparsity level')
        self.parser.add_argument('--module', type=str, default="AFWM_image_feature_encoder", help='module name to prune')
        self.parser.add_argument('--layer_idx', type=int, default=None, help='layer index to prune')

        self.isTrain = False
