from fairseq.optim import FairseqOptimizer, register_optimizer
from torch.optim import SGD

@register_optimizer('lars')
class FairseqLARS(FairseqOptimizer):
    """LARS optimizer."""

    def __init__(self, args, params):
        super().__init__(args)
        try:
            from .LARC import LARC
            cfg = self.optimizer_config
            opt = SGD(params, lr=cfg['lr'], momentum=cfg['momentum'])
            self._optimizer = LARC(opt, eps=cfg['eps'])
        except ImportError:
            raise ImportError('Please install apex to use LARS optimizer')

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--lars-momentum', default=0, metavar='B',
                            help='momentum for LARS optimizer')
        parser.add_argument('--lars-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for LARS optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'momentum': self.args.lars_momentum,
            'eps': self.args.lars_eps,
            'weight_decay': self.args.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return False
