import torch.optim
from . import FairseqOptimizer, register_optimizer

import torch_optimizer as optim


@register_optimizer('yogi')
class NewYogi(FairseqOptimizer):
    """This is the same Yogi but slightly different initialization function."""
    def __init__(
        self,
        args,
        params,
    ):
        super().__init__(args)

        self._optimizer = optim.Yogi(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas',
                            required=True,
                            metavar='B',
                            help='betas for Yogi optimizer')
        parser.add_argument('--adam-eps',
                            required=True,
                            type=float,
                            metavar='D',
                            help='epsilon for Yogi optimizer')
        parser.add_argument('--initial-accumulator',
                            required=True,
                            type=float,
                            metavar='D',
                            help='initial accumulator')
        parser.add_argument('--weight-decay',
                            '--wd',
                            default=0.0,
                            type=float,
                            metavar='WD',
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
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'initial_accumulator': self.args.initial_accumulator,
            'weight_decay': self.args.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True
