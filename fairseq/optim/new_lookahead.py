import torch.optim
from . import FairseqOptimizer, register_optimizer

import torch_optimizer as optim
import torch


@register_optimizer('lookahead')
class NewLookahead(FairseqOptimizer):
    """This is the same Lookahead but slightly different initialization function."""
    def __init__(self, args, params):
        super().__init__(args)
        opt_config = self.optimizer_config
        subopt = torch.optim.Adam(
            params,
            lr=opt_config['lr'],
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay'],
        )
        self._optimizer = optim.Lookahead(subopt, k=opt_config['k'], alpha=opt_config['alpha'])

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas',
                            required=True,
                            metavar='B',
                            help='betas for LookAhead optimizer')
        parser.add_argument('--adam-eps',
                            required=True,
                            type=float,
                            metavar='D',
                            help='epsilon for LookAhead optimizer')
        parser.add_argument('--weight-decay',
                            '--wd',
                            default=0.0,
                            type=float,
                            metavar='WD',
                            help='weight decay')
        parser.add_argument('--k', default=5, type=int, help='k in lookaherd')
        parser.add_argument('--alpha',
                            default=0.5,
                            type=int,
                            help='alpha in lookahead')
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
            'weight_decay': self.args.weight_decay,
            'k': self.args.k,
            'alpha': self.args.alpha,
        }

    @property
    def supports_flat_params(self):
        return True
