import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='vit_base CIFAR100 Training', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default='16', type=int)
    parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
    parser.add_argument('--opt', default="adamW")
    parser.add_argument('--nb_iter', type=int, default='150000')
    parser.add_argument('--out_dir', default="./outputs")
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth')
    parser.add_argument('--lr-scheduler', default=[200000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--beta', default=[0.9, 0.99], nargs="+", type=float, help="Adamw Betas")
    parser.add_argument('--weight-decay', default=0.3, type=float, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    return parser.parse_args()