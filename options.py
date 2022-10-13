import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='vit CIFAR100 Training', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default='32', type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--opt', default="adamW")
    parser.add_argument('--nb_iter', type=int, default=300000)
    parser.add_argument('--out_dir', default="./outputs")
    parser.add_argument("--resume-pth", type=str, default='/root/vit-tiny-patch16-224/pytorch_model.bin', help='resume pth')
    parser.add_argument('--lr-scheduler', default=[20000, 40000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--beta', default=[0.9, 0.99], nargs="+", type=float, help="Adamw Betas")
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--scheduler', default="cosine", help='cosine and MultiStep')
    return parser.parse_args()