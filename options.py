import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='vit CIFAR100 Training')
    parser.add_argument('--batch_size', default='64', type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--nb_iter', type=int, default='200000')
    parser.add_argument('--out_dir', default="./outputs")
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth')
    parser.add_argument('--lr-scheduler', default=[200000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--beta', default=[0.9, 0.99], nargs="+", type=float, help="Adamw Betas")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    return parser.parse_args()