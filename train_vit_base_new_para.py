# batch_size 32
import torch.optim as optim
import torch.backends.cudnn as cudnn
import wandb
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
from vit import vit_base
import utils
import torch
import vit_base_option
import json
from timm.data import create_transform
from aug import RandAugment
from model import ViT
best_acc = 0  # best test accuracy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
###### args ######
args = vit_base_option.get_args_parser()
torch.manual_seed(42)

###### wandb ######
wandb.init(project="new-train-cifar100")
wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
config = wandb.config
logger = utils.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

###### dataloader no aug######
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
# ])


transform_train = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )


transform_train.transforms[0] = transforms.Compose([
    transforms.Resize(int(args.input_size / args.eval_crop_ratio), interpolation=3),  # to maintain same ratio w.r.t. 224 images
    transforms.CenterCrop(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
# transform_train.transforms.insert(0, RandAugment(2, 14))
transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

logger.info('loading train loader')
trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
trainloader_iter = utils.cycle(trainloader)

logger.info('loading val loader')
valset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=100, shuffle=False, num_workers=8)
valloader_iter = utils.cycle(valloader)


###### Network ######
net = vit_base()
# net = ViT(
#     image_size = 32,
#     patch_size = 4,
#     num_classes = 100,
#     dim = 512,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512,
#     dropout = 0.1,
#     emb_dropout = 0.1)
print('==> Building model..', net)

if args.resume_pth :
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net = net.to(device)

###### device ######
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

###### optim ######
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, betas=(args.beta[0], args.beta[1]), weight_decay=args.weight_decay)
elif args.opt == "adamW":
    # optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(args.beta[0], args.beta[1]), weight_decay=args.weight_decay)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nb_iter)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

###### Training ######
nb_iter, train_loss, train_acc, train_correct, train_total, val_loss, val_total, val_correct, val_acc = 0, 0, 0, 0, 0, 0, 0, 0, 0
while nb_iter <= args.nb_iter:

    inputs, targets = next(trainloader_iter)
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    param_norms = None
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    param_norms = utils.clip_gradients(net, args.clip_grad)  # add clip_grad
    optimizer.step()
    scheduler.step()
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    train_total += targets.size(0)
    train_correct += predicted.eq(targets).sum().item()
    train_acc += 100. * train_correct / train_total
    if nb_iter % 100 == 0:
        train_loss /= 100
        train_acc /= 100
        wandb.log({'train_acc': train_acc, 'train_loss': train_loss,
               "lr": optimizer.param_groups[0]["lr"], 'iter':nb_iter})
        print('train_acc', train_acc)
        logger.info(f"Train. Iter {nb_iter} : \t train_loss. {train_loss:.5f} \t train_acc. {train_acc:.5f}")

        train_loss, train_acc, train_correct, train_total = 0, 0, 0, 0

###### eval ######

    net.eval()
    with torch.no_grad():
        inputs, targets = next(valloader_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += targets.size(0)
        val_correct += predicted.eq(targets).sum().item()
        # val_acc = torch.eq(outputs.argmax(-1), targets).float().mean()
        val_acc += 100.* val_correct / val_total
    if nb_iter % 500 == 0:
        val_loss /= 500
        val_acc /= 500
        wandb.log({'val_acc': val_acc, 'val_loss': val_loss,
                   "lr": optimizer.param_groups[0]["lr"], 'iter': nb_iter})
        logger.info(f"Eval. Iter {nb_iter} : \t val_loss. {val_loss:.5f} \t val_acc. {val_acc:.5f}")
        val_loss, val_acc, val_total, val_correct = 0, 0, 0, 0

###### saving ckpt ######
    if val_acc > best_acc:
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': val_acc,
            'nb_iter': nb_iter,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = val_acc
    nb_iter+=1

