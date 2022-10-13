# batch_size 256 Iter 32000 : 	 val_loss. 1.55306 	 val_acc. 59.43745 train_loss. 2.10715 	 train_acc. 49.09925
# vit no aug 50000
import torch.optim as optim
import torch.backends.cudnn as cudnn
import wandb
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import utils
import torch
import options
import json
from aug import RandAugment
from transformers import ViTFeatureExtractor, ViTForImageClassification
import requests
import timm

best_acc = 0  # best test accuracy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
###### args ######
args = options.get_args_parser()
torch.manual_seed(42)

###### wandb ######
wandb.init(project="vit-samll-pretrained")
wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
config = wandb.config
logger = utils.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

###### dataloader no aug######
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])
transform_train.transforms.insert(0, RandAugment(2, 14))
transform_test = transforms.Compose([
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
#
feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
net = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
model = timm.create_model(
                'vit_small_patch16_224',
                pretrained=True,
                num_classes=100,)
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
# print('==> Building model..', net)

# if args.resume_pth :
#     logger.info('loading checkpoint from {}'.format('/root/autodl-tmp/dino_deitsmall16_pretrain.pth'))
#     ckpt = torch.load('/root/autodl-tmp/dino_deitsmall16_pretrain.pth', map_location='cpu')
#     net.load_state_dict(ckpt, strict=True)
net.train()
net = net.to(device)
print('============loadchenggong')
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
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(args.beta[0], args.beta[1]), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nb_iter)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

###### Training ######
nb_iter, train_loss, train_acc, train_correct, train_total, val_loss, val_total, val_correct, val_acc = 0, 0, 0, 0, 0, 0, 0, 0, 0
while nb_iter <= args.nb_iter:

    inputs, targets = next(trainloader_iter)
    inputs = feature_extractor(images=inputs, size=32, return_tensors="pt")
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = net(**inputs)
    loss = criterion(outputs, targets)
    loss.backward()
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

