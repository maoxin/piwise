import numpy as np
import pandas as pd
import os, json
import torch

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

from piwise.dataset import ADE, ADE_Val
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Colorize, Colorize2
from piwise.visualize import Dashboard
from piwise.scores import runningScore

from tqdm import tqdm

NUM_CHANNELS = 3
NUM_CLASSES = 150

color_transform = Colorize(NUM_CLASSES + 1)
color_transform2 = Colorize2(NUM_CLASSES + 1)
image_transform = ToPILImage()
input_transform = ToTensor()

def train(args, model):
    # data loader
    loader = DataLoader(ADE(args.datadir),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(ADE_Val(args.datadir),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # loss function initiation
    object_info = pd.read_csv(os.path.join(args.datadir, 'object150_info.csv'))
    weight = torch.from_numpy( 1 / np.array(object_info.Ratio).astype('float32') )
    weight = weight / weight.max()
    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)

    # optimizer initiation
    optimizer = Adam(model.parameters(), 1e-4)
    if args.model.startswith('FCN'):
        optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
    if args.model.startswith('PSP'):
        optimizer = SGD(model.parameters(), 1e-2, .9, 1e-4)
    if args.model.startswith('Seg'):
        optimizer = SGD(model.parameters(), 1e-3, .9)
    if args.model.startswith('unet'):
        optimizer = SGD(model.parameters(), 1e-4, .99, 5e-4)

    # scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'max')

    # validation metrics
    running_metrics = runningScore(NUM_CLASSES)
    best_iou = -100.0 

    # visdom
    if args.steps_plot > 0:
        board = Dashboard(args.port)

    # load checkpoint
    epoch_start = 1
    if args.state:
        try:
            checkpoint = torch.load(args.state)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            epoch_start = checkpoint['epoch'] + 1
        except:
            raise Exception("state can be recognized")

    average_loss_epoch = []
    if os.path.isfile(os.path.join(os.getenv("HOME"), '.visdom/loss_all_epoch.json')):
        with open(os.path.join(os.getenv("HOME"), '.visdom/loss_all_epoch.json')) as f:
            d = json.loads(f.read())
        average_loss_epoch = d['jsons']['loss_over_epoch']['content']['data'][0]['y'][:epoch_start - 1]
    
    score_epoch = []
    if os.path.isfile(os.path.join(os.getenv("HOME"), '.visdom/scores_all_epoch.json')):
        with open(os.path.join(os.getenv("HOME"), '.visdom/scores_all_epoch.json')) as f:
            d = json.loads(f.read())
        score_epoch = d['jsons']['IoU_over_epoch']['content']['data'][0]['y'][:epoch_start - 1]

    # iteration
    for epoch in tqdm(range(epoch_start, args.num_epochs+epoch_start), 'epoch'):
        # train
        model.train()

        epoch_loss = []

        for step, (images, labels) in enumerate(tqdm(loader, 'train')):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            if args.steps_plot > 0 and step % args.steps_plot == 0:
                tqdm.write(f"epoch: {epoch}, step: {step} plot")
                image = inputs[0].cpu().data
                image[0] = image[0] * .229 + .485
                image[1] = image[1] * .224 + .456
                image[2] = image[2] * .225 + .406
                board.image(image,
                    f'input (epoch: {epoch}, step: {step})', f'{epoch // 30 * 30}-{(epoch // 30 + 1) * 30}')
                board.image(color_transform2(outputs[0].cpu().max(0)[1].unsqueeze(0).data),
                    f'output (epoch: {epoch}, step: {step})', f'{epoch // 30 * 30}-{(epoch // 30 + 1) * 30}')
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})', f'{epoch // 30 * 30}-{(epoch // 30 + 1) * 30}')

        average_loss_epoch.append(np.mean(epoch_loss))
        board.loss(average_loss_epoch)
        tqdm.write(f'Average Loss Epoch (epoch {epoch}): \t{average_loss_epoch[-1]}')

        # validation
        model.eval()
        for i_val, (images_val, labels_val) in enumerate(tqdm(val_loader, 'validation')):
            if args.cuda:
                images_val = images_val.cuda()
                labels_val = labels_val.cuda()

            images_val = Variable(images_val, volatile=True)
            labels_val = Variable(labels_val, volatile=True)

            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            tqdm.write(k, v)
        running_metrics.reset()

        score_epoch.append(score['Mean IoU : \t'])
        board.score(score_epoch)

        if score['Mean IoU : \t'] >= best_iou or epoch % 30 == 0:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            filename = os.path.join(args.savedir, f'{args.model}-{epoch:03}.pth')
            torch.save(state, filename)
            tqdm.write(f'save: {filename} (epoch: {epoch})')

        scheduler.step(best_iou)

def evaluate(args, model):
    model.eval()

    image = input_transform(Image.open(args.image))
    # image = ToTensor()(Image.open(args.image))
    label = model(Variable(image, volatile=True).unsqueeze(0))
    label = color_transform2(label[0].data.max(0)[1])

    image_transform(label).save(args.label)

def main(args):
    # load model
    Net = None
    if args.model == 'fcn8':
        Net = FCN8
    if args.model == 'fcn16':
        Net = FCN16
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'unet':
        Net = UNet
    if args.model == 'pspnet':
        Net = PSPNet
    if args.model == 'segnet':
        Net = SegNet
    assert Net is not None, f'model {args.model} not available'

    model = Net(NUM_CLASSES)

    # cuda set
    if args.cuda:
        model = model.cuda()

    # savedir set
    os.makedirs(args.savedir, exist_ok=True)

    # mode set
    if args.mode == 'eval':
        evaluate(args, model)
    if args.mode == 'train':
        train(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('image')
    parser_eval.add_argument('label')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--port', type=int, default=8097)
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--batch-size', type=int, default=1)
    parser_train.add_argument('--steps-loss', type=int, default=500)
    parser_train.add_argument('--steps-plot', type=int, default=0)
    parser_train.add_argument('--steps-save', type=int, default=500)
    parser_train.add_argument('--savedir', default='/media/jz76/hd_data/state_dic_class_segmentation')

    main(parser.parse_args())