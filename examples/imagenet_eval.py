from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys

sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="path_to_imagenet",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnetamobile',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex weight cache')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--llga', action='store_true', default=False,
                    help='enable LLGA')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('-i', '--iterations', default=0, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--warmup-iterations', default=0, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, int8, bfloat16')
parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument("--performance", action='store_true',
                    help="measure performance only, no accuracy.")
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('--channels_last', type=int, default=1,
                    help='use channels last format')
parser.add_argument('--config_file', type=str, default="./conf.yaml",
                    help='config file for int8 tuning')
parser.add_argument("--quantized_engine", type=str, default=None,
                    help="torch backend quantized engine.")

parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0

args = parser.parse_args()
assert not (args.ipex and args.cuda), "ipex and cuda can't be set together!"

# set quantized engine
if args.quantized_engine is not None:
    torch.backends.quantized.engine = args.quantized_engine
else:
    args.quantized_engine = torch.backends.quantized.engine
print("backends quantized engine is {}".format(torch.backends.quantized.engine))

if args.ipex:
    import intel_extension_for_pytorch as ipex
    print("import IPEX **************")

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        print("=> using pre-trained parameters '{}'".format(args.pretrained))
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                     pretrained=args.pretrained)
    else:
        model = pretrainedmodels.__dict__[args.arch](pretrained=None)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(max(model.input_size)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)



    # if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
    #     scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
    # else:
    #     scale = 0.875
    scale = 0.875
    opt = pretrainedmodels.pretrained_settings[args.arch]["imagenet"]

    print('Images transformed from size {} to {}'.format(
        int(round(max(opt["input_size"]) / scale)),
        opt["input_size"]))
    # print('Images transformed from size {} to {}'.format(
    #     int(round(max(model.input_size) / scale)),
    #     model.input_size))

    val_tf = pretrainedmodels.utils.TransformImage(
        opt,
        scale=scale,
        preserve_aspect_ratio=args.preserve_aspect_ratio
    )
    if not args.dummy:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, val_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader=""

    # define loss function (criterion) and optimizer
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    # else:
        # model = torch.nn.DataParallel(model)

    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("[INFO] Use NHWC model")
        except:
            pass

    if args.ipex:
        model.eval()
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        print("Use ipex model")

    if args.evaluate:
        if args.precision == "bfloat16":
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                validate(val_loader, model, criterion, args)
        else:
            validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # compute output
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        # measure data loading time
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    # switch to evaluate mode
    model.eval()
    image_size = pretrainedmodels.pretrained_settings[args.arch]["imagenet"]["input_size"]
    images = torch.randn(args.batch_size, *image_size)
    if args.channels_last:
        try:
            images = images.contiguous(memory_format=torch.channels_last)
            print("[INFO] Use NHWC input")
        except:
            pass

    with torch.no_grad():
        iterations = args.iterations
        warmup = args.warmup_iterations
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        total_time = 0.0
        total_samples = 0
        batch_time_list = []

        if args.precision == 'inc_int8':
            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization(args.config_file)
            dataset = quantizer.dataset('dummy', (args.batch_size, *image_size), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = common.Model(model)
            q_model = quantizer()
            model = q_model.model

        if args.precision == "fx_int8":
            print('Converting int8 model...')
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
            qconfig_mapping = get_default_qconfig_mapping(args.quantized_engine)
            prepared_model = prepare_fx(model.eval(), qconfig_mapping, images)
            with torch.no_grad():
                for i in range(args.warmup_iterations):
                    prepared_model(images)
            model = convert_fx(prepared_model)
            print('Convert int8 model done...')

        if args.jit:
            try:
                model = torch.jit.trace(model, images, check_trace=False)
                print("---- Use trace model.")
            except:
                model = torch.jit.script(model)
                print("---- Use script model.")
            if args.ipex:
                model = torch.jit.freeze(model)
            #warmup
            for i in range(10):
                model(images)

        if args.llga:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_llga_enabled(True)
            model = torch.jit.trace(model, images)
            print("---- Enable LLGA.")

        if args.dummy:
            # image_size = pretrainedmodels.pretrained_settings[args.arch]["imagenet"]["input_size"]
            target = torch.arange(1, args.batch_size + 1).long()
            if args.cuda:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            # print("Start convert to onnx!")
            # torch.onnx.export(model.module, images, args.arch + ".onnx", verbose=False)
            # print("End convert to onnx!")
            if args.profile:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    schedule=torch.profiler.schedule(
                        wait=int((iterations + warmup)/2),
                        warmup=2,
                        active=1,
                    ),
                    on_trace_ready=trace_handler,
                ) as p:
                    for i in range(iterations + warmup):
                        start = time.time()
                        output = model(images)
                        p.step()
                        end = time.time()
                        elapsed = end - start
                        print("Iteration: {}, inference time: {} sec.".format(i, end - start), flush=True)
                        # measure elapsed time
                        if i >= warmup:
                            batch_time_list.append(elapsed * 1000)
                            total_time += elapsed
                            total_samples += args.batch_size
            else:
                for i in range(iterations + warmup):
                    start = time.time()
                    output = model(images)
                    end = time.time()
                    elapsed = end - start
                    print("Iteration: {}, inference time: {} sec.".format(i, end - start), flush=True)
                    # measure elapsed time
                    if i >= warmup:
                        batch_time_list.append(elapsed * 1000)
                        total_time += elapsed
                        total_samples += args.batch_size
        else:
            for i, (input, target) in enumerate(val_loader):
                if not args.evaluate or iterations == 0 or i < iterations + warmup:
                    if args.channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    if args.cuda:
                        target = target.cuda()
                        input = input.cuda()

                    # compute output
                    start = time.time()
                    output = model(input)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                    losses.update(loss.data.item(), input.size(0))
                    top1.update(prec1.item(), input.size(0))
                    top5.update(prec5.item(), input.size(0))

                    # measure elapsed time
                    end = time.time()
                    elapsed = end - start
                    if i >= warmup:
                        batch_time_list.append(elapsed * 1000)
                        total_time += elapsed
                        total_samples += args.batch_size

                    if i % args.print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                               i, len(val_loader), batch_time=batch_time, loss=losses,
                               top1=top1, top5=top5))
                else:
                    break

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        # TODO: this should also be done with the ProgressMeter
        if args.evaluate:
            latency = total_time / total_samples * 1000
            throughput = total_samples / total_time
            print("\n", "-"*20, "Summary", "-"*20)
            print("Batch Size:\t {}".format(args.batch_size))
            print("inference latency:\t {:.3f} ms".format(latency))
            print("inference Throughput:\t {:.2f} samples/s".format(throughput))
            # P50
            batch_time_list.sort()
            p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
            p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
            p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
            print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                    % (p50_latency, p90_latency, p99_latency))


        return top1.avg, top5.avg

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                args.arch + '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


if __name__ == '__main__':
    main()
