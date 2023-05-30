#!/usr/bin/env python3
# coding:utf-8


import torch
import numpy as np
from tqdm import tqdm
import Dataset
import Nets
from Eveluate import metrics
import argparse
import torchvision.transforms.functional as TF
import os
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    # TODO:增加APEX支持
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument("--dataset", type=str, default="GF4_5", help="The choice of dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="The size of batch")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="The number of worker in dataloader. The larger the number, the faster the data set can be accessed")
    parser.add_argument("--edge_map", type=bool, default=True, help="Whether or not to use edge map to guide edge points sampling")
    parser.add_argument("--corner_map", type=bool, default=True, help="Whether or not to use corner map to guide corner points sampling")
    parser.add_argument("--networks", type=str, default="Resnet50_SGFANet_edge64_corner16", help="The choice of model")
    parser.add_argument("--dump", type=bool, default=False, help="Whether or not to store intermediate data locally (for debug)")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--model_path", type=str,
                        default=r"/media/dell/shihaoze1/lzp/Code/SGFANet_github/GF4_5Resnet50_SGFANet_edge64_corner16-model.ckpt")
    args = parser.parse_args()
    return args


def parse_output(outs, image_names, save_path=None):
    if save_path != None:
        save_path = save_path
    else:
        save_path = "predictions"
    os.makedirs(save_path, exist_ok=True)
    for i in range(outs.shape[0]):
        out = outs[i]
        out = torch.from_numpy(out)
        out = TF.to_pil_image(out.to(torch.float32))
        out.save(f"{save_path}/{image_names[i]}.png")


def evaluate(dataloader, net, net_fn, evaluator, writer=None, save_path=None):
    net.eval()
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Evaluate Mode", unit=" uttr")
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            _, out, label, image_name = net_fn(net, batch)
            evaluator.add_batch(label, out)
            parse_output(out, image_name, save_path)
        pbar.update(dataloader.batch_size)
    IoU = evaluator.IoU()
    OA = evaluator.OA()
    F1 = evaluator.F1()
    mIoU = evaluator.Mean_Intersection_over_Union()
    pbar.set_postfix(
        OA=f"{OA :.4f}",
        IoU=f"{IoU :.4f}",
        mIoU=f"{mIoU :.4f}",
        F1=f"{F1 :.4f}"
    )
    if writer is not None:
        writer.add_scalar(tag="OA", scalar_value=OA)
        writer.add_scalar(tag="IoU", scalar_value=IoU)
        writer.add_scalar(tag="mIoU", scalar_value=mIoU)
        writer.add_scalar(tag="F1", scalar_value=F1)
    pbar.close()
    return


def main(args):
    TestLoader = Dataset.setup_loaders(args)
    net, net_fn = Nets.setup_nets(args)
    writer = SummaryWriter(log_dir=f"./eval_runs/{args.dataset + args.networks}")  # save at run/日期时间-args.networks
    net = net.to(device)
    net = Nets.wrap_network_in_dataparallel(net, False)  # TODO:增加对APEX的支持
    net.load_state_dict(torch.load(args.model_path))
    print(f"{args.networks} build successful! ")
    evaluator = metrics.Evaluator(num_class=2)
    evaluate(TestLoader, net, net_fn, evaluator, writer=writer, save_path=f"{args.dataset + args.networks}")


if __name__ == '__main__':
    args = get_args()
    args.evaluate = True
    main(args)
