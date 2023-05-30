import torch
import logging
import Dataset
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import Nets
from tqdm import tqdm
import torch.optim as optim
from Eveluate import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument("--dataset", type=str, default="GF4_5", help="The choice of dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="The number of worker in dataloader. The larger the number, the faster the data set can be accessed")
    parser.add_argument("--edge_map", type=bool, default=True, help="Whether or not to use edge map to guide edge points sampling")
    parser.add_argument("--corner_map", type=bool, default=True, help="Whether or not to use corner map to guide corner points sampling")
    parser.add_argument("--networks", type=str, default="Resnet50_SGFANet_edge64_corner16", help="The choice of model")
    parser.add_argument("--dump", type=bool, default=False, help="Whether or not to store intermediate data locally (for debug only)")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--valid_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="./model.ckpt")
    parser.add_argument("--random_seed", type=int, default=300)
    args = parser.parse_args()
    args.evaluate = False
    args.save_path = "./" + args.dataset + args.networks + "-model"
    return args


# In order to reconstruct the results
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def valid(dataloader, net, net_fn, evaluator):
    net.eval()
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid Mode", unit=" uttr")
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, out, label, _ = net_fn(net, batch)
            for v in loss.values():
                total_loss += v.mean()
            evaluator.add_batch(label, out)
        pbar.update(dataloader.batch_size)
    IoU = evaluator.IoU()
    OA = evaluator.OA()
    F1 = evaluator.F1()
    pbar.set_postfix(
        loss=f"{total_loss :.4f}",
        IoU=f"{IoU :.4f}",
        OA=f"{OA :.4f}",
        F1=f"{F1 :.4f}",
    )
    evaluator.reset()
    pbar.close()
    net.train()

    return total_loss / len(dataloader), IoU, OA, F1

def main_epoch(args):  # train by epoch
    TrainLoader, ValLoader = Dataset.setup_loaders(args)
    net, net_fn = Nets.setup_nets(args)
    net = net.to(device)
    net = Nets.wrap_network_in_dataparallel(net, False)  # TODO: ADD APEX
    print(f"{args.networks} build successful! ")
    total_epochs = args.epochs
    total_steps = len(TrainLoader)
    pbar = tqdm(total=total_epochs, desc="Train Mode", unit="epoch")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    print(f"optimizer is AdamW, learning rate is {args.lr}")
    evaluator = metrics.Evaluator(num_class=2)  # Building and background
    writer = SummaryWriter(comment=args.dataset + args.networks)  # save at run/time-args.networks
    best_accuracy = -1
    """
    Training phases
    """
    for epoch in range(total_epochs):
        batch_loss = 0.0
        pbar.update()
        for i, batch in enumerate(TrainLoader):
            loss, out, label, _ = net_fn(net, batch)
            total_loss = 0.0
            #
            for v in loss.values():
                total_loss += v
            evaluator.add_batch(label, out)

            total_loss = total_loss.mean()  # The loss returned under multiple GPUs is a multidimensional vector, and requires mean
            batch_loss += total_loss.detach().cpu().item()

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # pbar.update()
            pbar.set_postfix(
                batch=f"{i}/{total_steps}",
                loss=f"{total_loss.detach().cpu().item():.4f}",
                epoch=epoch + 1
            )

        IoU = evaluator.IoU()
        pbar.set_postfix(
            epoch_sum_loss=f"{batch_loss:.4f}",
            epoch_IoU=f"{IoU:.4f}",
            epoch=epoch + 1
        )
        writer.add_scalar(tag="train loss", scalar_value=batch_loss, global_step=epoch + 1)
        writer.add_scalar(tag="IoU", scalar_value=IoU, global_step=epoch + 1)
        evaluator.reset()
        """
        Validation phases
        """
        if (epoch + 1) % args.valid_epochs == 0:
            pbar.close()
            val_loss, val_IoU, val_OA, val_F1 = valid(ValLoader, net, net_fn, evaluator)
            writer.add_scalar("val loss", scalar_value=val_loss, global_step=epoch + 1)
            writer.add_scalar("val IoU", scalar_value=val_IoU, global_step=epoch + 1)
            writer.add_scalar("val OA", scalar_value=val_OA, global_step=epoch + 1)
            writer.add_scalar("val F1", scalar_value=val_F1, global_step=epoch + 1)
            # keep the best model, using mIoU as metrics
            if val_IoU > best_accuracy:
                best_accuracy = val_IoU
                best_state_dict = net.state_dict()
            pbar = tqdm(total=total_epochs, desc="Train Mode", unit="step", initial=epoch + 1)  # 重启 pbar

        if (epoch + 1) % args.valid_epochs == 0 and best_state_dict is not None:
            torch.save(best_state_dict, args.save_path + ".ckpt")
            pbar.write(f"epoch {epoch + 1}, best model saved. (accuracy={best_accuracy:.4f})")
            best_state_dict = None


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = get_args()
    print(args)
    seed = args.random_seed
    same_seeds(seed)  # To ensure reproducibility, but at the cost of slower training
    main_epoch(args)
