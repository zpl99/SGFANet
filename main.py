
import torch
import argparse
import Nets
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument("--dataset", type=str, default="INRIA-0.3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--edge_map", type=bool, default=True)
    parser.add_argument("--corner_map", type=bool, default=True)
    parser.add_argument("--networks", type=str,
                        default="SGFANet_edge64_corner16")
    parser.add_argument("--dump", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=90000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--valid_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--valid_steps", type=float, default=500)
    parser.add_argument("--save_steps", type=float, default=500)
    parser.add_argument("--save_path", type=str, default="./model.ckpt")
    parser.add_argument("--useTiny", type=bool, default=False)
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


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    args = get_args()
    print(args)
    seed = args.random_seed
    same_seeds(seed)
    print("random seed is ", str(seed))
    model, fn = Nets.setup_nets(args)
    model.to(device)

    # An example for inference
    input = torch.rand([2, 3, 256, 256]).to(device)
    _, output = model(input)
    print(output.shape)

    # An example for training

    input = torch.rand([2, 3, 256, 256]).to(device)
    label = torch.ones([2, 1, 256, 256]).to(device)
    boundary= torch.ones([2, 1, 256, 256]).to(device)
    corner = torch.ones([2, 1, 256, 256]).to(device)

    loss, out = model(input, [label.float(), boundary.float(), corner.float()])
    total_loss = 0.0
    for v in loss.values():
        total_loss += v
    total_loss = total_loss.mean()
    total_loss.backward()
    print(total_loss)
    print(output.shape)







