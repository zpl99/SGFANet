

from Nets.SGFANet.sgfanet_main_file import DeepR50_SGFANet
from Loss import loss
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net

def SGFANet_fn(net, batch, threshold=0.5):
    result = batch
    image, label, body, boundary, image_name, corner = result["image"], result["label"], result["body"], result[
        "boundary"], result["image_name"], result["corner"]
    loss, out = net(image.to(device),
                    [label.float().to(device), boundary.float().to(device), corner.float().to(device)], image_name)
    label = label.detach().cpu().numpy()
    out_new = out.detach().cpu().numpy()
    out_new = np.where(out_new > threshold, 1, 0)
    return loss, out_new, label, image_name




def setup_nets(args):
    if args.networks == "SGFANet_edge64_corner16":
        model = DeepR50_SGFANet(num_classes=args.num_classes, edge_points=64, corner_points=16,
                                                 gated=True, criterion=loss.JointEdgeCornerSegLightLossSGFANet(classes=args.num_classes))
        return model, SGFANet_fn

    else:
        assert False, "please check the args"
