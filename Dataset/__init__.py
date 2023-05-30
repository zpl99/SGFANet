"""
Dataset setup and loaders
This file including the different datasets processing pipelines
"""

import Dataset.GF as GF4_5
from torch.utils.data import DataLoader
import config
from Transform.L_transforms import data_transform_pipline
from Transform.L_transforms import data_transform_pipline_for_s2image

shuffle_tag = True  # Control whether need shuffle the dataset


def setup_loaders(args):
    if args.dataset == "GF4_5":
        if args.evaluate:
            test_set = GF4_5.GF4_5Dataset("semantic", "test", data_transform_pipline,
                                          edge_map=args.edge_map, corner_map=args.corner_map, dump=args.dump, )
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True, shuffle=shuffle_tag)
            return test_loader
        train_set = GF4_5.GF4_5Dataset("semantic", "train", data_transform_pipline,
                                       edge_map=args.edge_map, corner_map=args.corner_map, dump=args.dump, )
        val_set = GF4_5.GF4_5Dataset("semantic", "val", data_transform_pipline, edge_map=args.edge_map,
                                     corner_map=args.corner_map,
                                     dump=args.dump)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                  shuffle=shuffle_tag)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                shuffle=shuffle_tag)
        return train_loader, val_loader
