import json
import os
import pandas as pd
import pickle
import numpy as np
import re

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None, seq_transform=None, secondary=False):
        self.image_dir = args.image_dir if split == 'train' else args.image_dir_test
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.seq_transform = seq_transform
        self.secondary = secondary
        if self.secondary:
            self.ann = json.loads(open(args.ann_path_secondary, 'r').read())
            self.sec_image_dir = args.image_dir_secondary
            self.examples = self.ann[self.split]
            for i in range(len(self.examples)):
                self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
                self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
                self.examples[i]['img_labels'] = np.ones(1).astype(np.int8)
                self.examples[i]['report_labels'] = np.ones(1).astype(np.int8)
        else:
            self.ann = pd.read_csv(os.path.join(self.ann_path, self.split + '.csv'))
            self.examples = []
            self.n_pathologies = self.ann.iloc[0]['N Pathologies Report']
            for i, row in self.ann.iterrows():
                example = {}
                example['id'] = i  # row['dicom_id']
                example['image_path'] = row['Image Path']
                if example['image_path'][0] == '[':
                    assert False, "list of images currently not supported (may be fixed)"
                    example['image_path'] = row['Path'].strip('][').split(', ')
                    example['image_path'] = [s.strip('\'') for s in example['image_path']]
                else:
                    example['image_path'] = os.path.join(self.image_dir, example['image_path'])
                example['ids'] = tokenizer(row['Report'])[:self.max_seq_length]
                example['mask'] = [1] * len(example['ids'])
                img_pathologies = row[3:3+row['N Pathologies Image']]
                example['img_labels'] = (np.array(img_pathologies) == 1).astype(np.int8)
                report_pathologies = row[-row['N Pathologies Report']-1:-1]
                example['report_labels'] = (np.array(report_pathologies) == 1).astype(np.int8)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        img_dir = self.sec_image_dir if self.secondary else self.image_dir
        image_1 = Image.open(os.path.join(img_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(img_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        if self.seq_transform is not None:
            report_ids = self.seq_transform(report_ids)
        report_masks = example['mask']
        img_labels = example['img_labels']
        report_labels = example['report_labels']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, img_labels, report_labels)
        return sample


class IuxraySingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        img_dir = self.sec_image_dir if self.secondary else self.image_dir
        image_1 = Image.open(os.path.join(img_dir, image_path[0])).convert('RGB')
        #image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            #image_2 = self.transform(image_2)
        # image = torch.stack((image_1, image_2), 0)
        image = image_1
        report_ids = example['ids']
        if self.seq_transform is not None:
            report_ids = self.seq_transform(report_ids)
        report_masks = example['mask']
        img_labels = example['img_labels']
        report_labels = example['report_labels']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, img_labels, report_labels)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        # image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path)
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        if self.seq_transform is not None:
            report_ids = self.seq_transform(report_ids)
        report_masks = example['mask']
        img_labels = example['img_labels']
        report_labels = example['report_labels']
        seq_length = len(report_ids)
        if len(report_masks) > len(report_ids):
            print(example['report'])
        sample = (image_id, image, report_ids, report_masks, seq_length, img_labels, report_labels)
        return sample
