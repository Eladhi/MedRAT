import numpy as np
import torch
import itertools
import random
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, IuxraySingleImageDataset, MimiccxrSingleImageDataset


def transform_adjust_contrast():
    def _func(img):
        return transforms.functional.adjust_contrast(img, contrast_factor=1.5)
    return _func


class SentenceShuffle(object):
    def __init__(self, eos, first=0, last=0):
        self.eos = eos
        self.first = first
        self.last = last

    def __call__(self, sample):
        key = lambda sep: sep == self.eos
        leave_last = (self.last == sample[-1])
        _sample = sample[1:-1] if leave_last else sample[1:]
        sublists = [list(group)+[self.eos] for is_key, group in itertools.groupby(_sample, key) if not is_key]
        random.shuffle(sublists)
        s = [item for sublist in sublists for item in sublist]
        if leave_last:
            sample = [self.first] + s + [self.last]
        else:
            sample = [self.first] + s
        return sample


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, secondary=False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.secondary = secondary

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.seq_transform = None
        else:
            if not self.secondary:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transform_adjust_contrast(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            self.seq_transform = None

        if self.secondary:
            self.dataset = IuxraySingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform, seq_transform=self.seq_transform, secondary=self.secondary)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform, seq_transform=self.seq_transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': True if split == 'train' else False,  # for contrastive
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch, img_labels_batch, report_labels_batch = zip(*data)
        image_batch = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        img_labels_batch = np.array([x for x in img_labels_batch])
        report_labels_batch = np.array([x for x in report_labels_batch])

        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch), torch.FloatTensor(img_labels_batch), torch.FloatTensor(report_labels_batch)
