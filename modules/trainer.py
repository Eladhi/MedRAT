import logging
import os
from abc import abstractmethod

import torch
import json
from numpy import inf
from tqdm import tqdm

from .contrastive import AugmentImage, AugmentText, SupConLoss


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, wandb):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        fh = logging.FileHandler(os.path.join(args.save_dir, 'log.log'), 'w+')
        self.logger.addHandler(fh)
        self.wandb = wandb

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args)

    @abstractmethod
    def _eval_model(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def eval(self):
        result = self._eval_model()
        log = {'epoch': 'eval'}
        log.update(result)
        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, test_reports = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log, test_reports, self.args.save_dir)
            self._print_best()

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log, test_reports, save_dir):
        with open(os.path.join(save_dir, 'reports_last.json'), 'w') as fout:
            json.dump(test_reports, fout)
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)
            with open(os.path.join(save_dir, 'reports_best.json'), 'w') as fout:
                json.dump(test_reports, fout)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, args):
        resume_path = str(args.resume)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if args.reset_lr:
            self.optimizer.param_groups[0]['lr'] = args.lr_ve
            self.optimizer.param_groups[1]['lr'] = args.lr_ed

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader, wandb=None):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, wandb)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.log_iter = 0
        self.lng_weight = args.lng_weight
        self.cls_weight = args.cls_weight
        self.cnt_weight = args.cnt_weight
        self.augment_image = AugmentImage()
        self.augment_text = AugmentText()
        self.contrastive_loss = SupConLoss()

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train on the training set.'.format(epoch, self.epochs))
        train_loss = 0
        lng_loss_acc = 0
        mlc_loss_acc = 0
        cont_loss_acc = 0
        self.model.train()

        for batch_idx, (images_id, images, reports_ids, reports_masks, img_labels, report_labels) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks, img_labels, report_labels = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device), img_labels.to(self.device), report_labels.to(self.device)
            output, src_mlc, tgt_mlc, zi1, zt1 = self.model(images, reports_ids, mode='train')
            # language & classification loss
            loss_lng = self.criterion[0](output, reports_ids, reports_masks)
            loss_mlc = self.criterion[1](src_mlc, img_labels) + self.criterion[1](tgt_mlc, report_labels)
            # contrastive
            if self.cnt_weight > 0:
                i1, i2 = self.augment_image(images)
                t1, t2 = self.augment_text(reports_ids)
                _, _, _, zi2, zt2 = self.model(i2, t2, mode='train')
                loss_cnt = self.contrastive_loss(torch.concat((torch.stack((zi1, zi2), dim=1), torch.stack((zt1, zt2), dim=1))), torch.concat((img_labels, report_labels)))  #for SupConLoss - with labels & combined
            else:
                loss_cnt = torch.zeros_like(loss_lng)
            loss = self.lng_weight * loss_lng + self.cls_weight * loss_mlc + self.cnt_weight * loss_cnt
            train_loss += loss.item()
            lng_loss_acc += loss_lng.item()
            mlc_loss_acc += loss_mlc.item()
            cont_loss_acc += loss_cnt.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))
                if self.wandb is not None:
                    self.wandb.log({"loss": train_loss / (batch_idx + 1),
                                    "language_loss": lng_loss_acc / (batch_idx + 1),
                                    "mlc_loss": mlc_loss_acc / (batch_idx + 1),
                                    "cont_loss": cont_loss_acc / (batch_idx + 1),
                                    "lr_ve": self.lr_scheduler.get_last_lr()[0],
                                    "lr_ed": self.lr_scheduler.get_last_lr()[1]}, step=self.log_iter)
                    self.log_iter += 1

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate on the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, img_labels, _) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks, img_labels = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), img_labels.to(self.device)
                output, _, others = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate on the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            mlc_loss_acc = 0
            lng_loss_acc = 0
            for batch_idx, (images_id, images, reports_ids, reports_masks, img_labels, _) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks, img_labels = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), img_labels.to(self.device)
                output, _, others = self.model(images, mode='sample')
                mlc_pred = torch.stack([o[0] for o in others])  # at 'sample' mode others contain auxiliary information
                loss_mlc = self.criterion[1](mlc_pred, img_labels)
                mlc_loss_acc += loss_mlc.item()
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            test_reports_list = []
            for ri, r in enumerate(test_gts):
                e = {}
                e['id'] = self.test_dataloader.dataset.examples[ri]['id']
                e['image_path'] = self.test_dataloader.dataset.examples[ri]['image_path']
                e['img_labels'] = self.test_dataloader.dataset.examples[ri]['img_labels'].tolist()
                e['report_labels'] = self.test_dataloader.dataset.examples[ri]['report_labels'].tolist()
                e['output'] = test_res[ri]
                e['gt'] = test_gts[ri]
                test_reports_list.append(e)

            log.update(**{'test_' + k: v for k, v in test_met.items()})
            if self.wandb is not None:
                self.wandb.log({'test_mlc_loss': mlc_loss_acc / (batch_idx + 1),
                                'test_lng_loss': lng_loss_acc / (batch_idx + 1)}, step=self.log_iter - 1, commit=True)

        return log, test_reports_list

    def _eval_model(self):
        log = {}
        print('Start to evaluate on the test set.')
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, img_labels, _) in enumerate(tqdm(self.test_dataloader)):
                images, reports_ids, reports_masks, img_labels = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), img_labels.to(self.device)
                output, _, others = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        return log
