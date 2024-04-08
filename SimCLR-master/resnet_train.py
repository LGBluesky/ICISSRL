import logging
import os
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import save_config_file, accuracy, save_checkpoint
import util.misc as misc
from timm.utils import accuracy
import json

class Resnet_train(object):   # resnet train
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=self.args.output_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
    
    
    def train_and_val(self, train_loader, val_loader):
        max_accuracy = 0.0
        for epoch_counter in range(self.args.epochs):
            train_stats = self.train(train_loader, epoch_counter)
            val_stats = self.val(val_loader)
            print(f"Accuracy of the network on the val dataset, val images: {val_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, val_stats["acc1"])

            if self.writer is not None:
                self.writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch_counter)
                self.writer.add_scalar('perf/val_acc2', val_stats['acc2'], epoch_counter)
                self.writer.add_scalar('perf/val_loss', val_stats['loss'], epoch_counter)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch_counter': epoch_counter
                         }
            if self.args.output_dir and misc.is_main_process():
                if self.writer is not None:
                    self.writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    def train(self, train_loader, epoch_counter):
        self.model.train()
        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch_counter)

        scaler = GradScaler()
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        train_loader_len = len(train_loader)
        n_iter = epoch_counter * train_loader_len
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")


        for i, (images, targets) in enumerate(metric_logger.log_every(train_loader, self.args.print_freq, header)):
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # loss_value = loss.item()
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())

            lr = self.optimizer.param_groups[0]['lr']
            metric_logger.update(lr=lr)

            if n_iter % self.args.print_freq == 0:
                self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                self.writer.add_scalar('learning_rate', lr, global_step=n_iter)
            n_iter += 1

        if self.writer.log_dir and (
                epoch_counter % self.args.check_frequency == 0 or epoch_counter+1 == self.args.epochs) and epoch_counter> self.args.check_min:
            # save model checkpoints
            checkpoint_name = 'checkpoint_{:03d}.pth'.format(epoch_counter)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'args': self.args
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def val(self, val_loader):
        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'val:'
        self.model.eval()
        for i, (images, targets) in enumerate(metric_logger.log_every(val_loader, self.args.print_freq, header)):
            images = images.to(self.args.device, non_blocking=True)
            targets = targets.to(self.args.device, non_blocking=True)

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)  # default

        print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
