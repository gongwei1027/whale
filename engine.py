import torch
import time
import torch.nn as nn
import numpy as np
import torchnet as tnt
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import (
    get_inp_var
)


class Engine(object):
    def __init__(self, worker=25, device_ids=None, epoch=0, start_epoch=0, max_epochs=50, *args, **kwargs):
        self.worker = worker
        self.use_gpu = torch.cuda.is_available()
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.epoch_step = []
        self.device_ids = device_ids
        # meters
        self.meter_loss = tnt.meter.AverageValueMeter()
        # time measure
        self.batch_time = tnt.meter.AverageValueMeter()
        self.data_time = tnt.meter.AverageValueMeter()

        self.use_pb = True
        self.loss = None
        self.loss_batch  = None

        self._state = {}

    def state(self, name):
        if name in self._state:
            return self._state[name]
        else:
            return None

    def set_state(self, key, value):
        self._state[key] = value

    def on_start_epoch(self):
        self.meter_loss.reset()
        self.batch_time.reset()
        self.data_time.reset()

    def on_end_epoch(self, training, display=True):
        loss = self.meter_loss.value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.epoch, loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self):
        self.set_state('target_gt', self.state('target').clone())
        self.state('target')[self.state('target') == 0] = 1
        self.state('target')[self.state('target') == -1] = 0

        input = self.state('input')
        self.set_state('feature', input)

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.loss_batch = self.loss.data[0]
        self.meter_loss.add(self.loss_batch)

        # if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
        #     loss = self.state['meter_loss'].value()[0]
        #     batch_time = self.state['batch_time'].value()[0]
        #     data_time = self.state['data_time'].value()[0]
        #     if training:
        #         print('Epoch: [{0}][{1}/{2}]\t'
        #               'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
        #               'Data {data_time_current:.3f} ({data_time:.3f})\t'
        #               'Loss {loss_current:.4f} ({loss:.4f})'.format(
        #             self.state['epoch'], self.state['iteration'], len(data_loader),
        #             batch_time_current=self.state['batch_time_current'],
        #             batch_time=batch_time, data_time_current=self.state['data_time_batch'],
        #             data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
        #     else:
        #         print('Test: [{0}/{1}]\t'
        #               'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
        #               'Data {data_time_current:.3f} ({data_time:.3f})\t'
        #               'Loss {loss_current:.4f} ({loss:.4f})'.format(
        #             self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
        #             batch_time=batch_time, data_time_current=self.state['data_time_batch'],
        #             data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def init_learning(self, model, criterion):

        self.best_score = 0

    def learning(self, model, criterion, train_iter, dev_iter, optimizer=None):
        self.init_learning(model, criterion)


        if self.use_gpu:
            train_iter.pin_memory = True
            dev_iter.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model, device_ids=self.device_ids).cuda()

            criterion = criterion.cuda()


        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)

            # train for one epoch
            self.train(train_iter, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(dev_iter, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_score
            self.best_score = max(prec1, self.best_score)


            # self.save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': self._state('arch'),
            #     'state_dict': model.module.state_dict() if self.use_gpu else model.state_dict(),
            #     'best_score': self.best_score,
            # }, is_best)

            print(' *** best={best:.3f}'.format(best=self.best_score))
        return self.best_score

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch()

        if self.use_pb:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.set_state('iteration', i)
            self.set_state('data_time_batch', time.time() - end)
            self.data_time.add(self.state('data_time_batch'))

            self.set_state('input', input)
            self.set_state('target', target)

            self.on_start_batch()

            if self.use_gpu:
                self.set_state('target', self.state('target').cuda())

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.set_state('batch_time_current', (time.time() - end))
            self.state('batch_time').add(self.state('batch_time_current'))
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True)


    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch()

        if self.use_pb:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.set_state('iteration', i)
            self.set_state('data_time_batch', time.time() - end)
            self.data_time.add(self.state('data_time_batch'))

            self.set_state('input', input)
            self.set_state('target', target)

            self.on_start_batch()

            if self.use_gpu:
                self.set_state('target', self.state('target').cuda())

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.set_state('batch_time_current', (time.time() - end))
            self.state('batch_time').add(self.state('batch_time_current'))
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False)

        return score

    # def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
    #     if self._state('save_model_path') is not None:
    #         filename_ = filename
    #         filename = os.path.join(self.state['save_model_path'], filename_)
    #         if not os.path.exists(self.state['save_model_path']):
    #             os.makedirs(self.state['save_model_path'])
    #     print('save model {filename}'.format(filename=filename))
    #     torch.save(state, filename)
    #     if is_best:
    #         filename_best = 'model_best.pth.tar'
    #         if self._state('save_model_path') is not None:
    #             filename_best = os.path.join(self.state['save_model_path'], filename_best)
    #         shutil.copyfile(filename, filename_best)
    #         if self._state('save_model_path') is not None:
    #             if self._state('filename_previous_best') is not None:
    #                 os.remove(self._state('filename_previous_best'))
    #             filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
    #             shutil.copyfile(filename, filename_best)
    #             self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state('epoch') == np.array(self.epoch_step)) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiPlexNetworkEngine(Engine):
    def __init__(self, *args, **kwargs):
        super(MultiPlexNetworkEngine, self).__init__(*args, **kwargs)
        # if self.difficult_examples is None:
        #     self.difficult_examples = False
        # self.ap_meter = AveragePrecisionMeter(self.difficult_examples)

    def on_start_epoch(self):
        Engine.on_start_epoch(self)
        # self.ap_meter.reset()


class GCNMultiPlexNetworkEngine(MultiPlexNetworkEngine):

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # feature_var = torch.autograd.Variable(self.state('feature')).float()
        feature_var = self.state('feature')
        target_var = torch.autograd.Variable(self.state('target')).float().view(1, -1)
        print(target_var.shape)
        inp_var = get_inp_var("eclipse")
        inp_var = torch.from_numpy(inp_var).float().detach()
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

        # compute output
        self.set_state('output', model(feature_var, inp_var))
        # self.set_state('output', self.state('output').argmax(dim=-1).float())
        print(type(self.state('output')), target_var, target_var.shape, self.state('output').shape)
        self.set_state('loss', criterion(self.state('output'), target_var))

        if training:
            optimizer.zero_grad()
            self.state('loss').backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()

