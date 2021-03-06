import torch
import time
import os
import shutil
import torch.nn as nn
import numpy as np
import torchnet as tnt
import torch.backends.cudnn as cudnn
from sklearn import metrics
from tqdm import tqdm
from utils import (
    get_inp_var
)


class Engine(object):
    def __init__(self, worker=25, device_ids=None, epoch=0, start_epoch=0,
                 max_epochs=10, model_path=None, difficult_examples=None, project=None, *args, **kwargs):
        self.test = kwargs.get('test', False)
        self.worker = worker
        self.use_gpu = torch.cuda.is_available()
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.epoch_step = []
        self.device_ids = device_ids
        # meters
        self.meter_loss = tnt.meter.AverageValueMeter()
        # ap_meter
        self.ap_meter = tnt.meter.AverageValueMeter()
        # time measure
        self.batch_time = tnt.meter.AverageValueMeter()
        self.data_time = tnt.meter.AverageValueMeter()

        self.use_pb = True
        self.loss = None
        self.loss_batch  = None

        self._state = {}

        self.model_path = model_path or os.getcwd()

        self.difficult_examples = difficult_examples

        self.project = project

    def state(self, name):
        if name in self._state:
            return self._state[name]
        else:
            return None

    def set_state(self, key, value):
        self._state[key] = value

    def on_start_epoch(self):
        self.meter_loss.reset()
        self.ap_meter.reset()
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

        input = self.state('input')[0]
        self.set_state('feature', input)

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        # print(self.loss, type(self.loss))
        # print(self.loss.cpu().data.item())
        # exit(1)
        self.loss_batch = self.loss.cpu().data.item()
        self.meter_loss.add(self.loss_batch)
        if not training:
            self.ap_meter.add(self.state('batch_score'))

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

    def learning(self, model, criterion, train_iter, dev_iter, test_iter, optimizer=None):
        self.init_learning(model, criterion)

        if self.use_gpu:
            train_iter.pin_memory = True
            dev_iter.pin_memory = True
            test_iter.pin_memory = True
            cudnn.benchmark = True

            if self.test:
                model_best_file = os.path.abspath(os.path.dirname(__file__)) + '/model_best_{}.pth.tar'.format(self.project)
                checkpoint = torch.load(model_best_file)
                model.load_state_dict(checkpoint['state_dict'])
            model = torch.nn.DataParallel(model, device_ids=self.device_ids).cuda()

            criterion = criterion.cuda()

        if self.test:
            self.validate(test_iter, model, criterion)
            return

        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            # train for one epoch
            self.train(train_iter, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(dev_iter, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_score
            self.best_score = max(prec1, self.best_score)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if self.use_gpu else model.state_dict(),
                'best_score': self.best_score,
            }, is_best)

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
            self.batch_time.add(self.state('batch_time_current'))
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
            self.batch_time.add(self.state('batch_time_current'))
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False)

        return score

    # def test(self, data_loader, model, criterion):
    #
    #     for i, (input, target) in enumerate(data_loader):

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self.model_path:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            filename = "{}/{}".format(self.model_path, self.project)
        torch.save(state, filename)
        if is_best:
            filename_best = "{}/{}".format(self.model_path, 'model_best_{}.pth.tar'.format(self.project))
            if os.path.exists(filename_best):
                os.remove(filename_best)
            shutil.copyfile(filename, filename_best)

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
        # decay = 0.1 if sum(self.epoch == np.array(self.epoch_step)) > 0 else 1.0
        decay = 0.1 if self.epoch % 50 == 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiPlexNetworkEngine(Engine):
    def __init__(self, *args, **kwargs):
        super(MultiPlexNetworkEngine, self).__init__(*args, **kwargs)

    def on_start_epoch(self):
        Engine.on_start_epoch(self)
        self.ap_meter.reset()

    def on_end_epoch(self, training, display=True):
        loss = self.meter_loss.value()[0]
        acc = self.ap_meter.value()[0]
        print('Epoch: [{0}]\t'
              'Loss {loss:.4f}\t'
              'acc {acc:.3f}'.format(self.epoch, loss=loss, acc=acc))
        return acc
        # map = 100 * self.state('ap_meter').value().mean()
        # loss = self.state('meter_loss').value()[0]
        # OP, OR, OF1, CP, CR, CF1 = self.state('ap_meter').overall()
        # OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state('ap_meter').overall_topk(3)
        # if display:
        #     if training:
        #         print('Epoch: [{0}]\t'
        #               'Loss {loss:.4f}\t'
        #               'mAP {map:.3f}'.format(self.state('epoch'), loss=loss, map=map))
        #         print('OP: {OP:.4f}\t'
        #               'OR: {OR:.4f}\t'
        #               'OF1: {OF1:.4f}\t'
        #               'CP: {CP:.4f}\t'
        #               'CR: {CR:.4f}\t'
        #               'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        #     else:
        #         print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
        #         print('OP: {OP:.4f}\t'
        #               'OR: {OR:.4f}\t'
        #               'OF1: {OF1:.4f}\t'
        #               'CP: {CP:.4f}\t'
        #               'CR: {CR:.4f}\t'
        #               'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        #         print('OP_3: {OP:.4f}\t'
        #               'OR_3: {OR:.4f}\t'
        #               'OF1_3: {OF1:.4f}\t'
        #               'CP_3: {CP:.4f}\t'
        #               'CR_3: {CR:.4f}\t'
        #               'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

    # def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
    #
    #     Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
    #
    #     # measure mAP
    #     self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])


class GCNMultiPlexNetworkEngine(MultiPlexNetworkEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # feature_var = torch.autograd.Variable(self.state('feature')).float()
        # feature_var = self.state('feature')

        feature_var = torch.autograd.Variable(self.state('feature'))
        # print(self.state('target'), type(self.state('target')))
        target_var_cpu = self.state('target').view(1, -1).transpose(1, 0).cpu()
        target_var = torch.autograd.Variable(self.state('target')).float().view(1, -1).transpose(1, 0)
        inp_var = get_inp_var(self.project)
        inp_var = torch.from_numpy(inp_var).float().detach()
        inp_var = torch.autograd.Variable(inp_var)
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

            true = self.state('target').data.cpu()

        # compute output
        gcn_output = model(feature_var, inp_var)
        self.set_state('output', gcn_output)
        target_var_cpu = torch.zeros(target_var.cpu().shape[0], gcn_output.cpu().shape[1]).scatter_(1, target_var_cpu, 1)
        self.loss = criterion(self.state('output').cpu(), target_var_cpu.cpu())

        if training:
            optimizer.zero_grad()
            self.loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            predic = gcn_output.detach().cpu().numpy().argmax(axis=1)
            train_acc = metrics.accuracy_score(true.numpy(), np.array(predic))
            self.set_state('batch_score', train_acc)

            # k = 5  # 当k>=2时
            # predic = np.argsort(gcn_output.detach().cpu().numpy(), axis=1)[:, -k:]
            # # print(predic)
            # # exit(1)
            # count = 0.0
            # for i, row in enumerate(predic):
            #     if true.numpy()[i] in row:
            #         count += 1
            # train_acc_k = count / len(true)
            # self.set_state('batch_score', train_acc_k)

            # report = metrics.classification_report(true, predic, digits=4)
            # print(report)


