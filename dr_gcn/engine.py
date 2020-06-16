import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from util import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
from sklearn.metrics import confusion_matrix

tqdm.monitor_interval = 0


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

        ## adding loss_recorder, accuracy_recorder, kappa_recorder to record
        if self._state('loss_recorder') is None:
            self.state['loss_recorder_train'] = []
            self.state['loss_recorder_val'] = []
        if self._state('accuracy_recorder') is None:
            self.state['accuracy_recorder_train'] = []
            self.state['accuracy_recorder_val'] = []
        if self._state('kappa_recorder') is None:
            self.state['kappa_recorder'] = []

        ## adding counting_number
        if self._state('counting_number') is None:
            self.state['counting_number'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    # when start the epoch, rest the parameters
    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    # when end the epoch, show the info during one epoch training
    # 计算loss
    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.10f}'.format(self.state['epoch'], loss=loss))
                # record the training loss to self.state['loss_recorder']
                self.state['loss_recorder_train'].append(loss)
            else:
                print('Test: \t Loss {loss:.10f}'.format(loss=loss))
                # recorder the testing loss to self.state['loss_recorder_val']
                self.state['loss_recorder_val'].append(loss)
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].data
        # TODO
        self.state['loss_batch'] = self.state['loss_batch'].cpu()
        self.state['meter_loss'].add(self.state['loss_batch'])


        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])
        # print('here is first target_var ', target_var)

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                # transforms.Resize((672, 448)),
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                # transforms.Resize((672,448)),
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                print(checkpoint['epoch'])


                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()

            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            print('hello world')
            return

        # TODO define optimizer

        # 导入预训练模型， start_epoch 为已经训练过的
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
        ## save recorder to file
        self.state['loss_recorder_train'] = np.array(self.state['loss_recorder_train'])
        np.save('checkpoints/aptos/loss_recorder_train.npy', self.state['loss_recorder_train'])
        self.state['loss_recorder_val'] = np.array(self.state['loss_recorder_val'])
        np.save('checkpoints/aptos/loss_recorder_val.npy', self.state['loss_recorder_val'])

        self.state['accuracy_recorder_train'] = np.array(self.state['accuracy_recorder_train'])
        np.save('checkpoints/aptos/accuracy_recorder_train.npy', self.state['accuracy_recorder_train'])
        self.state['accuracy_recorder_val'] = np.array(self.state['accuracy_recorder_val'])
        np.save('checkpoints/aptos/accuracy_recorder_val.npy', self.state['accuracy_recorder_val'])

        self.state['kappa_recorder'] = np.array(self.state['kappa_recorder'])
        np.save('checkpoints/aptos/kappa_recorder.npy', self.state['kappa_recorder'])
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        ## init a recorder to calculate kappa
        # self.state['kappa_pre']=[]
        # self.state['kappa_lab']=[]
        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                #    self.state['target'] = self.state['target'].cuda(async=True)
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                #    self.state['target'] = self.state['target'].cuda(async=True)
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)

# 这个class 继承子Engine类
class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()
        # init the correct number of predict label
        # init the total number of running examples
        self.state['correct_counter'] = 0
        self.state['total_counter'] = 0

        if not training:
            self.state['auc_pre']=torch.empty([4,5]).float()
            self.state['auc_lab']=torch.empty([4,5]).long()

            self.state['kappa_pre'] = torch.empty([4]).long()
            self.state['kappa_lab'] = torch.empty([4]).long()

    ##  showing the result after one epoch
    ## kappa value and accuracy
    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # define the kappa and accuracy
        Engine.on_end_epoch(self, training, model, criterion, data_loader, display=True)

        # reset the number of correct and total examples
        print('correct number is ', self.state['correct_counter'])
        print('total_number is ', self.state['total_counter'])
        self.state['accuracy_each_epoch'] = self.state['correct_counter'] / self.state['total_counter']
        ### record loss, accuracy, kappa value
        if training:
            self.state['accuracy_recorder_train'].append(self.state['accuracy_each_epoch'])
        else:
            self.state['accuracy_recorder_val'].append(self.state['accuracy_each_epoch'])
            ## record kappa value
            self.state['kappa_pre'] = self.state['kappa_pre'][4:]
            self.state['kappa_lab'] = self.state['kappa_lab'][4:]
            print('\n#################kappa value##############')
            ## qwk evaluation

            self.state['kappa_lab'] = self.state['kappa_lab']
            self.state['kappa_pre'] = self.state['kappa_pre']
            '''quadric weight'''
            print(cohen_kappa_score(self.state['kappa_lab'], self.state['kappa_pre'], weights='quadratic'))
            self.state['kappa_recorder'].append(cohen_kappa_score(self.state['kappa_lab'], self.state['kappa_pre'], weights='quadratic'))

            returnns = confusion_matrix(self.state['kappa_lab'], self.state['kappa_pre'], labels=[0,1,2,3,4])
            print('\n################# specificity and sensitivity ##############')
            print(returnns)
            specificity, sensitivity=[],[]
            for i in range(5):
                tn = returnns[0,0]+returnns[1,1]+returnns[2,2]+returnns[3,3]+returnns[4,4]-returnns[i,i]
                fn = returnns[i,0]+returnns[i,1]+returnns[i,2]+returnns[i,3]+returnns[i,4]-returnns[i,i]
                tp = returnns[i,i]
                fp = returnns[0,i]+returnns[1,i]+returnns[2,i]+returnns[3,i]+returnns[4,i]-returnns[i,i]

                specificity.append(float(tn/(tn+fp)))
                if tp+fn == 0:
                    sensitivity.append(float(1))
                else:
                    sensitivity.append(float(tp/(tp+fn)))
            print('specificity and sensitivity : ',specificity, ' ', sensitivity)

            # print('auc for 5 class', roc_auc)
            '''linear without quatdic '''
            # print(cohen_kappa_score(self.state['kappa_lab'], self.state['kappa_pre']))
            # self.state['kappa_recorder'].append(cohen_kappa_score(self.state['kappa_lab'], self.state['kappa_pre']))

        self.state['correct_counter'] = 0
        self.state['total_counter'] = 0
        print('#### for epoch the accuracy is ', self.state['accuracy_each_epoch'])
        return self.state['accuracy_each_epoch']

    # def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
    #     map = 100 * self.state['ap_meter'].value().mean()
    #     loss = self.state['meter_loss'].value()[0]
    #     OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
    #     OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
    #     if display:
    #         if training:
    #             print('Epoch: [{0}]\t'
    #                   'Loss {loss:.4f}\t'
    #                   'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
    #             print('OP: {OP:.4f}\t'
    #                   'OR: {OR:.4f}\t'
    #                   'OF1: {OF1:.4f}\t'
    #                   'CP: {CP:.4f}\t'
    #                   'CR: {CR:.4f}\t'
    #                   'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    #         else:
    #             print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
    #             print('OP: {OP:.4f}\t'
    #                   'OR: {OR:.4f}\t'
    #                   'OF1: {OF1:.4f}\t'
    #                   'CP: {CP:.4f}\t'
    #                   'CR: {CR:.4f}\t'
    #                   'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    #             print('OP_3: {OP:.4f}\t'
    #                   'OR_3: {OR:.4f}\t'
    #                   'OF1_3: {OF1:.4f}\t'
    #                   'CP_3: {CP:.4f}\t'
    #                   'CR_3: {CR:.4f}\t'
    #                   'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
    #
    #     return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        '''this part is weakpoint'''
        # self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

# 这个类继承自MultiLabelMAPEngine
class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()

        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        # print(self.state['input'].shape)
        # print('inp_var form engine is ',inp_var.shape)
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True


        # # compute output
        # print('feature_var from engine is ', feature_var.shape)
        self.state['output'] = model(feature_var, inp_var)
        target_var = target_var.narrow(1,0,1)
        # show the result of network
        # test on accuracy
        # TODO 添加record_accuracy来记录准确率
        softmax = nn.Softmax(dim=1)
        pre_label = softmax(self.state['output'])
        pre_label = torch.max(pre_label,1)[1]
        # record the correct number of predicted label
        target_var2 = target_var.long()
        self.state['correct_counter'] += (pre_label == target_var2.squeeze(1)).sum().item()
        self.state['total_counter'] += target_var2.size(0)


        if not training:
            # record auc_pre and auc_lab
            # path = 'checkpoint/auc_files/'
            # path1 = 'checkpoint/auc_file2/'
            # val模式下记录kappa_lab 和 kappa_pre  用cat拼接到一起
            # y_label = label_binarize(target_var2.cpu(), classes=[0,1,2,3,4])
            # y_label = torch.from_numpy(y_label)
            # torch.save(y_label, path+str(self.state['counting_number'])+'file.pt')
            # torch.save(self.state['output'], path1+str(self.state['counting_number'])+'file.pt')
            # self.state['counting_number'] += 1
            # self.state['auc_pre'] = torch.cat([self.state['auc_pre'],self.state['output'].cpu()], 0)
            # self.state['auc_pre'] = torch.cat([self.state['auc_pre'], self.state['output'].cpu()], 0)
            # self.state['auc_lab'] = torch.cat([self.state['auc_lab'], y_label], 0)

            self.state['kappa_pre'] = torch.cat([self.state['kappa_pre'], pre_label.cpu()], 0)
            self.state['kappa_lab'] = torch.cat([self.state['kappa_lab'], target_var2.squeeze(1).cpu()], 0)



        # TODO
        #
        self.state['loss'] = criterion(self.state['output'], target_var2.squeeze(1))

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        # self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]
