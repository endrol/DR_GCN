import argparse
from engine import *
from models import *
from dr import *
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', metavar='DIR',default='/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/aptos',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='checkpoints/aptos/model_best.pth.tar',
#                     type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# to train a whole new model
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_dr():
    warnings.filterwarnings('ignore')
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    '''for those dataset are big'''
    csv_file = pd.read_csv('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/aptos/train.csv')

    x,y = csv_file.iloc[:,0],csv_file.iloc[:,1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


    #define datasets
    train_dataset = DRclassification(args.data, 'train_images',
                                     set2='2. Groundtruths/a. IDRiD_Disease Grading_Training Labels',train_data=x_train,label_data=y_train,
                                     inp_name='descripion/feature_vector.pkl',)
    val_dataset = DRclassification(args.data, 'train_images',
                                   set2='2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels',train_data=x_test,label_data=y_test,
                                   inp_name='descripion/feature_vector.pkl')

    # num_classes defines what?
    # define the cluster numbers, label numbers
    num_classes = 20

    # load model
    # and load adj file for adjacent information
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='descripion/dr_adj.pkl')

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size,'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    # 保存训练的模型
    state['save_model_path'] = 'checkpoints/aptos/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

def calculate_auc():
    auc_lab = next(os.walk('/media/endrol/datafile/pycharmProjects/from_linux/ml_gcn/checkpoint/auc_files'))[2]
    auc_pre = next(os.walk('/media/endrol/datafile/pycharmProjects/from_linux/ml_gcn/checkpoint/auc_file2'))[2]

    auc_lab.sort()
    auc_pre.sort()
    print(len(auc_pre), len(auc_lab))
    auc_pre_value = torch.empty([4,5]).float()
    auc_lab_value = torch.empty([4,5]).long()
    for pre_tensor in auc_pre:
        tensor = torch.load('/media/endrol/datafile/pycharmProjects/from_linux/ml_gcn/checkpoint/auc_file2/{}'.format(pre_tensor))
        auc_pre_value = torch.cat([auc_pre_value,tensor.cpu()],0)
    for lab_tensor in auc_lab:
        tensor = torch.load('/media/endrol/datafile/pycharmProjects/from_linux/ml_gcn/checkpoint/auc_files/{}'.format(lab_tensor))
        auc_lab_value = torch.cat([auc_lab_value, tensor],0)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(auc_lab_value[:, i].detach().numpy(), auc_pre_value[:, i].detach().numpy())
        roc_auc[i] = auc(fpr[i], tpr[i])

    print('roc_auc for each class ',roc_auc)

def see_logical():
    kappa_pre = torch.load('/media/endrol/datafile/pycharmProjects/from_linux/ml_gcn/checkpoint/aptos/kappa_pre.pt')
    kappa_lab = torch.load('/media/endrol/datafile/pycharmProjects/from_linux/ml_gcn/checkpoint/aptos/kappa_lab.pt')

    a = kappa_lab.detach().numpy()
    print(a[np.where(a == 0)])
    print(kappa_pre)

if __name__ == '__main__':
    main_dr()
    # calculate_auc()
    # see_logical()
