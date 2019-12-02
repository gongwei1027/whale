import argparse
from engine import *
from models import *
from textcnn import *
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_eclipse():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    cur_dir = os.getcwd()
    dataset = '{}/NetworkData/'.format(cur_dir)
    embedding = 'random'
    use_gpu = torch.cuda.is_available()

    config = Config(dataset, embedding, "eclipse")
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, True)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    model = gcn_resnet101(filename="eclipse", config=config)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {
        'max_epochs': 20,
        'evaluate': args.evaluate,
        'resume': args.resume,
        'worker': 25,
        'epoch': 0,
    }
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/coco/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiPlexNetworkEngine(**state)
    engine.learning(model, criterion, train_iter, dev_iter, optimizer)

if __name__ == '__main__':
    main_eclipse()
