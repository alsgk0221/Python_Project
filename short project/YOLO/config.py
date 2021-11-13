'''
모델에 기본적으로 학습을 시키기 위해 parameter들을 setting
lr weight 나 epoch 과 같은 값들을 setting 한다.
'''

import argparse
def parse_args():

    parser = argparse.ArgumentParser(description='Pytorch implementation of Classification models.')

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64*5)
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--lr_patience', type=int, default=10)

    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--save_location', type=str, default='/data/fire/')
    parser.add_argument('--load_model', type=bool,default=False)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)

    parser.add_argument('--class_number', type=int, default=2)
    parser.add_argument('--summary_location', type=str, default='./summary')

    return parser.parse_args()
