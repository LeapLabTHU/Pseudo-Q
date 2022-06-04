import os
import torch
import shutil
import argparse

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('root_path', default=None, type=str)
parser.add_argument('dataset', default=None, type=str)
args = parser.parse_args()


def merge_file():

    files = os.listdir(args.root_path)
    out_file = []

    for file in files:
        if file not in ['code', args.dataset]:
            out_file += torch.load(os.path.join(args.root_path, file))

    if os.path.exists(os.path.join(args.root_path, '{}_train_pseudo.pth'.format(args.dataset))):
        os.remove(os.path.join(args.root_path, '{}_train_pseudo.pth'.format(args.dataset)))

    print('### INFO ### Length of pseudo train sample: {}'.format(len(out_file)))

    if not os.path.exists(os.path.join(args.root_path, args.dataset)):
        os.makedirs(os.path.join(args.root_path, args.dataset))
    torch.save(out_file, os.path.join(args.root_path, args.dataset, '{}_train_pseudo.pth'.format(args.dataset)))


def get_pseudo_train_number():
    files = os.listdir(args.root_path)
    for file in files:
        if os.path.isdir(os.path.join(args.root_path, file)):
            train_file = torch.load(os.path.join(args.root_path, file, 'unc', 'unc_train_pseudo.pth'))
            print('### INFO ### Length of pseudo train {} sample: {}'.format(file, len(train_file)))


if __name__ == '__main__':
    merge_file()
    # get_pseudo_train_number()
