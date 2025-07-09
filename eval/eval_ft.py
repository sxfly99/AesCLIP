import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
from models.aesclip import AesCLIP_reg
from dataset import AVA
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--path_to_images', type=str, default='/AVA/images',
                        help='directory to images')

    parser.add_argument('--path_to_save_csv', type=str,
                        default="./data",
                        help='directory to csv_folder')

    parser.add_argument('--experiment_dir_name', type=str,
                        default='./result/ava/',
                        help='directory to project')

    parser.add_argument('--batch_size', type=int, default=64, help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers',
                        )
    args = parser.parse_args()
    return args


opt = init()


def adjust_learning_rate(params, optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = params.init_lr * (0.5 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')

    train_ds = AVA(train_csv_path, opt.path_to_images, if_train=True)
    test_ds = AVA(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader



def validate(opt, model, loader):
    model.eval()
    model.cuda()
    true_score = []
    pred_score = []
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.type(torch.FloatTensor)
        y = y.cuda()
        with torch.no_grad():
            y_pred = model(x)
        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()


    lcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    print('PLCC', lcc_mean[0])
    print('SRCC', srcc_mean[0])

    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)
    print('ACC', acc)
def start_eval(opt):
    train_loader, test_loader = create_data_part(opt)
    model = AesCLIP_reg(clip_name='ViT-B/16', weight='./pretrained_weights/AesCLIP')
    model.load_state_dict(torch.load('./pretrained_weights/IAA_weight'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model = model.cuda()
    validate(opt, model=model, loader=test_loader)

if __name__ == "__main__":
    start_eval(opt)