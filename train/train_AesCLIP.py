import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from models.aesclip import AesCLIP
from dataset import AVA_MAV
from utils import AverageMeter, InfoNCE
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

import nni
from nni.utils import merge_parameter

def init():
    parser = argparse.ArgumentParser(description="Train AesCLIP_weight")

    parser.add_argument('--path_to_images', type=str, default='AVA/images/',
                        help='directory to images')
    parser.add_argument('--path_to_save_json', type=str, default="./attributes_comments/",
                        help='directory to csv_folder')
    parser.add_argument('--experiment_dir_name', type=str, default='./pretrained_weights/',
                        help='directory to project')

    parser.add_argument('--init_lr', type=int, default=6e-5, help='learning_rate'
                        )
    parser.add_argument('--num_comments', type=int, default=4, help='num of aesthetics comments'
                        )

    parser.add_argument('--temperature', default=0.07, type=float,
                    help='temperature for contrastive learning')
    parser.add_argument('--n_ctx', default=6, type=int,
                    help='length of context')

    parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
    parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
    parser.add_argument("--num_epoch", default=20, type=int,
                help="epochs of training")
    parser.add_argument("--batch_size", default=64, type=int,
                help="batch size of training")
    parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')   
    args = parser.parse_args()
    return args



def adjust_learning_rate(params, optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = params['init_lr'] * (0.5 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_score(opt,y_pred):
    w = torch.from_numpy(np.linspace(1,10, 10))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    train_json_path = os.path.join(opt['path_to_save_json'], 'train.json')
    test_json_path = os.path.join(opt['path_to_save_json'], 'test.json')

    train_ds = AVA_MAV(train_json_path, opt['path_to_images'], if_train=True)
    test_ds = AVA_MAV(test_json_path, opt['path_to_images'], if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, test_loader


def train(opt, epoch, model,loader, optimizer, criterion_it, writer=None, global_step=None, name=None):
    model.train()
    train_losses_i = AverageMeter()
    param_num = 0
    for param in model.parameters():
        if param.requires_grad==True:
            param_num += int(np.prod(param.shape))
    print('Trainable params: %.4f million' % (param_num / 1e6))
    for idx, (x, y, t0, t1) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.cuda()
        
        img_embedding, text_embedding = model(x, t0, t1)
        # 对比loss
        loss = criterion_it(img_embedding, text_embedding) + criterion_it(text_embedding, img_embedding)

        # loss回传
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if idx % 500 == 0:
            f.write(
            '| epoch:%d | Batch:%d | loss:%.3f \r\n'
            % (epoch, idx, loss.item()))
            f.flush()

        train_losses_i.update(loss.item(), x.size(0))
    
    return train_losses_i.avg


def validate(opt, model, loader, optimizer,criterion_it, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses_i = AverageMeter()
    for idx, (x, y, t0, t1) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            img_embedding, text_embedding = model(x, t0, t1)
        # 对比loss
        loss = criterion_it(img_embedding, text_embedding) + criterion_it(text_embedding, img_embedding)
        validate_losses_i.update(loss.item(), x.size(0))

    return validate_losses_i.avg


def start_train(opt):
    log_name = 'AesCLIP.txt'
    log_txt = os.path.join(opt['experiment_dir_name'], log_name)
    global f
    f = open(log_txt, 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_data_part(opt)
    model = AesCLIP(clip_name='ViT-B/16', fusion='trans', normalization_sign=True)
    model.to(device)
    model.train()
    # TODO: change optimizer
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-3, betas=(0.9, 0.999), lr=opt['init_lr'], weight_decay=0.01)

    criterion_it = InfoNCE(temperature=opt['temperature'])
    criterion_it.cuda()
    # criterion_ix.cuda()
    best_test_loss = float('inf')

    writer = SummaryWriter(log_dir=os.path.join(opt['experiment_dir_name'], 'logs_a'))

    for e in range(opt['num_epoch']):
        optimizer = adjust_learning_rate(opt, optimizer, e)
        print("*******************************************************************************************************")
        print("第%d个epoch的学习率：%f" % (1+e, optimizer.param_groups[0]['lr']))
        train_loss = train(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer,
                           criterion_it=criterion_it,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt['experiment_dir_name']}_by_batch")

        test_loss = validate(opt, model=model, loader=test_loader, optimizer=optimizer,
                            criterion_it=criterion_it,
                            writer=writer, global_step=len(test_loader) * e,
                            name=f"{opt['experiment_dir_name']}_by_batch")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_name = 'AesCLIP_weight--e{:d}-train{:.4f}-test{:.4f}'.format(e + 1, train_loss, test_loss)
            torch.save(model.clip_model.state_dict(), os.path.join(opt['experiment_dir_name'], model_name))
        print('Best Test Loss:', best_test_loss)
        f.write(
            'epoch:%d, train_loss:%.5f,test_loss:%.5f\r\n'
            % (e, train_loss, test_loss))
        f.flush()

        writer.add_scalars("epoch_loss", {'train': train_loss, 'test': test_loss},
                           global_step=e)
    writer.close()
    f.close()


if __name__ =="__main__":
    opt = init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.filterwarnings('ignore')
    # print(os.getcwd())
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(opt, tuner_params))
    print(params)
    
    start_train(params)

    # start_train(opt)
    # f.close()
