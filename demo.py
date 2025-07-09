import torch
import numpy as np
import warnings
import models.clip as clip
warnings.filterwarnings("ignore")
from models.aesclip import AesCLIP_reg, zs_AesCLIP
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--path_to_images', type=str, default='AVA/images',
                        help='directory to images')
    args = parser.parse_args()
    return args

opt = init()

def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.cuda()

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def predict(opt):
    model = AesCLIP_reg(clip_name='ViT-B/16', weight="./pretrained_weights/AesCLIP")
    model.load_state_dict(torch.load("./pretrained_weights/IAA_weight"))
    zs_model = zs_AesCLIP(clip_name='ViT-B/16', weight='./pretrained_weights/AesCLIP')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    zs_model.to(device)

    model.eval()
    zs_model.eval()
    _, preprocess = clip.load('ViT-B/16', device)
    image = Image.open('img/example.jpg').convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    ## model score
    aes_score = model(image_input)
    __, aes_score = get_score(opt, aes_score)
    aes_score = aes_score[0]
    print('Predicted Aesthetic Score: ', aes_score)
    ## Zero-shot score
    zs_aes_score = zs_model(image_input, ['good image', 'bad image'])[0].item()*10

    print('Predicted Zero-shot Aesthetic Score: ', zs_aes_score)



if __name__ == "__main__":
    predict(opt)