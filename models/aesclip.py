# -*- coding: utf-8 -*-
"""
Created on 2024/3/14 17:13

@DESCRIPTION: 

@author: Xiangfei
"""
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import models.clip as clip
# import clip

class AesCLIP(nn.Module):
    def __init__(self, clip_name, fusion='mlp',normalization_sign=True):
        super(AesCLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_size = self.select_clip(clip_name)
        self.clip_model = clip_model.float()
        self.clip_size = clip_size['feature_size']
        self.mlp_compose = nn.Linear(1024, 512)
        self.trans_compose = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8), 6)
        self.normalization = normalization_sign
        self.fusion = fusion

    def select_clip(self, clip_name):
        param = {'feature_size': 512}
        if clip_name == 'RN50':
            clip_model, _ = clip.load("RN50", device=self.device)
            param['feature_size'] = 1024
        elif clip_name == 'ViT-B/16':
            clip_model, _ = clip.load("ViT-B/16", device=self.device)
            param['feature_size'] = 768

        else:
            raise IOError('model type is wrong')
        return clip_model, param

    def forward(self, x, text0, text1):
        # img_embedding
        img_embedding = self.clip_model.visual(x)
        img_embedding = img_embedding @ self.clip_model.visual.proj
        # text_embedding0
        text_tokens0 = torch.cat([clip.tokenize(text) for text in text0])
        text_embedding0 = self.clip_model.encode_text(text_tokens0.to(self.device)).float()
        # text_embedding1
        text_tokens1 = torch.cat([clip.tokenize(text) for text in text1])
        text_embedding1 = self.clip_model.encode_text(text_tokens1.to(self.device)).float()
        if self.normalization:
            img_embedding_norm = F.normalize(img_embedding, dim=1)
            text_embedding_norm = F.normalize(text_embedding0, dim=1)
            residual = torch.cat((img_embedding_norm, text_embedding_norm), 1)
        else:
            residual = torch.cat((img_embedding, text_embedding0), 1)

        if self.fusion == 'mlp':
            residual = self.mlp_compose(residual)
        elif self.fusion == 'trans':
            residual = self.mlp_compose(residual)
            residual = self.trans_compose(residual)
        composition_embedding = residual + img_embedding

        return composition_embedding, text_embedding1

class AesCLIP_reg(nn.Module):
    def __init__(self, clip_name,  weight):
        super(AesCLIP_reg, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_size = self.select_clip(clip_name)
        print('Loading AesCLIP weights:', clip_model.load_state_dict(torch.load(weight)))
        self.aesclip = clip_model.float()
        self.clip_size = clip_size['feature_size']
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_size, 10),
            nn.Softmax()
        )
    def select_clip(self, clip_name):
        param = {'feature_size': 768}
        if clip_name == 'RN50':
            clip_model, _ = clip.load("RN50", device=self.device)
            param['feature_size'] = 1024
        elif clip_name == 'ViT-B/16':
            clip_model, _ = clip.load("ViT-B/16", device=self.device)
        else:
            raise IOError('model type is wrong')
        return clip_model, param

    def forward(self, x):
        # img_embedding
        img_embedding = self.aesclip.visual(x)
        a = self.mlp(img_embedding)

        return a

class zs_AesCLIP(nn.Module):
    def __init__(self, clip_name, weight):
        super(zs_AesCLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = self.select_clip(clip_name)
        print('Loading AesCLIP weights:', clip_model.load_state_dict(torch.load(weight)))
        self.clip_model = clip_model.float()

    def select_clip(self, clip_name):
        param = {'feature_size': 768}
        if clip_name == 'RN50':
            clip_model, _ = clip.load("RN50", device=self.device)
            param['feature_size'] = 1024
        elif clip_name == 'ViT-B/16':
            clip_model, _ = clip.load("ViT-B/16", device=self.device)
        else:
            raise IOError('model type is wrong')
        return clip_model, param

    def forward(self, x, texts):
        pred_score = []
        with torch.no_grad():
            img_embedding = self.clip_model.visual(x)
            img_embedding = img_embedding @ self.clip_model.visual.proj
            text_tokens = torch.cat([clip.tokenize(text) for text in texts])
            text_embedding = self.clip_model.encode_text(text_tokens.to(self.device)).float()
            img_embedding /= img_embedding.norm(dim=-1, keepdim=True)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_embedding @ text_embedding.T).softmax(dim=-1)
            for item in similarity:
                pred_score.append(item[0])
            return pred_score