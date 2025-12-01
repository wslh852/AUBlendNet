import torch
import torch.nn as nn
from models.lib.wav2vec import Wav2Vec2Model
from models.utils import init_biased_mask, enc_dec_mask, PeriodicPositionalEncoding
from base import BaseModel
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import os
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class AUBlendNet(BaseModel):
    def __init__(self, args):
        super(AUBlendNet, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.args = args

        self.feature_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.onehot_feature_map = nn.Linear(1, args.AU_num+1)
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 8, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=args.n_head, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_layers)
        self.blocks = nn.ModuleList([
            DiTBlock(args.feature_dim, args.n_head, mlp_ratio=2.0) for _ in range(args.num_layers)
        ])
        # motion decoder
        self.feat_map = nn.Linear(args.feature_dim, args.face_quan_num*args.zquant_dim, bias=False)
        # style embedding
        self.learnable_style_emb = nn.Linear(args.vertice_dim, args.feature_dim)
 
        self.device = args.device
        nn.init.constant_(self.feat_map.weight, 0)
        # nn.init.constant_(self.feat_map.bias, 0)

        from models.stage1 import VQAutoEncoder
     
        self.autoencoder = VQAutoEncoder(args)
        self.autoencoder.load_state_dict(torch.load(args.vqvae_pretrained_path)['state_dict'])
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        

    def forward(self, template, data, onehot, criterion):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1,V*3)

        # style embedding
        obj_embedding = self.learnable_style_emb(template) # 1024
        #obj_embedding = obj_embedding.unsqueeze(1)
        xt_input = template.permute(0,2,1)
        xt_input = self.onehot_feature_map(xt_input)
        xt_input = xt_input.permute(0,2,1)
        xt_input = self.feature_map(xt_input) # 1024

        # gt motion feature extraction
        feat_q_gt, _ = self.autoencoder.get_quant(data - template)
        feat_q_gt = feat_q_gt.permute(0,2,1)

        for block in self.blocks:
            xt_input = block(xt_input , obj_embedding[0])  
        feat_out = self.feat_map(xt_input)

        feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1]*self.args.face_quan_num, -1)
        
        # feature quantization
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)

        # feature decoding
        vertice_out = self.autoencoder.decode(feat_out_q)
        vertice_out = vertice_out + template

        # loss 
        loss_motion = criterion(vertice_out, data) # (batch, seq_len, V*3)
        loss_reg = criterion(feat_out, feat_q_gt.detach())

        return self.args.motion_weight*loss_motion + self.args.reg_weight*loss_reg, [loss_motion, loss_reg]



    def predict(self, template, data, onehot):
        template = template.unsqueeze(1) # (1,1,V*3)
        # style embedding
        obj_embedding = self.learnable_style_emb(template) # 1024
        #obj_embedding = obj_embedding.unsqueeze(1)
        xt_input = template.permute(0,2,1)
        xt_input = self.onehot_feature_map(xt_input)
        xt_input = xt_input.permute(0,2,1)
        xt_input = self.feature_map(xt_input)

        feat_q_gt, _ = self.autoencoder.get_quant(data - template)
        feat_q_gt = feat_q_gt.permute(0,2,1)
        for block in self.blocks:
            xt_input = block(xt_input , obj_embedding[0])  
        feat_out = self.feat_map(xt_input)

        feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1]*self.args.face_quan_num, -1)
        
        # feature quantization
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)

        # feature decoding
        vertice_out = self.autoencoder.decode(feat_out_q)
        vertice_out = vertice_out + template
        return vertice_out
