# MIT License
#
# Copyright (c) 2020 Hila Chefer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Transformer-LRP ViT of Chefer, Gur & Wolf, "Transformer Interpretability
Beyond Attention Visualization", CVPR 2021.

Vendored from github.com/hila-chefer/Transformer-Explainability (MIT license,
notice above):

* ``modules/layers_ours.py``: the LRP layers (the subset used by the ViT);
* ``baselines/ViT/ViT_LRP.py``: the ViT with relevance propagation (its ViT
  code is derived from timm, "Hacked together by / Copyright 2020 Ross
  Wightman");
* ``baselines/ViT/ViT_explanation_generator.py``: ``LRP.generate_LRP``.

The relevance rules and the attribution computation are unchanged.  Changes
to the source files: ``einops.rearrange`` is written as the equivalent
``reshape``/``permute`` and ``to_2tuple`` is inlined;
``torch.nn.init.trunc_normal_`` replaces the copied initialiser;
``generate_LRP`` places the one-hot seed on the device of the model output
instead of calling ``.cuda()``; the pretrained-weight factories and some
developer notes are omitted.  Weights are loaded by the caller
(see :mod:`bwgame.xai.methods.tibav`).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# --------------------------------------------------------------------------
# modules/layers_ours.py
# --------------------------------------------------------------------------

def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:  # noqa: E712
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs


class GELU(nn.GELU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)


class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:  # noqa: E712
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs


class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R


# --------------------------------------------------------------------------
# baselines/ViT/ViT_LRP.py
# --------------------------------------------------------------------------

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        # rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        q, k, v = qkv.reshape(b, n, 3, h, -1).permute(2, 0, 3, 1, 4)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        # rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        # rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)
        b, n = cam.shape[:2]
        cam = cam.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        # rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
        cam_qkv = torch.stack([cam_q, cam_k, cam_v]).permute(1, 3, 0, 2, 4).reshape(b, n, -1)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                          (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        # Transformer attribution of Chefer et al. (CVPR 2021)
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam

        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


# --------------------------------------------------------------------------
# baselines/ViT/ViT_explanation_generator.py
# --------------------------------------------------------------------------

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:  # noqa: E711
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(output.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)
