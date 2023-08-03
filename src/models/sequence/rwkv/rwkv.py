########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import logging
from types import SimpleNamespace

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

logger = logging.getLogger(__name__)

RWKV_HEAD_QK_DIM = 0


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y
        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
else:
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16()
        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)

########################################################################################################

class RWKV_TimeMix_RWKV5_Preview(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        self.head_size = 64
        self.n_head = self.n_embd // self.head_size
        assert self.n_embd % self.n_head == 0

        self.chunk_len = 512
        assert self.ctx_len % self.chunk_len == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -8 + 7 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            self.time_first = nn.Parameter(torch.ones(self.n_head) * (-3.0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, self.n_embd)

    @MyFunction
    def jit_func(self, x):
        B, TT, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
        v = self.value(xv).view(B, TT, self.n_head, self.head_size).transpose(1, 2)                 # BTC -> BHTS

        return r, k, v

    @MyFunction
    def jit_func_2(self, r, k, v, w, wk, wb, ws):
        B, H, TT, S = r.size()
        T = self.chunk_len

        s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
        x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

################################################################################
########
        for i in range(TT // T):
            rr = r[:, :, i*T:i*T+T, :]
            kk = k[:, :, :, i*T:i*T+T]
            vv = v[:, :, i*T:i*T+T, :]

            x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

            s = ws * s + (kk * wk) @ vv
########
################################################################################
        
        x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
        x = self.ln_x(x).view(B, TT, H*S)
        return self.output(x)
    
    def forward(self, x):
        H = self.n_head
        T = self.chunk_len

        r, k, v = self.jit_func(x)

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
        u = torch.exp(self.time_first.float()).unsqueeze(-1)

################################################################################
########
        ws = w.pow(T).reshape(1, H, 1, 1)

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(1, H, T, T)
########
################################################################################

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        return self.jit_func_2(r, k, v, w, wk, wb, ws)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            
            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if 'a' in os.environ["RWKV_MY_TESTING"]:
            self.register_buffer("att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
            d_qkv = args.n_embd // 16
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)
            with torch.no_grad():
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)

    if 'a' not in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def jit_func(self, x):
            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            return sr, k, v

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v = self.jit_func(x)
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            return self.output(rwkv)

    if 'a' in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def QKV(self, q, k, v):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.att_mask == 0, float('-inf'))
            att = F.softmax(att, dim = -1)
            x = att @ v
            return x

        @MyFunction
        def jit_funcQKV(self, x):
            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            qq = self.qq(xqq)
            kk = self.kk(xkk)
            vv = self.vv(xvv)
            return sr, k, v, qq, kk, vv

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v, qq, kk, vv = self.jit_funcQKV(x)
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
            return rwkv

########################################################################################################

class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'r' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_TimeMix_RWKV5_Preview(args, layer_id)
            else:
                self.att = RWKV_TimeMix(args, layer_id)

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


def RWKV_Init(model, args):  # fancy initialization of all lin & emb layer in the model
    print("\n[--> first run, init model params (very slow for large models) <--]")
    print("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  # final projection?
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            # print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {name}")

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
    

class RWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        if type(config) is dict:
            config = SimpleNamespace(**config)
        self.config = config

        # self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.d_model = self.config.n_embd
        self.blocks = nn.ModuleList([Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        #self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.head = nn.Linear(config.n_embd, config.n_embd , bias=False)

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(
                torch.ones(config.ctx_len, config.ctx_len)))

        # self.ctx_len = config.ctx_len

        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config) 
        except:
            pass

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    # def get_ctx_len(self):
        # return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, state=None, **kwargs):
        #        idx = idx.to(self.emb.weight.device)

        self.step += 1
        # B, T = idx.size()
        # __import__('remote_pdb').set_trace()
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)

        # if RWKV_HEAD_QK_DIM > 0:
        #     q = self.head_q(x)[:, :T, :]
        #     k = self.head_k(x)[:, :T, :]
        #     c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
        #     c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            
        #     if '32' in os.environ['RWKV_FLOAT_MODE']:
        #         c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)
        #     elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
        #         c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()
        #     elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
        #         c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).bfloat16()

        return x, state
        #return L2Wrap.apply(loss, x)

    @property
    def d_state(self):
        return self.n_layers * self.d_model

    @property
    def d_output(self):
        return self.d_model

# class RWKV(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.args = config
#         if not hasattr(config, 'dim_att'):
#             config.dim_att = config.n_embd
#         if not hasattr(config, 'dim_ffn'):
#             config.dim_ffn = config.n_embd * 4
#         if not hasattr(config, 'tiny_att_layer'):
#             config.tiny_att_layer = -1
#         if not hasattr(config, 'tiny_att_dim'):
#             config.tiny_att_dim = -1

#         self.emb = nn.Embedding(config.vocab_size, config.n_embd)

#         self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

#         self.ln_out = nn.LayerNorm(config.n_embd)
#         self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         if config.head_qk > 0:
#             self.head_q = nn.Linear(config.n_embd, config.head_qk, bias=False)
#             self.head_k = nn.Linear(config.n_embd, config.head_qk, bias=False)
#             self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
    
#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear)):
#             module.weight.data.normal_(mean=0.0, std=0.01)
#         if isinstance(module, (nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=1e-5)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

#     def forward(self, idx):
#         args = self.args
#         B, T = idx.size()
#         assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

#         x = self.emb(idx)
#         x_emb = x

#         if args.tiny_att_dim > 0:
#             for block in self.blocks:
#                 if args.grad_cp == 1:
#                     x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
#                 else:
#                     x = block(x, x_emb)
#         else:
#             for block in self.blocks:
#                 if args.grad_cp == 1:
#                     x = deepspeed.checkpointing.checkpoint(block, x)
#                 else:
#                     x = block(x)

#         x = self.ln_out(x)

#         if args.head_qk > 0:
#             q = self.head_q(x)[:, :T, :]
#             k = self.head_k(x)[:, :T, :]
#             c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
#             c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

#             if "32" in os.environ["RWKV_FLOAT_MODE"]:
#                 c = c @ F.one_hot(idx, num_classes=args.vocab_size)
#             elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
#                 c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
#             elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
#                 c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

#             x = self.head(x) + c
#         else:
#             x = self.head(x)

#         return x