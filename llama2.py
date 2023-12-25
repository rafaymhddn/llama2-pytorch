import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int =4086
    n_layers: int = 32
    n_heads:int = 32 # for Q
    n_kv_heads: Optional[int] = None # for K an V
    vocab_size: int = -1 # set with the tokenizer
    multiple_of: int = 256 # for ff layer
    ffn_dim_multiple: Optional[float] = None # to increase no of params, GQA reduces the head, allows for comparision 
    norm_eps: float = 1e-5

    # For KV Catch
    max_batch_size: int = 32
    max_seq_len:int = 2048

    device: str = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float= 1e-6):
        super().__init__()
        self.eps = eps 
        # gamma param
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) = (B, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        # (dim) *(B, seq_len, dim) = (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4* args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiple is not None:
            hidden_dim = int(args.ffn_dim_multiple * hidden_dim)
    
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))  # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)            # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V             # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = self.w2(x)              # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        return x
    

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # as mention in the paper embd must be even
    assert head_dim % 2 ==0, "Dimenstions must be even"
    
    # building thera parametes
    # formula: theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2, ... dim / 2]
    # Shape : (head_dim / 2) -> embedding applied to multiple heads after splitting for each head
    theta_numerator = torch.arange(0, head_dim,2).float()
    theta = 1.0/(theta ** (theta_numerator / head_dim)).to(device)
    
    # construct the position ("" param)
    m = torch.arange(seq_len,device=device)

    # Multiple each theta by each postion of outer product
    # Shape: (seq_len) x (head_dim/2) = (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()

    # Converting to complex form i.e. polar coord c = R * exp(1 * m * theta), where R=1
    # Shape: (seq_len, gead_dim/2) -> (seq_len, head_dim/2) 
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # shape: (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim/2) = (B, seq_len, h, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # B, seq_len, h, head_dim/2) -> (B, seq_len, h, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, h, head_dim/2, 2) -> (B, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # Assume x is of shape (batch_size, seq_len, n_kv_heads, head_dim)
        x = x[:, :, :, None, :]  # Add a new dimension for n_rep: (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x_expanded = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # Expand to (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        x_reshaped = x_expanded.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # Flatten: (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
    
        return x_reshaped


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) :
        super().__init__()

        # no of heads for key and value
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        #self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # no of heads for query
        self.n_heads_q = args.n_heads
        # no of times head of key and value should be repeated to match queries
        self.n_rep = self.n_heads_q//self.n_kv_heads
        # dimension of each head
        self.head_dim = args.dim//args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads *  self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *  self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *  self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim,   args.dim, bias=False)

        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x

        else:
            # Assume x is of shape (batch_size, seq_len, n_kv_heads, head_dim)
            x = x[:, :, :, None, :]  # Add a new dimension for n_rep: (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
            x_expanded = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # Expand to (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
            x_reshaped = x_expanded.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # Flatten: (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)

            return x_reshaped

    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape #(B, 1, Dim)

        # apply wk, wq, wv matrics to queries, keys and values
        #(B, 1, dim) -> (B, 1, h_q & head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, h_kv * dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary pos embedding to Q and V
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, device=x.device)

        # KV Cache -> append at end of key and value, will be use at next iteration
        # for every batch replace start_pos : start_pos + seq_len tokens
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Now dot produce (1) Q with (all)K^T = QK^t with (all) V = (1) attention token

        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]   # (B, seq_len_kv, h_kv, head_dim)
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len] # (B, seq_len_kv, h_kv, head_dim)

        # Grouped-query: no of head for queries can be more than for key and vaue (only in 70B)
        # Multhead attention work around repeate(n_rep = n_heads_q//n_kv_heads/) k and v to reach the no of q 

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Move head before seq dimension as each head will watch all the seq and part of embd

        xq = xq.transpose(1,2)          # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        keys = keys.transpose(1, 2)     # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2) # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)

        # Self attention 
        
        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # Take all heads, concatinate and multiply ny Wo = MHA
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim) ~ remove dimension fro head
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)



    

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim//args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalize before self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # normalize before feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex:torch.Tensor):
        # (B, seq_len, dim)
        #h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)  # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)m(x), start_pos, freq_complex)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "please set vocab size, based on tokenzier"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps = args.norm_eps)

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, 
                                                              self.args.max_seq_len*2, device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int): # KV catch seq length is always 1 as previous precomputed token are cached
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, seq_len) -> (B, seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retriece the pairs (m, theta) correspounding to the pos [start_pos, start_pos + seq_len]
        freq_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        # consecutivelt apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freq_complex)
        h = self.norm(h)

        output = self.output(h).float()

        return output