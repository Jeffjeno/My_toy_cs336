import torch
import torch.nn as nn
from einops import rearrange,einsum
class Multi_Head_Self_attention(nn.Module):
    def __init__(self,d_model,num_heads,max_seq_len : int | None = None, theta : float | None = None):
        """
        Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k
        self.Q_weights = nn.Parameter(torch.empty(self.d_model,self.d_model))
        self.K_weights = nn.Parameter(torch.empty(self.d_model,self.d_model))
        self.V_weights = nn.Parameter(torch.empty(self.d_model,self.d_model))
        self.O_weights = nn.Parameter(torch.empty(self.d_model,self.d_model))

        if max_seq_len and theta:
            self.max_seq_len = max_seq_len 
            self.theta = theta
            from cs336_basics.rope import RoPE
            self.rope = RoPE(self.theta,self.d_k,self.max_seq_len)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
        
        seq = x.shape[-2]
        # b s d_model
        Q = einsum(self.Q_weights,x,"d_a d_b , ... seq d_b -> ... seq d_a") # 2bsd2
        K = einsum(self.K_weights,x,"d_a d_b , ... seq d_b -> ... seq d_a")
        V = einsum(self.V_weights,x,"d_a d_b , ... seq d_b -> ... seq d_a")
        if token_positions is not None:
            batched_Q = self.rope.forward(rearrange(Q,"... seq (h d_k) -> ... h seq d_k",h=self.num_heads,d_k = self.d_k),token_positions)
            batched_K = self.rope.forward(rearrange(K,"... seq (h d_k) -> ... h seq d_k",h=self.num_heads,d_k = self.d_k),token_positions)
        else:
            batched_Q = rearrange(Q,"... seq (h d_k) -> ... h seq d_k",h=self.num_heads,d_k = self.d_k)
            batched_K = rearrange(K ,"... seq (h d_k) -> ... h seq d_k",h=self.num_heads,d_k = self.d_k)
            
        batched_V = rearrange(V,"... seq (h d_k) -> ... h seq d_k",h=self.num_heads,d_k = self.d_k)
        
        
        batched_tri = torch.triu(torch.ones(seq,seq, dtype= torch.bool)).T
        batched_attention = scaled_dot_product_attention(batched_Q,batched_K,batched_V,batched_tri)
        MHA_attention = rearrange(batched_attention,"... h q v -> ... q (h v)")
        return einsum(self.O_weights,MHA_attention,"d_model d_p , ... seq d_p -> ... seq d_model")
        