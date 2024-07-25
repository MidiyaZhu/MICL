import warnings
from typing import Callable, Optional, List, Union,Tuple
from functools import wraps, partial
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import math
from icl.util_classes.predictor_classes import Predictor


class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def gpt2_attn(self, query, key, value, attention_mask=None, head_mask=None, attention_adapter=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights,
                                   self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel, predictor: Predictor, n_demo, device,n_class_tokens,n_head):
        self.n_demo = n_demo
        self.n_head = n_head
        self.n_class_tokens = n_class_tokens
        self.device = device
        self.model = model
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)
        self.predictor = predictor

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        if 'llama' in self.model.config._name_or_path:
            class_poss, final_poss = self.predictor.get_posllama({'input_ids': input_ids})
        else:
            class_poss, final_poss = self.predictor.get_posgpt({'input_ids': input_ids})
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)
            attention_adapter.class_poss = class_poss
            attention_adapter.final_poss = final_poss

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                attention_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad, use_abs=True):
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self, *args, **kwargs):
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(self.grad_process(attention_adapter.params.grad, *args, **kwargs))
        return grads

    def params(self):
        params = []
        for attention_adapter in self.attention_adapters:
            params.append(attention_adapter.weight)
        return params


def manager_decoractor(manager: AttentionerManagerBase):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator
class GPT2AttentionerManager_o(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, n_demo, predictor: Predictor, device, n_head=1):
        super().__init__(model, predictor, n_demo, device,n_head=n_head)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter = AttentionAdapter(n_demo=self.n_demo, device=self.device,
                                                 n_head=self.n_head)
            layer.attn._attn = partial(gpt2_attn, layer.attn,
                                       attention_adapter=attention_adapter)
            attention_adapters.append(attention_adapter)
        return attention_adapters

class GPT2AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, n_demo, predictor: Predictor, device,n_class_tokens, n_head=1):
        super().__init__(model, predictor, n_demo, device,n_class_tokens,n_head=n_head)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter = AttentionAdapter(n_demo=self.n_demo, device=self.device, n_class_tokens=self.n_class_tokens,
                                                 n_head=self.n_head)
            layer.attn._attn = partial(gpt2_attn, layer.attn,
                                       attention_adapter=attention_adapter)
            attention_adapters.append(attention_adapter)
        return attention_adapters


class AttentionAdapter(AttentionAdapterBase):
    def __init__(self, n_demo, n_head, n_class_tokens,device) -> None:
        super().__init__()
        self.n_demo = n_demo
        self.n_head = n_head
        # self.weight = torch.nn.Parameter(torch.zeros((n_head, n_demo), requires_grad=True, device=device))
        self.weight = torch.nn.Parameter(torch.zeros((n_head, n_class_tokens), requires_grad=True, device=device))

        self.class_poss = None
        self.final_poss = None
        # self.device=device


    def _forward(self, attn_weights):
        class_poss = self.class_poss
        # class_poss = [self.class_poss[0],self.class_poss[2]]
        final_poss = self.final_poss
        weight = self.weight.exp()
        bsz, n_head, seq_len, _ = attn_weights.shape
        assert bsz == 1
        mask_mat = torch.ones((1, n_head, seq_len, seq_len), device=attn_weights.device)
        # mask_mat[:, :, final_poss, class_poss] = weight.reshape(1, self.n_head, self.n_demo)
        mask_mat[:, :, final_poss, class_poss] = weight.reshape(1, self.n_head, len(self.class_poss))
        # print(f'weight\n{weight.reshape}')
        return attn_weights * mask_mat

    @property
    def grad(self):
        return self.weight.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.weight.grad is not None:
            if set_to_none:
                self.weight.grad = None
            else:
                self.weight.grad = torch.zeros_like(self.weight.grad)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def llama2_self_attn( self, hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            padding_mask: Optional[torch.LongTensor] = None,attention_adapter=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if attention_adapter is not None:
            attn_weights = attention_adapter(attn_weights)

        attn_weights = attn_weights.to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class Llama2AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, n_demo, predictor: Predictor, device,n_class_tokens, n_head=1):
        super().__init__(model, predictor, n_demo, device,n_class_tokens,n_head=n_head)
        # super().__init__(model)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.model.layers):
            attention_adapter = AttentionAdapter(n_demo=self.n_demo, device=self.device,n_class_tokens=self.n_class_tokens,
                                                 n_head=self.n_head)
            layer.self_attn.forward = partial(llama2_self_attn, layer.self_attn,
                                       attention_adapter=attention_adapter)
            attention_adapters.append(attention_adapter)
        return attention_adapters
