import torch
import lib
import math
import typing as ty
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init


from torch import Tensor
from typing import Union, Tuple, List, Type, Optional
from math import ceil
from default_config import n_features, default_model_configs


def attenuated_kaiming_uniform_(tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='leaky_relu'):
    fan = nn_init._calculate_correct_fan(tensor, mode)
    gain = nn_init.calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)




class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        self.weight = nn.Parameter(Tensor(d_numerical+ len(categories), d_token))
        self.weight2 = nn.Parameter(Tensor(d_numerical+ len(categories), d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        self.bias2 = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        attenuated_kaiming_uniform_(self.weight)
        attenuated_kaiming_uniform_(self.weight2)
        nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        nn_init.kaiming_uniform_(self.bias2, a=math.sqrt(5))

    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = torch.cat((x_num, x_cat), dim=1)
        assert x_some is not None
        x1 = self.weight[None]  * x_some[:, :, None] + self.bias[None]
        x2 = self.weight2[None] * x_some[:, :, None] + self.bias2[None]

        return torch.tanh(x1) * x2



class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, init_scale: float = 0.01
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for i, m in enumerate([self.W_q, self.W_k, self.W_v]):
            attenuated_kaiming_uniform_(m.weight, scale=init_scale)
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            attenuated_kaiming_uniform_(self.W_out.weight)
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )
    
    def get_attention_mask(self, input_shape, device):
        bs, _, seq_len = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = seq_ids[None, None, :].repeat(bs, seq_len, 1) <= seq_ids[None, :, None]
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        return attention_mask

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_scores = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        masks = self.get_attention_mask(attention_scores.shape, attention_scores.device)
        attention = F.softmax(attention_scores + masks, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x



class F_ELU(nn.Module):

    def __init__(
        self,
        input_size: int,
        bias: bool = True,
        activation: Type[nn.Module] = nn.ELU,
        device: Union[str, torch.device] = "cuda",
    ):

        super().__init__()
        self.weight = nn.Parameter(torch.normal(mean=0, std=1.0, size=(1, input_size*n_features)))
        self.bias = nn.Parameter(torch.zeros(size=(1, input_size*n_features)), requires_grad=bias)
        self.activation = activation()
        self.to(device)

    def forward(self, X: Tensor) -> Tensor:

        out = X
        if len(X.shape) > 2:
            out = out.reshape((X.shape[0], -1))
        out = out * self.weight + self.bias
        if len(X.shape) > 2:
            out = out.reshape(X.shape)
        out = self.activation(out)
        out = out.view(-1, default_model_configs['d_token'])
        return out


class MLP(nn.Module):

    def __init__(
        self,
        #task: str = 'classification',
        task: str = 'regression',
        input_size: int = default_model_configs['d_token'],
        hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512),
        output_size: int = default_model_configs['d_token'],
        activation: Type[nn.Module] = nn.ELU,
        dropout: Union[float, Tuple[float], List[float]] = 0.0,
        dropout_first: bool = False,
        use_bn: bool = True,
        bn_momentum: float = 0.1,
        ghost_batch: Optional[int] = 1,
        f_elu: bool = True,
        use_skip: bool = True,
        weighted_sum: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):

        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        dropout_len = len(hidden_sizes) + (1 if dropout_first else 0)

        if isinstance(dropout, float):
            dropout = [dropout] * dropout_len
        elif not len(dropout) == dropout_len:
            raise ValueError(
                f"expected a single dropout value or {dropout_len} values "
                f"({'one more than' if dropout_first else 'same as'} hidden_sizes)"
            )

        main_layers: List[nn.Module] = []

        if f_elu:
            main_layers.append(F_ELU(input_size))

        if dropout_first and dropout[0] > 0:
            main_layers.append(nn.Dropout(dropout[0]))
            dropout = dropout[1:]

        input_size_i = input_size
        for hidden_size_i, dropout_i in zip(hidden_sizes, dropout):
            main_layers.append(nn.Linear(input_size_i, hidden_size_i, bias=(not use_bn)))
            if use_bn:
                if ghost_batch is None:
                    bnlayer = nn.BatchNorm1d(hidden_size_i, momentum=bn_momentum)
                else:
                    bnlayer = GhostBatchNorm(
                        hidden_size_i, ghost_batch, momentum=bn_momentum
                    )
                main_layers.append(bnlayer)
            main_layers.append(activation())
            if dropout_i > 0:
                main_layers.append(nn.Dropout(dropout_i))
            input_size_i = hidden_size_i

        main_layers.append(
            nn.Linear(input_size_i, output_size, bias=(task != "classification"))
        )

        self.main_layers = nn.Sequential(*main_layers)

        self.use_skip = use_skip
        if use_skip:
            skip_linear = nn.Linear(input_size, output_size, bias=(task != "classification"))
            if f_elu:
                self.skip_layers = nn.Sequential(F_ELU(input_size), skip_linear)
            else:
                self.skip_layers = skip_linear
            if weighted_sum:
                self.mix = nn.Parameter(torch.tensor([0.0]))
            else:
                self.mix = torch.tensor([0.0], device=device)
        else:
            self.skip_layers = None
            self.mix = None

        self.to(device)

    def weight_sum(self) -> Tuple[Tensor, Tensor]:

        w1_sum = 0.0
        w2_sum = 0.0
        for layer_group in (self.main_layers, self.skip_layers):
            if layer_group is None:
                continue
            for layer in layer_group:
                if not isinstance(layer, nn.Linear):
                    continue
                w1_sum += layer.weight.abs().sum()
                w2_sum += (layer.weight ** 2).sum()
        return w1_sum, w2_sum

    def forward(self, X: Tensor) -> Tuple[float, float]:

        out = self.main_layers(X)
        if self.use_skip:
            mix = torch.sigmoid(self.mix)
            skip_out = self.skip_layers(X)
            out = mix * skip_out + (1 - mix) * out
        return out


class GhostNorm(nn.Module):
    """
    Ghost Normalization
    https://arxiv.org/pdf/1705.08741.pdf

    """

    def __init__(
        self,
        inner_norm: nn.Module,
        virtual_batch_size: int,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Parameters
        ----------
        inner_norm : torch.nn.Module (initialiezd)
            examples: `nn.BatchNorm1d`, `nn.LayerNorm`
        virtual_batch_size : int
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()
        self.virtual_batch_size = 64
        self.inner_norm = inner_norm
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor

        Return
        ------
        torch.Tensor

        """
        chunk_size = int(ceil(x.shape[0] / self.virtual_batch_size))
        chunk_norm = [self.inner_norm(chunk) for chunk in x.chunk(chunk_size, dim=0)]
        return torch.cat(chunk_norm, dim=0)


class GhostBatchNorm(GhostNorm):
    """
    Ghost Normalization, using BatchNorm1d as inner normalization
    https://arxiv.org/pdf/1705.08741.pdf

    """
    def __init__(
        self,
        num_features: int,
        virtual_batch_size: int = 64,
        momentum: float = 0.1,
    ):
        super().__init__(
            inner_norm=nn.BatchNorm1d(num_features, momentum=momentum),
            virtual_batch_size=virtual_batch_size,
        )


class RDFPNET(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        prenormalization: bool,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
        init_scale: float = 0.1,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        n_tokens = d_numerical + len(categories) if categories is not None else d_numerical
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.n_categories = 0 if categories is None else len(categories)

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, init_scale=init_scale
                    ),
                    'linear0': nn.Linear(d_token, d_token * 2),
                    'norm1': make_normalization(),
                }
            )
            attenuated_kaiming_uniform_(layer['linear0'].weight, scale=init_scale)
            nn_init.zeros_(layer['linear0'].bias)

            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = lib.get_activation_fn('tanglu')
        self.last_activation = nn.PReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout


        self.head = nn.Linear(d_token, d_out)
        attenuated_kaiming_uniform_(self.head.weight)
        self.last_fc = nn.Linear(n_tokens, 1)
        attenuated_kaiming_uniform_(self.last_fc.weight)



    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
    
    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        assert x_cat is None
        x = self.tokenizer(x_num, x_cat)
        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = self._end_residual(x, x_residual, layer, 0)


            x_residual = self._start_residual(x, layer, 1)
            ffsm = MLP()
            x_residual = ffsm(x_residual)
            x_residual = x_residual.view(x.shape[0], x.shape[1], x_residual.shape[1])
            x = self._end_residual(x, x_residual, layer, 1)

        x = self.last_fc(x.transpose(1,2))[:,:,0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)

        return x
