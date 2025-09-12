"""
Definition of decoders classes.
Adapted from original code at https://github.com/BenSaunders27/ProgressiveTransformersSLP
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.helpers import freeze_params, subsequent_mask
from .transformer_layers import PositionalEncoding, TransformerDecoderLayer


class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size

class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(
            self,
            num_layers: int = 4,
            num_heads: int = 8,
            hidden_size: int = 512,
            ff_size: int = 2048,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            freeze: bool = False,
            trg_size: int = 97,
            decoder_trg_trg_: bool = True,
            apply_root_quat_treatment: bool = False,
    ):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param freeze: set to True keep all decoder parameters fixed
        :param apply_root_quat_treatment: if True, will apply:
                                            - nothing to the 1st 3 values of the outputs
                                            - tanh + L2 normalization to [3:-1] values (quaternions)
                                            - sigmoid to the counter values
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # Dynamic output size depending on the target size
        self._output_size = trg_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                decoder_trg_trg=decoder_trg_trg_
            ) for _ in range(num_layers)]
        )

        self.pe = PositionalEncoding(hidden_size,mask_count=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        # OR root joint vector + 4 * N_bones + 1 (in the case we use quaternions)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        self.apply_root_quat_treatment = apply_root_quat_treatment

        if freeze:
            freeze_params(self)

    def forward(
            self,
            trg_embed: Tensor = None,
            encoder_output: Tensor = None,
            src_mask: Tensor = None,
            trg_mask: Tensor = None,
    ):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param trg_mask: to mask out target paddings Note that a subsequent mask is applied here.
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        # add position encoding to word embedding
        x = self.pe(trg_embed)
        # Dropout if given
        x = self.emb_dropout(x)

        padding_mask = trg_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        # Apply each layer to the input
        decoder_self_attention_outputs = torch.zeros(
            len(self.layers), x.shape[0], x.shape[1], self._hidden_size, device=x.device
        )

        for k, layer in enumerate(self.layers):
            x, h1 = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)
            decoder_self_attention_outputs[k] = h1

        # Apply a layer normalisation
        x = self.layer_norm(x)
        # Output layer turns it back into vectors of size trg_size
        output = self.output_layer(x)

        if self.apply_root_quat_treatment:
            # `output` is of shape (N_batch, T, 3 + 4 * N_bones + 1)
            # with output[n, t] = [xRoot, yRoot, zRoot, q1_1, q2_1, q3_1, q4_1, ..., q1_Nbones, q2_Nbones, q3_Nbones, q4_Nbones, counterValue]
            output_new = output.clone()

            # 1) We apply tanh to the quaternions to enforce the values to be in [-1, 1]
            output_new[:, :, 3:-1] = torch.tanh(output[:, :, 3:-1])

            # 2) We divide each quaternion Q=[q1, q2, q3, q4] by its L2 norm to ensure |Q_new| = 1
            N, T = output.shape[:2]
            quaternions = output[:, :, 3:-1].view(N, T, -1, 4)
            norms = torch.norm(quaternions, p=2, dim=-1, keepdim=True)  # L2 norms over last dimension => shape=(N_batch, T, N_bones, 1)
            quaternions = quaternions / (norms + 1e-8)  # normalize avoiding division by zero
            output_new[:, :, 3:-1] = quaternions.view(N, T, -1)

            # 3) We apply sigmoid to the counter to enforce the values to be in [0, 1]
            output_new[:, :, -1:] = torch.sigmoid(output[:, :, -1:])

            return output_new, x, decoder_self_attention_outputs, None  # `decoder_self_attention_outputs` for contrastive losses computations
        else:
            return output, x, decoder_self_attention_outputs, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers), self.layers[0].trg_trg_att.num_heads
        )
