"""
Function for autoregressive computation of Transformer decoder outputs from the encoder output.
Used for predictions (once the model is already trained).

More precisely:
    1) the 1st frame of the target pose sequence and the Transformer encoder output are given
       as inputs of the Transformer decoder to produce a 1st output sequence of shape (B, 1, D)d

    2) the last frame of the output sequence is taken and added to the 1st frame, and the
       resulting sequence is given as input along with the encoder output to the Transformer decoder to
       produce a 2nd output sequence of shape (B, 2, D)

    3) the last frame of this output sequence is taken and added to the 2 frames given as inputs in step 2),
       and again passed as inputs to the decoder to produce a 3rd output sequence of shape (B, 3, D)

    4) etc... this is repeated once reaching T and having a sequence of shape (B, T, D) made of all last frames
       predictions of previous decoding steps

Original code from https://github.com/BenSaunders27/ProgressiveTransformersSLP
"""


import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from .decoders import Decoder
from .embeddings import Embeddings


def greedy(
        src_mask: Tensor,
        embed: Embeddings,
        decoder: Decoder,
        encoder_output: Tensor,
        trg_input: Tensor,
        model,
) -> (np.array, np.array):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # Initialise the input
    # Extract just the BOS first frame from the target
    ys = trg_input[:,:1,:].float()

    # If the counter is coming into the decoder or not
    ys_out = ys

    # Set the target mask, by finding the padded rows
    trg_mask = trg_input != 0.0
    trg_mask = trg_mask.unsqueeze(1)

    # Find the maximum output length for this batch
    max_output_length = trg_input.shape[1]

    # If just count in, input is just the counter
    if model.just_count_in:
        ys = ys[:,:,-1:]

    for i in range(max_output_length):
        # ys here is the input
        # Drive the timing by giving the GT timing - add in the counter to the last column

        if model.just_count_in:
            # If just counter, drive the input using the GT counter
            ys[:,-1] = trg_input[:, i, -1:]

        else:
            # Give the GT counter for timing, to drive the timing
            ys[:,-1,-1:] = trg_input[:, i, -1:]

        # Embed the target input before passing to the decoder
        trg_embed = embed(ys)

        # Cut padding mask to required size (of the size of the input)
        padding_mask = trg_mask[:, :, :i+1, :i+1]
        # Pad the mask (If required) (To make it square, and used later on correctly)
        pad_amount = padding_mask.shape[2] - padding_mask.shape[3]
        padding_mask = (F.pad(input=padding_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)

        # Pass the embedded input and the encoder output into the decoder
        with torch.no_grad():
            out, _, _, _ = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                src_mask=src_mask,
                trg_mask=padding_mask,
            )

            if model.future_prediction != 0:
                # Cut to only the first frame prediction
                out = torch.cat((out[:, :, :out.shape[2] // (model.future_prediction)],out[:,:,-1:]),dim=2)

            if model.just_count_in:
                # If just counter in trg_input, concatenate counters of output
                ys = torch.cat([ys, out[:,-1:,-1:]], dim=1)

            # Add this frame prediction to the overall prediction
            ys = torch.cat([ys, out[:,-1:,:]], dim=1)

            # Add this next predicted frame to the full frame output
            ys_out = torch.cat([ys_out, out[:,-1:,:]], dim=1)

    return ys_out, None
