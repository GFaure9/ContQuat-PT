# coding: utf-8
"""
Module to represents whole models
"""
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from .initialization import initialize_model
from .embeddings import Embeddings
from .encoders import Encoder, TransformerEncoder
from .decoders import Decoder, TransformerDecoder
from ..data.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from .search import greedy
from ..data.vocabulary import Vocabulary
from ..data.batch import Batch
from ..losses.loss import LossWithSupCont, LossWithSBERTCont
from ..architecture.prepro_layers_for_LsBERTCon import SBERTLossPreproLayers

from utils.skeleletal_structures_helper import ORIGINAL_S2SL_SKEL, ORIGINAL_S2SL_SKEL_INVERTED_HANDS, generate_t_pose
from utils.skeletal_representations import cartesian_to_quaternion_pose, quaternion_to_cartesian_pose


class Model(nn.Module):
    """
    Base Model class
    """

    # Added by me (GF)
    # skel_name = "original_s2sl_skel"
    # skel_structure = ORIGINAL_S2SL_SKEL
    # skel_name = "original_s2sl_skel_inverted_hands"
    # skel_structure = ORIGINAL_S2SL_SKEL_INVERTED_HANDS

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int,
                 sent_emb_size: int = None,  # added by me (GF)
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.trg_embed = trg_embed

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in", True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in", False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise", False)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)

        # added by me (GF)
        # self.use_quaternions = model_cfg.get("quaternions", False)
        self.trg_is_quat = (cfg["data"]["trg"] == "quat")
        self.cfg = cfg.copy()

        # added by me (GF)
        self.prepro_layers_for_LsBERTCon = None
        if "sbert_contrastive_loss" in cfg["training"].keys():
            if cfg["training"]["sbert_contrastive_loss"]:
                if "sbert_cont_params" in cfg["model"].keys():
                    if cfg["model"]["sbert_cont_params"] != {}:
                        self.prepro_layers_for_LsBERTCon = SBERTLossPreproLayers(
                            input_dim=decoder._hidden_size,
                            output_dim=sent_emb_size,
                            **cfg["model"]["sbert_cont_params"],
                        )

    # pylint: disable=arguments-differ
    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                src_mask: Tensor,
                src_lengths: Tensor,
                trg_mask: Tensor = None,
                src_input: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)

        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:, :, -1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds, torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate * noise

        # Decode the target sequence
        skel_out, dec_hidden, decoder_self_attention_outputs, _ = self.decode(
            encoder_output=encoder_output, src_mask=src_mask, trg_input=trg_input, trg_mask=trg_mask
        )  # `decoder_self_attention_outputs` added by me (GF)

        gloss_out = None

        return skel_out, gloss_out, decoder_self_attention_outputs  # `decoder_self_attention_outputs` added by me (GF)

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)

        return encode_output

    def decode(self, encoder_output: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):

        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # ======= added by me (GF)
        # N, T = trg_input.shape[:-1]
        # x, counter = trg_input[:, :, :-1], trg_input[:, :, -1:]  # copy removing counter + copy counter
        # x_pts = x.reshape(N, T, -1, 3)  # reshape to (N_batch, T, num_pts, 3)
        #
        # if self.use_quaternions:
        #     # Go from shape=(N_batch, T, 3 * num_pts + 1[counter]) to shape=(N_batch, T, 3[root] + num_bones * 4[quaternions] + 1[counter])
        #     # here `num_pts` should be 50 and `num_bones` should be 49
        #     # so we go from (N_batch, T, 151) to (N_batch, T, 200)
        #     trg_input_q = torch.zeros(N, T, 3 + len(self.skel_structure) * 4 + 1)  # initialize tensor of root + quaternions + counter
        #     for n in range(N):
        #         first_skel = x_pts[n, 0, :, :].cpu().numpy()
        #         bones_lengths = [np.linalg.norm(first_skel[a] - first_skel[b]) for (a, b, _) in self.skel_structure]
        #         for t in range(T):
        #             root_pt = x_pts[n, t, 0, :]
        #             sequence_t_pose = generate_t_pose(
        #                 skel_name=self.skel_name,
        #                 bones_lengths=bones_lengths,
        #                 root_pt=root_pt.cpu().numpy()
        #             )
        #             skel_q = torch.from_numpy(cartesian_to_quaternion_pose(
        #                 skel_pose=x_pts[n, t].cpu().numpy(),
        #                 skel_resting_pose=sequence_t_pose,
        #                 skel_structure=self.skel_structure
        #             )[0].flatten()).to(device="cuda")
        #             trg_input_q[n, t] = torch.cat((root_pt, skel_q, counter[n, t]))
        #     trg_input_q = trg_input_q.to(device="cuda")
        #     # Embed the target using a linear layer
        #     trg_embed = self.trg_embed(trg_input_q)
        # ========================
        # else:
        #     # Embed the target using a linear layer
        #     trg_embed = self.trg_embed(trg_input)  # original

        trg_embed = self.trg_embed(trg_input)
        # Apply decoder to the embedded target
        decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                                      src_mask=src_mask, trg_mask=trg_mask)
        # ======= added by me (GF)
        # if self.use_quaternions:
        #     # Go from shape=(N_batch, T, 3[root] + num_bones * 4[quaternions] + 1[counter]) to shape=(N_batch, T, 3 * num_pts + 1[counter])
        #     # here `num_pts` should be 50 and `num_bones` should be 49
        #     # so we go from (N_batch, T, 200) to (N_batch, T, 151)
        #     root_pts, quaternions, counter = decoder_output[0][:, :, :3], decoder_output[0][:, :, 3:-1], decoder_output[0][:, :, -1:]
        #     output_cartesian = torch.zeros(N, T, trg_input.shape[-1])
        #     for n in range(N):
        #         first_skel = x_pts[n, 0, :, :].cpu().numpy()
        #         bones_lengths = [np.linalg.norm(first_skel[a] - first_skel[b]) for (a, b, _) in self.skel_structure]
        #         for t in range(T):
        #             root_pt = root_pts[n, t].detach()
        #             sequence_t_pose = generate_t_pose(
        #                 skel_name=self.skel_name,
        #                 bones_lengths=bones_lengths,
        #                 root_pt=root_pt.cpu().numpy()
        #             )
        #             skel_cartesian = torch.from_numpy(quaternion_to_cartesian_pose(
        #                 root_pt=root_pt.cpu().numpy(),
        #                 skel_quaternions=quaternions[n, t].detach().cpu().numpy().reshape(-1, 4),
        #                 skel_resting_pose=sequence_t_pose,
        #                 bones_lengths=np.array(bones_lengths),
        #                 skel_structure=self.skel_structure,
        #             ).flatten()).to(device="cuda")
        #             output_cartesian[n, t] = torch.cat((skel_cartesian, counter[n, t]))
        #     # Update decoder output
        #     return output_cartesian.clone().to(device="cuda"), decoder_output[1], decoder_output[2], decoder_output[3]
        # ========================
        # else:
        #     return decoder_output

        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input

        # todo(gf): [remark for contrastive loss]
        #           The `self.forward(...)` should also return the following tensors X and Y:
        #               - Y: (N_num_transformer_decoder_layers, N_batch, dim_transformer_decoder_self_attention)
        #               - X: (N_batch, dim_output_transformer_encoder)
        #           => Question: with dim_dec_self_att=(Ty, Dy) and dim_enc_out=(Tx, Dx),
        #                        we know that Dy=Dx (normally?) but do we have Tx=Ty (I think not...).
        #                        --> how to deal with Tx != Ty ???

        skel_out, _, decoder_self_attention_outputs = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask)  # `decoder_self_attention_outputs` added by me (GF) for contrastive losses computations

        # compute batch loss using skel_out and the batch target

        # added by me (GF)
        # ------------------------------
        sup_cont_loss_kwargs = {}
        if isinstance(loss_function, LossWithSupCont):
            sup_cont_loss_kwargs = {
                "outputs_decoder_self_att_layers": decoder_self_attention_outputs,
                # shape=(num_self_att_layer, N_batch, T, dim_output_transformer_decoder_self_att)
                "glosses_ids_batch": batch.src,  # shape=(N_batch, max_gloss_sequence_length)
                "batch_normalization_compensation_factor": batch.nseqs if loss_function.compensate_batch_normalization else 1.,
            }

        sbert_cont_loss_kwargs = {}
        if isinstance(loss_function, LossWithSBERTCont):
            sbert_cont_inputs = decoder_self_attention_outputs  # shape=(num_self_att_layer, N_batch, T, dim_output_transformer_decoder_self_att)
            if self.prepro_layers_for_LsBERTCon is not None:
                assert self.prepro_layers_for_LsBERTCon.linear.out_features == batch.sent_emb.shape[
                    -1]  # checking that the output size of the FFN equals the dimension of sentences embeddings
                sbert_cont_inputs = torch.stack([self.prepro_layers_for_LsBERTCon(tens) for tens in sbert_cont_inputs],
                                                dim=0)  # shape=(num_self_att_layer, N_batch, dim_sentence_embedding)

            sbert_cont_loss_kwargs = {
                "outputs_decoder_self_att_layers": sbert_cont_inputs,
                "sbert_embeddings_batch": batch.sent_emb,  # shape=(N_batch, dim_sentence_embedding)
                "batch_normalization_compensation_factor": batch.nseqs if loss_function.compensate_batch_normalization else 1.,
            }
        # ------------------------------

        batch_loss = loss_function(skel_out, batch.trg, **sup_cont_loss_kwargs, **sbert_cont_loss_kwargs)

        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss, noise

    def run_batch(self, batch: Batch, max_output_length: int, ) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # ============ added by me (GF)
        # if self.use_quaternions is True
        # --> transforming the `batch.trg_input` in the correct transformer decoder input format
        # trg_input = batch.trg_input
        # N, T = trg_input.shape[:-1]
        # x, counter = trg_input[:, :, :-1], trg_input[:, :, -1:]  # copy removing counter + copy counter
        # x_pts = x.reshape(N, T, -1, 3)  # reshape to (N_batch, T, num_pts, 3)
        #
        # if self.use_quaternions:
        #     if not self.just_count_in:
        #         # Go from shape=(N_batch, T, 3 * num_pts + 1[counter]) to shape=(N_batch, T, 3[root] + num_bones * 4[quaternions] + 1[counter])
        #         # here `num_pts` should be 50 and `num_bones` should be 49
        #         # so we go from (N_batch, T, 151) to (N_batch, T, 200)
        #         trg_input_q = torch.zeros(N, T, 3 + len(
        #             self.skel_structure) * 4 + 1)  # initialize tensor of root + quaternions + counter
        #         for n in range(N):
        #             first_skel = x_pts[n, 0, :, :].cpu().numpy()
        #             bones_lengths = [np.linalg.norm(first_skel[a] - first_skel[b]) for (a, b, _) in
        #                              self.skel_structure]
        #             for t in range(T):
        #                 root_pt = x_pts[n, t, 0, :]
        #                 sequence_t_pose = generate_t_pose(
        #                     skel_name=self.skel_name,
        #                     bones_lengths=bones_lengths,
        #                     root_pt=root_pt.cpu().numpy()
        #                 )
        #                 skel_q = torch.from_numpy(cartesian_to_quaternion_pose(
        #                     skel_pose=x_pts[n, t].cpu().numpy(),
        #                     skel_resting_pose=sequence_t_pose,
        #                     skel_structure=self.skel_structure
        #                 )[0].flatten()).to(device="cuda")
        #                 trg_input_q[n, t] = torch.cat((root_pt, skel_q, counter[n, t]))
        #         trg_input_q = trg_input_q.to(device="cuda")
        #         trg_input = trg_input_q.clone()
        #     else:
        #         raise ValueError(
        #             "Using quaternion representation when `just_count_in==True` has not been implemented yet!")
        # ===============================

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding
        stacked_output, stacked_attention_scores = greedy(
            encoder_output=encoder_output,
            src_mask=batch.src_mask,
            embed=self.trg_embed,
            decoder=self.decoder,
            trg_input=batch.trg_input,  # original: `batch.trg_input`
            # trg_input=trg_input,  # original: `batch.trg_input`
            model=self)

        # ======= added by me (GF)
        # if self.use_quaternions:
        #     # Go from shape=(N_batch, T, 3[root] + num_bones * 4[quaternions] + 1[counter]) to shape=(N_batch, T, 3 * num_pts + 1[counter])
        #     # here `num_pts` should be 50 and `num_bones` should be 49
        #     # so we go from (N_batch, T, 200) to (N_batch, T, 151)
        #     root_pts, quaternions, counter = stacked_output[:, :, :3], stacked_output[:, :, 3:-1], stacked_output[:, :,
        #                                                                                            -1:]
        #     output_cartesian = torch.zeros(N, T, batch.trg_input.shape[-1])
        #     for n in range(N):
        #         first_skel = x_pts[n, 0, :, :].cpu().numpy()
        #         bones_lengths = [np.linalg.norm(first_skel[a] - first_skel[b]) for (a, b, _) in self.skel_structure]
        #         for t in range(T):
        #             root_pt = root_pts[n, t].detach()
        #             sequence_t_pose = generate_t_pose(
        #                 skel_name=self.skel_name,
        #                 bones_lengths=bones_lengths,
        #                 root_pt=root_pt.cpu().numpy()
        #             )
        #             skel_cartesian = torch.from_numpy(quaternion_to_cartesian_pose(
        #                 root_pt=root_pt.cpu().numpy(),
        #                 skel_quaternions=quaternions[n, t].detach().cpu().numpy().reshape(-1, 4),
        #                 skel_resting_pose=sequence_t_pose,
        #                 bones_lengths=np.array(bones_lengths),
        #                 skel_structure=self.skel_structure,
        #             ).flatten()).to(device="cuda")
        #             output_cartesian[n, t] = torch.cat((skel_cartesian, counter[n, t]))
        #     # Update output
        #     stacked_output = output_cartesian.clone().to(device="cuda")
        # ========================

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                sent_emb_size: int = None,  # added by me (GF)
                ) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :param sent_emb_size: size of the sentences embeddings (only if using L_sBERTCont with prepro)
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1) * future_prediction + 1

    # Define source embedding
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    # ======== options to use quaternion representation added by me (GF)
    # skel_structure = Model.skel_structure
    # if cfg["quaternions"]:
    #     in_trg_size_q = 3 + 4 * len(skel_structure) + 1  # 3[root] + 4 * N_bones + 1[counter]
    #     trg_linear = nn.Linear(in_trg_size_q, cfg["decoder"]["embeddings"]["embedding_dim"])
    # # ==============================================
    # else:
    #     trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.)  # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
        "for transformer, emb_size must be hidden_size"

    # Transformer Encoder
    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_size=src_embed.embedding_dim,
                                 emb_dropout=enc_emb_dropout)

    ## Decoder -------
    dec_dropout = cfg["decoder"].get("dropout", 0.)  # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    # Transformer Decoder
    # ===== added by me (GF)
    # trg_size_q = 3 + 4 * len(skel_structure) + 1
    # transformer_decoder_out_size = trg_size_q if cfg["quaternions"] else out_trg_size  # added by me (GF)
    # =====================
    transformer_decoder_out_size = out_trg_size
    root_quat_postpro = (full_cfg["data"]["trg"] == "quat")  # whether to ensure unit quaternions and counter in [0, 1]
    decoder = TransformerDecoder(
        **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
        emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
        trg_size=transformer_decoder_out_size, decoder_trg_trg_=decoder_trg_trg,
        apply_root_quat_treatment=root_quat_postpro,
        # use_quaternions=cfg["quaternions"]  # added by me (GF)
    )

    # Define the model
    model = Model(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        trg_embed=trg_linear,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        cfg=full_cfg,
        in_trg_size=in_trg_size,
        out_trg_size=out_trg_size,
        sent_emb_size=sent_emb_size  # added by me (GF) - for when L_sBERTCont is computed with prepro layers before
    )

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model