"""
Implementation of different training losses:
- RegLoss: L_SLP := MSE or L1 loss
- RootQuatLoss: L_SLP := L_Geodesic + MSE_root_joint
- LossWithSBERTCont: L_tot := L_SLP + sca * L_SBERTSupCont
- LossWithSupCont: L_tot := L_SLP + sca * L_GlossSupCont
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from torch import vmap


class LossWithSBERTCont(nn.Module):
    def __init__(
            self,
            cfg,
            target_loss: nn.Module,
            compensate_batch_normalization: bool = True,
    ):
        super(LossWithSBERTCont, self).__init__()

        self.criterion_target = target_loss

        model_cfg = cfg["model"]

        sbert_con_loss_params = model_cfg.get("sbert_con_params", {})
        self.criterion_sbert_cont = SBERTContDecoderSA(**sbert_con_loss_params)

        self.sbert_cont_loss_scale = model_cfg.get("sbert_cont_loss_scale", 1.0)
        self.compensate_batch_normalization = compensate_batch_normalization

    def forward(
            self,
            preds: Tensor,
            targets: Tensor,
            outputs_decoder_self_att_layers: Tensor,
            sbert_embeddings_batch: Tensor,
            batch_normalization_compensation_factor: 1.,
    ):
        trg_loss = self.criterion_target(preds, targets)
        sbert_cont_loss = self.criterion_sbert_cont(
            outputs_decoder_self_att_layers=outputs_decoder_self_att_layers,
            sbert_embeddings_batch=sbert_embeddings_batch,
        )

        # print(f"Target Loss: {trg_loss / batch_normalization_compensation_factor} "
        #       f"| Contrastive Loss: {self.sbert_cont_loss_scale * sbert_cont_loss}")

        return trg_loss + batch_normalization_compensation_factor * self.sbert_cont_loss_scale * sbert_cont_loss


class SBERTContDecoderSA(nn.Module):
    def __init__(self, **sbert_con_loss_params):
        super(SBERTContDecoderSA, self).__init__()

        self.criterion = SBERTSimilarityContrastiveLoss(**sbert_con_loss_params)

    def forward(self, outputs_decoder_self_att_layers: Tensor, sbert_embeddings_batch):
        # `outputs_decoder_self_att_layers`: shape=(N_self_att_layers, M_batch, dim_output_self_attention)
        # `sbert_embeddings_batch`: shape=(M_batch, max_gloss_sequence_length) -> line k is the SBERT embedding of the k-th sentence of the batch

        s = 0.
        for outputs in outputs_decoder_self_att_layers:
            s += self.criterion(outputs, sbert_embeddings_batch)

        return s / len(outputs_decoder_self_att_layers)  # averaging over number of layers


class SBERTSimilarityContrastiveLoss(nn.Module):
    """
    SBERT-based Contrastive loss (computed on a batch) defined as:

            L_sBERTCon := || ( CosSim(zi, zj) )_{(i,j)} - ( CosSim(sBERTi, sBERTj) )_{(i,j)} ||^2 / NumberOfPairs

    Where:
        - {(i,j)} are all possible pairs of batch IDs without order (i.e (i, j) and (j, i) will be viewed as the same)
        - z{k} is the (pose) embedding of the sequence n°{k}
        - sBERT{k} is the sentence transformers embedding of the sentence associated to k-th pose sequence
    and

        CosSim(a, b) := < a , b > / |a| |b|  (cosine similarity)
    """

    def __init__(self, **kwargs):
        super(SBERTSimilarityContrastiveLoss, self).__init__()

    def forward(self, z_batch: Tensor, sbert_embeddings_batch: Tensor):
        # ----- Reminder -----
        # `z_batch` shape = (M_batch, T, dim_hidden_layer) OR (M_batch, dim_sbert_embedding) if prepro (pooling+FFN) was performed
        # `sbert_embeddings_batch` shape = (M_batch, dim_sbert_embedding)
        # --------------------

        # 1/ Compute cosine similarities vectors (shape K = number of unique pairs (i, j) w/ i < j)
        z_batch_unique_cos_sims = pairwise_cosine_similarity(z_batch)  # shape=(K,)
        sbert_batch_unique_cos_sims = pairwise_cosine_similarity(sbert_embeddings_batch)  # shape=(K,)

        # 2/ Compute squared l2-norm distance between vectors
        normalization_factor = 1 / z_batch_unique_cos_sims.shape[0]
        dist = normalization_factor * torch.norm(z_batch_unique_cos_sims - sbert_batch_unique_cos_sims, p=2) ** 2

        return dist


def pairwise_cosine_similarity(x: Tensor) -> Tensor:
    """
    Compute cosine similarity between all unique pairs (i < j) of flattened vectors from input tensor x.
    I.e. will return:
            V = (< xi , xj > / |xi| |xj|)_{(i,j) \\in UniquePairs}

    Args:
        x (Tensor): Input tensor of shape (M, a, b, c, ...)

    Returns:
        Tensor: 1D tensor of shape (M * (M - 1) // 2,) containing cosine similarities
    """
    M = x.size(0)
    x_flat = x.view(M, -1)  # shape (M, a*b*c*...)
    x_norm = F.normalize(x_flat, dim=1)  # normalization -> each xi divided by || xi || (l2 norm)

    # --- pairwise dot products -> cosine similarities matrix
    cos_sim_matrix = x_norm @ x_norm.T  # shape (M, M) | symmetric, with diagonal of ones

    # --- extract upper triangular (excluding diagonal), i.e., all unique (i, j) with i < j
    i, j = torch.triu_indices(M, M, offset=1, device=x.device)
    cos_sims = cos_sim_matrix[i, j]

    return cos_sims  # shape = (M * (M - 1) // 2,)


class LossWithSupCont(nn.Module):
    def __init__(
            self,
            cfg,
            target_loss: nn.Module,
            compensate_batch_normalization: bool = True,
    ):
        super(LossWithSupCont, self).__init__()

        self.criterion_target = target_loss

        model_cfg = cfg["model"]

        sup_cont_loss_params = model_cfg.get("sup_cont_params", {'tau': 1.0})
        self.criterion_sup_cont = SupContDecoderSA(**sup_cont_loss_params)

        self.sup_cont_loss_scale = model_cfg.get("sup_cont_loss_scale", 1.0)
        self.compensate_batch_normalization = compensate_batch_normalization

    def forward(
            self,
            preds: Tensor,
            targets: Tensor,
            outputs_decoder_self_att_layers: Tensor,
            glosses_ids_batch: Tensor,
            batch_normalization_compensation_factor: 1.,
    ):
        trg_loss = self.criterion_target(preds, targets)
        sup_cont_loss = self.criterion_sup_cont(
            outputs_decoder_self_att_layers=outputs_decoder_self_att_layers,
            glosses_ids_batch=glosses_ids_batch,
        )

        # print(f"Target Loss: {trg_loss / batch_normalization_compensation_factor} "
        #       f"| Contrastive Loss: {self.sup_cont_loss_scale * sup_cont_loss}")

        return trg_loss + batch_normalization_compensation_factor * self.sup_cont_loss_scale * sup_cont_loss


class SupContDecoderSA(nn.Module):
    def __init__(self, tau: float = 1):
        super(SupContDecoderSA, self).__init__()

        self.criterion = SupervisedContrastiveLoss(tau=tau)

    def forward(self, outputs_decoder_self_att_layers: Tensor, glosses_ids_batch):
        # `outputs_decoder_self_att_layers`: shape=(N_self_att_layers, M_batch, dim_output_self_attention)
        # `glosses_ids_batch`: shape=(M_batch, max_gloss_sequence_length) each line is a sequence of vocab glosses IDs

        device = outputs_decoder_self_att_layers.device
        random_gen = torch.Generator(device=device).manual_seed(42)

        # --- with vmap
        # sup_cont_losses = torch.vmap(self.criterion, in_dims=(0, None, None), randomness="same")(
        #     outputs_decoder_self_att_layers, glosses_ids_batch, random_gen,
        # )  # shape=(N_self_att_layers,)
        # # N.B: param `in_dims=(0, None, None)` is to compute the function over args of dim 0 but keep dim 1 arg constant
        # # i.e. over `outputs_decoder_self_att_layers` and not `glosses_ids_batch` nor `random_generator`
        #
        # return sup_cont_losses.mean()

        # --- sequential (without vmap)
        s = 0.
        for outputs in outputs_decoder_self_att_layers:
            s += self.criterion(outputs, glosses_ids_batch, random_gen)

        return s / len(outputs_decoder_self_att_layers)  # averaging over number of layers


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive loss (computed on a batch) defined as:

            L_supCont :=
            sum_{i=0^I}(
                log( 1/|A(i)| sum_{a in A(i)}( exp(zi za / tau) / sum_{b in B(i)}( exp(zi zb / tau) ) ) )
            )

    Where:
        - i=0, 1, ..., I are the indices of anchor words of gloss/text inputs corresponding to sequences embeddings of the batch
        - A(i) are the indices of the sequences in the batch that contain the anchor word n°{i}
        - B(i) are the indices of the sequences in the batch that DO NOT contain the anchor word n°{i}
        - z{k} is the (pose) embedding of the sequence n°{k}
        - tau is a scalar temperature parameter

    (same as in "A Data-Driven Representation for Sign Language Production" Walsh et al. (2024)
    https://doi.org/10.48550/arXiv.2404.11499
    /!\ but with the logarithm of the original version of "Supervised Contrastive Learning" Khosla et al.(2020)
    https://doi.org/10.48550/arXiv.2004.11362)

    Objectives:
    overcome natural variation between signers or for the same signer /
    remove unwanted features not related to semantics (i.e. less 'noise' in embeddings)
    """

    def __init__(self, tau: float = 1.0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, z_batch: Tensor, glosses_ids_batch: Tensor, random_generator=None):
        device = z_batch.device

        # 1/ retrieve the set of glosses IDs (from all sequences of the batch)
        #    (no duplicates)
        gloss_ids = torch.unique(glosses_ids_batch)

        # 2/ iterate over glosses IDs and for each ID {i}:
        #       (a) find the sequence zi where {i} appears the most (if multiple candidates, randomly select one)
        #       (b) define Ai as the list of sequences containing {i} - without zi
        #       (c) define Bi as the list of sequences not containing {i}
        #       (d) compute the i-th term of the L_supCont sum and increment the sum
        s = 0.
        for i in gloss_ids:

            with torch.no_grad():
                # --- step (a)
                counts = (glosses_ids_batch == i).sum(dim=1)  # occurrences of id {i} in each row | shape=(M_batch,)
                max_count = counts.max()  # max count
                row_indices = (counts == max_count).nonzero(as_tuple=True)[0]  # row indices with that max count
                g = random_generator if random_generator is not None else torch.Generator(device=device).manual_seed(
                    42)  # for 'local' randomness
                random_idx = row_indices[torch.randint(len(row_indices), (1,), generator=g,
                                                       device=device)].item()  # random row index among those with most {i}
                zi = z_batch[random_idx].clone()

                # --- step (b)
                mask = (counts > 0)  # rows containing {i}
                mask[random_idx] = False  # exclude zi | shape=(M_batch,)
                Ai = z_batch[mask]

                # --- step (c)
                mask[random_idx] = True  # put back zi in mask to exclude it with ~mask
                Bi = z_batch[~mask]

                # L2 normalization
                zi = F.normalize(zi, dim=-1)
                Ai = F.normalize(Ai, dim=-1)
                Bi = F.normalize(Bi, dim=-1)

            # --- step (d)
            if len(Ai) == 0 or len(Bi) == 0:
                continue  # skipping the ID {i}

            # uncomment to only compute the per-anchor sub-loss if 20% of batch sequences contain the anchor gloss
            # if len(Ai) < 0.2 * len(z_batch):
            #     continue

            zi_dot_Ai = torch.sum(zi.flatten() * Ai.flatten(1) / self.tau, dim=1)  # shape = (len(Ai),)
            zi_dot_Bi = torch.sum(zi.flatten() * Bi.flatten(1) / self.tau, dim=1)  # shape = (len(Bi),)
            s += (torch.logsumexp(zi_dot_Ai, dim=0) - torch.logsumexp(zi_dot_Bi, dim=0)
                  - torch.log(torch.tensor(len(Ai), dtype=torch.float32)))

        return -s  # /!\ the `-` (minus) is very important since we want that min(L_supCont) maximizes the zi . za


class RootQuatLoss(nn.Module):
    """
    Loss for (root joint 3D, quaternions) outputs.
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RootQuatLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "mse_mse":
            self.criterion_root = nn.MSELoss()
            self.criterion_quaternions = nn.MSELoss()
        if self.loss == "mse_mgd":
            self.criterion_root = nn.MSELoss()
            self.criterion_quaternions = MeanGeodesicDistance()
        else:
            print("No valid loss found in configuration file. Taking 'mse_mgd' by default.")
            self.criterion_root = nn.MSELoss()
            self.criterion_quaternions = MeanGeodesicDistance()
        self.criterion_counter = nn.MSELoss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad

        trg_size = model_cfg.get("trg_size", 199)  # 3d root + N_bones * 4d quaternions = 3 + 49 * 4
        self.target_size = trg_size + 1  # adding the counter

        self.root_loss_scale = model_cfg.get("root_loss_scale", 1.0)
        self.quaternions_loss_scale = model_cfg.get("quaternions_loss_scale", 1.0)
        self.counter_loss_scale = model_cfg.get("counter_loss_scale", 1.0)

    def forward(self, preds: Tensor, targets: Tensor):
        # `preds`: shape=(M_batch, T * (3 + N_bones * 4 + 1))
        # `targets`: shape=(M_batch, T * (3 + N_bones * 4 + 1))

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Extract root joints, quaternions and counter from inputs
        M = targets_masked.shape[0]  # batch size
        assert preds_masked.shape[0] == M

        # --- reshaping
        preds_masked_reshaped = preds_masked.reshape(M, -1, self.target_size)  # shape=(M, T, 3 + N_bones * 4 + 1)
        targets_masked_reshaped = targets_masked.reshape(M, -1, self.target_size)

        # --- root joints (M_batch, T, 3)
        preds_root_masked = preds_masked_reshaped[:, :, :3].reshape(M, -1)  # shape=(M, T * 3)
        targets_root_masked = targets_masked_reshaped[:, :, :3].reshape(M, -1)

        # --- quaternions (M_batch, T, N_bones * 4)
        preds_quaternions_masked = preds_masked_reshaped[:, :, 3:-1].reshape(M, -1)  # shape=(M, T * N_bones * 4)
        targets_quaternions_masked = targets_masked_reshaped[:, :, 3:-1].reshape(M, -1)
        if "mgd" in self.loss:
            preds_quaternions_masked = preds_quaternions_masked.reshape(M, -1, 4)  # shape=(M, T * N_bones, 4)
            targets_quaternions_masked = targets_quaternions_masked.reshape(M, -1, 4)  # shape=(M, T * N_bones, 4)

        # --- root joints (M_batch, T, 3)
        preds_counter_masked = preds_masked_reshaped[:, :, -1:].reshape(M, -1)  # shape=(M, T * 1)
        targets_counter_masked = targets_masked_reshaped[:, :, -1:].reshape(M, -1)

        # Calculate losses just over the masked predictions
        ar, aq, ac = self.root_loss_scale, self.quaternions_loss_scale, self.counter_loss_scale

        loss_root = self.criterion_root(preds_root_masked, targets_root_masked)
        loss_quaternions = self.criterion_quaternions(preds_quaternions_masked, targets_quaternions_masked)
        loss_counter = self.criterion_counter(preds_counter_masked, targets_counter_masked)

        loss = ar * loss_root + aq * loss_quaternions + ac * loss_counter

        return loss


class MeanGeodesicDistance(nn.Module):
    """
    Mean geodesic distance between pairs of tensors of quaternions (V, V').
    Formula:
                d(V, V') := (1/N) * sum( arccos(2 (qi . qi')^2 - 1)) )

                    if      V  = (q1, ..., qi, ..., qN)
                    and     V' = (q1', ..., qi', ..., qN')
    """

    def __init__(
            self,
            reduction: str = "sum"  # /!\ NO "mean" reduction because division by N_batch is done in TrainManager _train_batch() method
    ):
        super(MeanGeodesicDistance, self).__init__()
        self.reduction = reduction

    def forward(self, preds_quaternions, targets_quaternions):
        # Notes:
        # `preds_quaternions`: shape=(M_batch, N, 4)
        # `targets_quaternions`: shape=(M_batch, N, 4)
        # all quaternions are already normalized (unit norms)

        # ---- dot product along last dimension (quaternion components)
        dot_products = torch.sum(preds_quaternions * targets_quaternions, dim=-1)  # shape=(M_batch, N)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)  # to ensure numerical stability

        # ---- distance with clamping to ensure correct input value for arc cosine and numerical stability
        # (+/- 1e-6 so that the derivative of acos which is 1 / (1 - x^2) do not go 1/0)
        distances = torch.acos(torch.clamp(2 * (dot_products ** 2) - 1, -1.0 + 1e-6, 1.0 - 1e-6))  # shape=(M_batch, N)

        # ---- mean over all limbs quaternions for each sample in the batch
        mean_geo_distance_per_pair = distances.mean(dim=-1)  # shape=(M_batch,)

        # ---- reduction over the batch pairs
        if self.reduction == "mean":
            return mean_geo_distance_per_pair.mean()  # scalar
        if self.reduction == "sum":
            return mean_geo_distance_per_pair.sum()  # scalar
        else:
            return mean_geo_distance_per_pair  # shape=(M_batch,)


class RegLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()

        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked)

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss
