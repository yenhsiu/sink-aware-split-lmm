import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import matplotlib.pyplot as plt
import numpy as np
import json
from turboquant.compressors_v3 import MSECompressor, StratifiedCompressor

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output


# ── PruneSID selection (ported from PruneSID: "Prune Redundancy, Preserve
#    Essence: Vision Token Compression in VLMs via Synergistic
#    Importance-Diversity", Fang et al., ICLR 2026) ─────────────────────────
# Selects tokens via PCA-based clustering + intra-cluster similarity NMS
# instead of raw top-k by CLS-attention. This bounds how much budget any
# single attention "sink" outlier (see Darcet et al. 2023, "Vision
# Transformers Need Registers") can consume, since it can win at most its own
# cluster's quota rather than crowding out the global top-k ranking.

def _prunesid_batch_pca(features, min_components=32):
    standard_features = torch.sigmoid(features.to(torch.float32)).transpose(2, 1)[:, :, 1:]
    U, S, V = torch.pca_lowrank(standard_features, q=min_components)
    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=-1)
    return V, belong_components


def _prunesid_similarity_nms(similarity_matrix, scores, threshold):
    """
    similarity_matrix: [batch, group, N, N]
    scores: [batch, N, group]
    threshold: [batch]
    """
    new_scores = scores.clone()
    return_scores = new_scores.clone()
    given_score = 1000
    keep = []

    batch_size, group, N, _ = similarity_matrix.shape
    threshold = threshold.unsqueeze(1).unsqueeze(2).expand(batch_size, group, N)
    batch_idx = torch.arange(batch_size, device=new_scores.device).view(-1, 1, 1).expand(-1, group, N)
    group_idx = torch.arange(group, device=new_scores.device).view(1, -1, 1).expand(batch_size, -1, N)
    col_idx = torch.arange(N, device=new_scores.device).view(1, 1, -1).expand(batch_size, group, -1)
    while new_scores.sum() > 0:
        max_values, max_idx = new_scores.max(dim=1)

        row_idx = max_idx.unsqueeze(-1).expand(-1, -1, N)
        sim_row = similarity_matrix[batch_idx, group_idx, row_idx, col_idx]
        max_idx[max_values == 0] = -1
        given_score_indices_0, given_score_indices_2 = torch.where(max_idx != -1)
        given_score_indices_1 = max_idx[given_score_indices_0, given_score_indices_2]
        return_scores[given_score_indices_0, given_score_indices_1, given_score_indices_2] = given_score
        new_scores[given_score_indices_0, given_score_indices_1, given_score_indices_2] = 0
        given_score -= 1
        keep.append(max_idx.unsqueeze(-1))

        condition = sim_row > threshold
        new_scores[condition.transpose(1, 2)] = 0
        if new_scores.sum() == 0:
            break
    keep = torch.cat(keep, dim=-1)
    return keep, return_scores


def _detect_sink_count(sorted_scores: torch.Tensor, max_candidates: int = 2, ratio_threshold: float = 100.0) -> int:
    """Number of attention sinks: the tokens scoring more than
    `ratio_threshold`x the median are sink candidates, and at most the
    `max_candidates` highest of them are sinks. Since the candidates form a
    prefix of the descending order, this equals checking the top
    `max_candidates` scores in order. A candidate is flagged if it's more than
    `ratio_threshold`x the *median* of the full score distribution (the
    median is a robust stand-in for "typical, non-sink" score -- using the
    adjacent-rank ratio instead was found to be miscalibrated once attention
    is summed properly across heads, since summing dilutes the per-head
    cliff sharpness even though each candidate is still ~100x the typical
    token). Checked in descending order; stops at the first candidate that
    doesn't clear the bar (later ones, being smaller, can't clear it either).
    Returns 0 if even the top candidate isn't a sink."""
    if len(sorted_scores) == 0:
        return 0
    median = sorted_scores.median().item()
    if median <= 0:
        return 0
    n = min(max_candidates, len(sorted_scores))
    count = 0
    for i in range(n):
        if sorted_scores[i].item() > ratio_threshold * median:
            count = i + 1
        else:
            break
    return count


def _capa_contribution(vision_tower, image_forward_outs, layer_idx: int, dtype, weighted: bool = True) -> torch.Tensor:
    """Per-patch-token contribution score, decomposed per head (Kobayashi-
    style decomposed-attention norm):

        C_i = || sum_h A_{i,h} (x_i W_{V,h}) W_{O,h} ||_2      (weighted=True, the CAPA score)
        U_i = || sum_h        (x_i W_{V,h}) W_{O,h} ||_2       (weighted=False)

    where A_{i,h} is the CLS query's attention weight on key token i in head
    h (same layer/quantity as cls_attn_sum, just not yet summed over heads),
    x_i is the *pre-LayerNorm1* residual-stream input to this layer (what
    v_proj actually consumes -- CLIP is pre-LN), and W_{V,h}/W_{O,h} are that
    layer's value/output-projection weights restricted to head h.

    C_i (weighted=True) is the literal CAPA definition -- how much a token
    actually injects into the residual stream *given the attention weight it
    received*. Diagnostic finding (2026-08-11, 25-image spot check): C_i
    still lets attention-sink outliers rank moderately high (median rank ~12
    of 576) purely because A_{i,h} is a multiplicative factor -- a huge
    attention weight partially compensates for a near-zero value vector.
    U_i (weighted=False) drops that factor entirely, isolating the value
    vector's own magnitude -- on the same spot check, previously-flagged
    sink candidates ranked dead last by U_i (median rank 575 of 576, 0/49 in
    the top 10), i.e. U_i cleanly exposes that their *content* carries
    essentially nothing, independent of how much attention they happened to
    receive. LLAVA_SINK_MODE='value_rank' uses U_i for tier ranking.

    Returns [B, 576] (patch-index space, CLS excluded), aligned with
    cls_attn_sum / scores_b elsewhere in token_prunesid_quant_select.
    """
    layer = vision_tower.vision_model.encoder.layers[layer_idx]
    self_attn = layer.self_attn
    num_heads, head_dim = self_attn.num_heads, self_attn.head_dim

    # hidden_states[-3] == input to `layer_idx` (== attentions[-2]'s layer);
    # hidden_states[-2] (used elsewhere as "features") is that layer's OUTPUT.
    pre_ln_input = image_forward_outs.hidden_states[-3].to(dtype)
    normed = layer.layer_norm1(pre_ln_input)
    v = self_attn.v_proj(normed)                      # [B, 577, C]
    B, S, C = v.shape
    v_heads = v.view(B, S, num_heads, head_dim)[:, 1:, :, :]  # [B, 576, heads, head_dim] -- drop CLS

    if weighted:
        attn_weights = image_forward_outs.attentions[layer_idx]   # [B, heads, 577, 577]
        a_cls = attn_weights[:, :, 0, 1:].transpose(1, 2)          # [B, 576, heads]
        per_head = (a_cls.unsqueeze(-1) * v_heads)                 # [B, 576, heads, head_dim]
    else:
        per_head = v_heads                                         # [B, 576, heads, head_dim], no A_{i,h} factor

    flat = per_head.reshape(B, S - 1, C)             # [B, 576, C], heads concatenated
    contribution = flat @ self_attn.out_proj.weight.T  # [B, 576, C]
    return contribution.norm(dim=-1)                   # [B, 576]


def _fit_tier_sizes(n4: int, n2: int, n1: int, n_total: int, n_non_sink: int):
    """Cap n4/n2 by how many non-sink tokens are actually available among the
    selected N; n1 absorbs everything else (all sink tokens + any non-sink
    overflow), so the three tiers always sum to exactly n_total."""
    n4_actual = min(n4, n_non_sink)
    n2_actual = min(n2, n_non_sink - n4_actual)
    n1_actual = n_total - n4_actual - n2_actual
    return [(n4_actual, 4), (n2_actual, 2), (n1_actual, 1)]

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer # default: -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.total_tokens = 0

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        # _prunesid_batch_pca calls torch.pca_lowrank, which internally draws
        # a random Gaussian test matrix for its randomized low-rank SVD (see
        # torch docs) -- unseeded, this makes PruneSID's token selection (and
        # therefore every downstream accuracy number) non-deterministic
        # across process runs. Confirmed empirically 2026-08-11: identical
        # config run twice with no seed produced different MME answers (2/12
        # flipped on a spot check); with this seed set, two runs produced
        # byte-identical output. Seeded once per process, matching this
        # codebase's existing seed=42 convention (SPLIT_SEED, compressor
        # seed) -- not reset per-image, so it's a fixed deterministic
        # sequence over the whole run, reproducible run-to-run. Configurable
        # via LLAVA_PCA_SEED (default 42) so multi-seed robustness checks can
        # confirm a result isn't just an artifact of one arbitrary draw.
        pca_seed = int(os.environ.get('LLAVA_PCA_SEED', '42'))
        torch.manual_seed(pca_seed)
        torch.cuda.manual_seed_all(pca_seed)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        use_quant = os.environ.get('LLAVA_USE_QUANT', 'false').lower() == 'true'
        if use_quant:
            quant_mode = os.environ.get('LLAVA_QUANT_MODE', 'uniform')
            if quant_mode == 'stratified':
                groups_str = os.environ.get('LLAVA_STRAT_GROUPS', '40:4,40:2,20:1')
                groups = [(int(n), int(b)) for n, b in (g.split(':') for g in groups_str.split(','))]
                self.compressor = StratifiedCompressor(head_dim=1024, groups=groups, seed=42, device="cuda")
            else:
                quant_bits = int(os.environ.get('LLAVA_QUANT_BITS', '2'))
                self.compressor = MSECompressor(head_dim=1024, bits=quant_bits, seed=42, device="cuda")
            self.quant_mode = quant_mode
        else:
            self.compressor = None
            self.quant_mode = None

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # penultimate layer output
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    @staticmethod
    def _n_tokens_from_env() -> int:
        """Compute N = sum of n values from LLAVA_STRAT_GROUPS, fallback to LLAVA_N_TOKENS."""
        strat = os.environ.get('LLAVA_STRAT_GROUPS', '')
        if strat:
            return sum(int(g.split(':')[0]) for g in strat.split(',') if g)
        return int(os.environ.get('LLAVA_N_TOKENS', '65'))

    def _strat_tier_sizes(self):
        """(n4, n2, n1) parsed from self.compressor.groups (bits -> count); 0 if not stratified."""
        if self.compressor is None or self.quant_mode != 'stratified':
            return 0, 0, 0
        by_bits = {b: n for n, b in self.compressor.groups}
        return by_bits.get(4, 0), by_bits.get(2, 0), by_bits.get(1, 0)


    def token_prune_select(self, images):
        """
        Select top-N tokens by CLS attention score, no merging.
        Returns tokens sorted by attention (desc) for stratified quantization.
        """
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        C = image_features.shape[2]

        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]
        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)
        cls_attn = attn[:, 0, 1:]  # [B, N]

        n_tokens = self._n_tokens_from_env()
        # topk returns indices sorted by value (largest first) — used directly for stratified grouping
        _, top_idx = torch.topk(cls_attn, n_tokens, dim=1, largest=True)
        index = top_idx.unsqueeze(-1).expand(-1, -1, C)
        selected = torch.gather(image_features, dim=1, index=index)  # [B, n_tokens, C]
        return selected

    def token_prune_merge_select(self, images):
        """
        Prune + Merge + Sort for stratified quantization.
        Same merge logic as token_prune_merge_advanced (PruMerge paper),
        but with fixed N = LLAVA_N_TOKENS and output sorted by attention for stratified quant.
        """
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N_total, C = image_features.shape  # N_total = 576

        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]
        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)
        cls_attn = attn[:, 0, 1:]  # [B, 576]

        n_tokens = self._n_tokens_from_env()

        # top-N tokens (sorted by attention descending — order matters for stratified quant)
        _, top_idx = torch.topk(cls_attn, n_tokens, dim=1, largest=True, sorted=True)
        index = top_idx.unsqueeze(-1).expand(-1, -1, C)

        x_others = torch.gather(image_features, dim=1, index=index)          # [B, N, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=top_idx)         # [B, N]

        if n_tokens >= N_total:
            # No leftover (non-topk) tokens to merge in -- every token is
            # already selected. Falling through to the merge loop below would
            # degenerate into averaging every token with every *other* kept
            # token (since `compl` is empty), which is a different, far more
            # destructive operation than the intended "merge unselected
            # tokens into selected ones". Nothing to merge: return as-is.
            return x_others

        compl = complement_idx(top_idx, N_total)                              # [B, 576-N]
        non_topk = torch.gather(image_features, dim=1,
                                index=compl.unsqueeze(-1).expand(-1, -1, C)) # [B, 576-N, C]
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)           # [B, 576-N]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, 576, C]
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)            # [B, N, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1,
                                    index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, 576-N, C]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # same as original: for each top token, find k=32 most similar tokens
        # from all other tokens (other top + non-top), weighted sum + self
        updated_x_others = torch.zeros_like(x_others)
        left_tokens = n_tokens

        for b in range(B):
            for i in range(left_tokens):
                key_i_norm = Key_others_norm[b, i, :].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)
                after_i_Key  = Key_others_norm[b, i+1:, :].unsqueeze(0)
                before_i_x   = x_others[b, :i, :].unsqueeze(0)
                after_i_x    = x_others[b, i+1:, :].unsqueeze(0)
                before_i_attn = x_others_attn[b, :i].unsqueeze(0)
                after_i_attn  = x_others_attn[b, i+1:].unsqueeze(0)

                rest_keys  = torch.cat([before_i_Key, after_i_Key,
                                        non_topk_Key_norm[b, :, :].unsqueeze(0)], dim=1)
                rest_x     = torch.cat([before_i_x, after_i_x,
                                        non_topk[b, :, :].unsqueeze(0)], dim=1)
                rest_attn  = torch.cat([before_i_attn, after_i_attn,
                                        non_topk_attn[b, :].unsqueeze(0)], dim=1)

                cos_sim = torch.bmm(key_i_norm, rest_keys.transpose(1, 2))
                k = min(n_tokens, rest_keys.shape[1])
                _, cluster_indices = torch.topk(cos_sim, k=k, dim=2, largest=True)

                cluster_indices_flat = cluster_indices.view(-1)
                cluster_tokens = rest_x[:, cluster_indices_flat, :]
                weights = rest_attn[:, cluster_indices_flat].unsqueeze(-1)

                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
                updated_x_others[b, i, :] = weighted_avg + x_others[b, i, :]

        # already sorted by attention (topk with sorted=True), preserve order
        return updated_x_others  # [B, N, C]

    @torch.no_grad()
    def token_prunesid_quant_select(self, images):
        """
        PruneSID token selection (PCA-cluster + intra-cluster similarity NMS,
        diversity-aware -- see module-level docstring above _prunesid_batch_pca)
        followed by sink-aware stratified quantization:

          1. Select N tokens the PruneSID way (bounds how much any single
             attention-sink outlier can dominate the selection).
          2. Rank the selected tokens by CLS-attention score.
          3. Detect attention-sink outliers among them via a dynamic cliff
             (see _detect_sink_count) -- these are *kept* in the sequence
             (per Xiao et al. 2023 / Darcet et al. 2023, forcibly removing
             sink tokens can destabilize the model) but excluded from the
             high-precision ranking, i.e. pinned to the lowest bit tier
             regardless of their (sink-inflated) raw attention rank. This is
             LLAVA_SINK_MODE='pin' (default). LLAVA_SINK_MODE='capa' instead
             computes each sink candidate's CAPA attention-contribution score
             (see _capa_contribution) and only pins it if that score is below
             the median contribution of the N selected tokens -- a sink
             candidate that's actually injecting real information into the
             residual stream is left to compete for tiers normally (still by
             raw attention score, since only the pin/spare call used CAPA).
             LLAVA_SINK_MODE='capa_rank' goes further and drops sink
             detection entirely: every one of the N selected tokens is
             ranked directly by CAPA contribution, so the bit-tier a token
             gets is decided by how much it actually contributes, full stop
             -- no outlier heuristic, no separate sink-candidate pool.
             LLAVA_SINK_MODE='capa_full' goes further still: CAPA also
             replaces CLS-attention as the score PruneSID's own cluster-NMS
             selection (step 1 above) picks representatives by -- so a fake
             sink (huge attention, near-zero real contribution) can't even
             win its cluster's selection round, let alone dominate the
             ranking afterward. CAPA becomes the only importance signal used
             anywhere in this method.
             LLAVA_SINK_MODE='value_rank' ranks the N selected tokens like
             'capa_rank' but by U_i (see _capa_contribution, weighted=False)
             -- the value vector's own magnitude with the attention-weight
             factor dropped entirely. Diagnostic finding: previously-flagged
             sink candidates still rank moderately high (~top 2%) by the
             attention-weighted C_i, purely because a huge A_{i,h} partially
             offsets a near-zero value vector -- but rank dead last (median
             575/576) by U_i, which isolates that their content itself
             carries nothing. U_i separates true sinks far more cleanly.
             LLAVA_SINK_MODE='capa_rank_cap' combines both findings: rank
             by C_i (capa_rank) since it's the strongest single ranking
             signal, but a detected sink candidate (_detect_sink_count,
             same heuristic as 'pin') is barred from the 4-bit tier no
             matter how high C_i ranks it -- capped at 2-bit max, still
             competing normally for 2-bit/1-bit by C_i rank otherwise.
          4. The remaining non-sink tokens (or, in 'capa_rank'/'capa_full'/
             'value_rank'/'capa_rank_cap', every token by contribution
             rank) fill the 4-bit/2-bit tiers by score rank as usual; turboquant's
             StratifiedCompressor quantizes the reordered patch tokens. CLS
             token (always index 0, kept full-precision, not counted in the
             N-token budget) is concatenated back on afterwards.
        """
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True, output_attentions=True,
        )
        attn_weights = image_forward_outs.attentions[-2]          # [B, heads, 577, 577]
        hidden_states = image_forward_outs.hidden_states[-2].to(images.dtype)  # [B, 577, C]
        B, _, C = hidden_states.shape

        cls_attn = attn_weights[:, :, 0, 1:]          # [B, heads, 576]
        cls_attn_sum = cls_attn.sum(dim=1)             # [B, 576] -- same score PruneSID itself ranks/clusters by

        need_token_num = self._n_tokens_from_env()

        # 'capa_full' feeds CAPA contribution into PruneSID's own cluster-NMS
        # selection (not just the post-selection tier ranking) -- computed
        # here, before PCA/NMS, since it needs to replace cls_attention_sum
        # as the score _prunesid_similarity_nms picks representatives by.
        sink_mode_early = os.environ.get('LLAVA_SINK_MODE', 'pin').lower()
        capa_score_early = None
        if sink_mode_early in ('capa_rank', 'capa_full', 'capa_rank_cap') or (sink_mode_early == 'capa' and os.environ.get('LLAVA_SINK_FILTER', 'true').lower() == 'true'):
            capa_layer_idx_early = len(self.vision_tower.vision_model.encoder.layers) - 2
            capa_score_early = _capa_contribution(self.vision_tower, image_forward_outs, capa_layer_idx_early, images.dtype)  # [B, 576]

        # 'value_rank': tier ranking uses U_i (unweighted value-vector
        # magnitude, see _capa_contribution's weighted=False path) instead of
        # C_i -- doesn't touch PruneSID selection (unlike 'capa_full').
        value_score_early = None
        if sink_mode_early == 'value_rank':
            value_layer_idx_early = len(self.vision_tower.vision_model.encoder.layers) - 2
            value_score_early = _capa_contribution(self.vision_tower, image_forward_outs, value_layer_idx_early, images.dtype, weighted=False)  # [B, 576]

        # ── 1. PruneSID selection: PCA-cluster + NMS diversity ──
        projector_lengths, belong_components = _prunesid_batch_pca(
            hidden_states, min_components=max(1, int(need_token_num / 4))
        )
        cls_attention_sum = cls_attn_sum
        # 'capa_full': CAPA contribution drives which token wins each NMS
        # round (and therefore which tokens are even selected), not just
        # attention magnitude -- the whole point being that a fake sink
        # (huge attention, near-zero real contribution) shouldn't get to act
        # as a cluster's "representative" and suppress its real neighbors.
        selection_score = capa_score_early if sink_mode_early == 'capa_full' else cls_attention_sum
        projector_scores = selection_score.unsqueeze(-1).repeat(1, 1, projector_lengths.shape[-1]).to(projector_lengths.dtype)
        projector_mask = belong_components.unsqueeze(-1).repeat(1, 1, projector_lengths.shape[-1])
        index_map = torch.arange(projector_lengths.shape[-1], device=projector_lengths.device).unsqueeze(0).unsqueeze(0).repeat(projector_lengths.shape[0], projector_lengths.shape[1], 1)
        weights_mask = torch.where(projector_mask != index_map)
        projector_lengths[weights_mask] = 0
        projector_scores[weights_mask] = 0

        normalized_states = F.normalize(hidden_states[:, 1:, :], p=2, dim=-1)
        similarity = torch.bmm(normalized_states, normalized_states.transpose(2, 1)).to(torch.float32)
        triu_mask = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
        sim_mean = (similarity * triu_mask).sum(-1).sum(-1) / triu_mask.sum(-1).sum(-1)
        ratio = need_token_num / 32

        group_similarity = similarity.clone().unsqueeze(1).repeat(1, projector_lengths.shape[-1], 1, 1)
        group_similarity_masks = torch.zeros_like(group_similarity)
        group_index = torch.arange(group_similarity.shape[1], device=group_similarity.device).unsqueeze(0).unsqueeze(-1).repeat(group_similarity.shape[0], 1, group_similarity.shape[-2])
        group_belong_components = belong_components.unsqueeze(1).repeat(1, group_similarity.shape[1], 1)
        group_similarity_masks[group_index == group_belong_components] = 1
        group_similarity_masks[torch.where(group_similarity_masks.transpose(3, 2) == 1)] = 1
        group_similarity[group_similarity_masks != 1] = 0
        group_ids, projector_scores = _prunesid_similarity_nms(group_similarity, projector_scores, ratio * sim_mean)
        group_ids_mask = (group_ids != -1)
        keep_nms_counts = group_ids_mask.sum(dim=-1)
        group_counts = (projector_mask == index_map).sum(dim=1)
        group_lower_bound = torch.ones_like(group_counts, device=group_counts.device)
        group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
        group_upper_bound = torch.ones_like(group_counts, device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0), dim=0)[0]
        while torch.any(group_upper_bound.sum(-1) < need_token_num):
            group_upper_bound[torch.where(group_upper_bound.sum(-1) < need_token_num)] += 1
            group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]

        other_token_nums = need_token_num - group_lower_bound.sum(-1) - 1
        other_token_nums[other_token_nums < 0] = 0
        norm_group_counts = keep_nms_counts / keep_nms_counts.sum(-1, keepdim=True)
        cumulative_sum = torch.cumsum(norm_group_counts, dim=-1)
        other_token_d = (cumulative_sum * other_token_nums.unsqueeze(-1).expand(-1, group_counts.shape[1])).round().int()
        other_token_d = other_token_d - torch.cat([torch.zeros((other_token_d.shape[0], 1), device=other_token_d.device), other_token_d[:, :-1]], dim=-1)
        group_token_d = other_token_d + group_lower_bound
        group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
        group_sort_index = torch.argsort(keep_nms_counts, dim=-1, descending=True)
        filling_group = torch.zeros(group_counts.shape[0], device=group_counts.device).int()
        while torch.any(group_token_d.sum(-1) < other_token_nums + group_lower_bound.sum(-1)):
            need_filling_batch = torch.where(group_token_d.sum(-1) < other_token_nums + group_lower_bound.sum(-1))[0]
            filling_num = torch.min(torch.stack(
                [group_upper_bound[need_filling_batch, group_sort_index[need_filling_batch, filling_group[need_filling_batch]]] - group_token_d[need_filling_batch, group_sort_index[need_filling_batch, filling_group[need_filling_batch]]],
                 other_token_nums[need_filling_batch] + group_lower_bound[need_filling_batch].sum(-1) - group_token_d[need_filling_batch].sum(-1)]
            ), dim=0)[0]
            group_token_d[need_filling_batch, group_sort_index[need_filling_batch, filling_group[need_filling_batch]]] += filling_num
            filling_group[need_filling_batch] += 1
        projector_sort_index = torch.argsort(projector_scores, dim=1, descending=True)
        projector_sort_index = projector_sort_index.transpose(1, 2).reshape(-1, group_similarity.shape[-1])
        group_token_d = group_token_d.reshape(-1)
        important_indices = []
        for i in range(len(group_token_d)):
            important_indices.append(projector_sort_index[i][:int(group_token_d[i])])
        important_indices = [important_indices[i:i + group_similarity.shape[1]] for i in range(0, len(important_indices), group_similarity.shape[1])]

        # ── 2/3/4. per-batch-item: rank selected patches by score, flag
        #    sink outliers, assign bit tiers, quantize ──
        n4, n2, n1 = self._strat_tier_sizes()
        sink_enabled = os.environ.get('LLAVA_SINK_FILTER', 'true').lower() == 'true'
        sink_ratio = float(os.environ.get('LLAVA_SINK_RATIO', '100.0'))
        sink_max_candidates = int(os.environ.get('LLAVA_SINK_MAX_CANDIDATES', '2'))
        # 'pin' (default): every detected sink candidate is unconditionally
        # pinned to the lowest tier (original behavior).
        # 'capa': a detected sink candidate is only pinned if its CAPA
        # attention-contribution score (see _capa_contribution) is below the
        # median contribution among the N PruneSID-selected tokens -- i.e. if
        # it's genuinely uninformative, not just attention-inflated. A sink
        # candidate whose contribution is at/above that median is judged to
        # be doing real work and competes for tiers normally instead -- but
        # still by *raw attention score*, since only the pin/spare decision
        # used CAPA, not the tier ranking itself.
        # 'capa_rank': no sink detection at all. Every one of the N
        # PruneSID-selected tokens is ranked directly by CAPA contribution
        # and tiers fill top-down by that score -- sink or not, a token's
        # actual contribution alone decides its bit-width. Supersedes 'pin'
        # and 'capa': there's no separate outlier-detection heuristic left to
        # tune (sink_ratio / sink_max_candidates are unused in this mode).
        # 'capa_full': like 'capa_rank' for tier ranking, but CAPA also
        # already drove the PruneSID selection step itself above (see
        # selection_score) -- the deepest integration, CAPA is the only
        # importance signal used anywhere in this method.
        # 'value_rank': like 'capa_rank', but ranks by U_i (unweighted value
        # magnitude) instead of C_i -- see _capa_contribution's docstring for
        # why U_i separates true attention-sink tokens (near-zero intrinsic
        # content) more cleanly than the attention-weighted CAPA score does.
        sink_mode = sink_mode_early  # already read above, before PCA/NMS
        capa_score = capa_score_early
        value_score = value_score_early

        batch_out = []
        for b in range(B):
            patch_idx = torch.cat(important_indices[b]).to(hidden_states.device)  # 0..574 indexing (into the 575 non-CLS tokens' PCA/similarity space)
            scores_b = cls_attention_sum[b]  # [576], patch-index space (0..575)

            if sink_mode in ('capa_rank', 'capa_full'):
                # Pure contribution ranking, no sink concept: tier sizes fall
                # back to plain budget caps (n_non_sink == every selected
                # token, since none are exempted/pinned a priori). In
                # 'capa_full' patch_idx itself was also chosen by CAPA (see
                # selection_score above) -- this just ranks that set.
                sel_capa = capa_score[b, patch_idx]  # [N], same order as patch_idx
                ordered_local = torch.argsort(sel_capa, descending=True)
                tier_sizes_actual = _fit_tier_sizes(n4, n2, n1, len(patch_idx), len(patch_idx))
            elif sink_mode == 'value_rank':
                # Same pure-ranking structure as capa_rank, but by U_i.
                sel_value = value_score[b, patch_idx]  # [N], same order as patch_idx
                ordered_local = torch.argsort(sel_value, descending=True)
                tier_sizes_actual = _fit_tier_sizes(n4, n2, n1, len(patch_idx), len(patch_idx))
            elif sink_mode == 'capa_rank_cap':
                # capa_rank's C_i ranking (best-performing single signal at
                # the lowest budget), plus a hard ceiling: a token flagged as
                # a raw-attention sink candidate (same _detect_sink_count
                # heuristic as 'pin'/'capa') is barred from the 4-bit tier no
                # matter how high its C_i ranks it, since we've shown its
                # *unweighted* value content (U_i) is typically near-zero --
                # C_i only ranks it highly because of the A_{i,h} factor. It
                # still competes normally for 2-bit/1-bit by C_i rank among
                # the rest, it just can never win a 4-bit slot.
                full_sorted_scores, full_sorted_idx = torch.sort(scores_b, descending=True)
                sink_count = _detect_sink_count(full_sorted_scores, sink_max_candidates, sink_ratio)
                sink_patch_set = set(full_sorted_idx[:sink_count].tolist())

                sel_capa = capa_score[b, patch_idx]  # [N], same order as patch_idx
                rank_order = torch.argsort(sel_capa, descending=True).tolist()  # local indices, best C_i first
                selected_list = patch_idx.tolist()
                rank_pos = {local: pos for pos, local in enumerate(rank_order)}

                normal_local = [i for i in rank_order if selected_list[i] not in sink_patch_set]
                sink_local   = [i for i in rank_order if selected_list[i] in sink_patch_set]

                tier4_local = normal_local[:n4]                         # sink tokens never eligible here
                remaining_local = normal_local[n4:] + sink_local
                remaining_local.sort(key=lambda i: rank_pos[i])         # restore overall C_i rank order

                ordered_local = torch.tensor(tier4_local + remaining_local, device=hidden_states.device)
                tier_sizes_actual = _fit_tier_sizes(n4, n2, n1, len(patch_idx), len(patch_idx))
            else:
                if sink_enabled:
                    full_sorted_scores, full_sorted_idx = torch.sort(scores_b, descending=True)
                    sink_count = _detect_sink_count(full_sorted_scores, sink_max_candidates, sink_ratio)
                    sink_patch_set = set(full_sorted_idx[:sink_count].tolist())

                    if sink_mode == 'capa' and sink_patch_set:
                        selected_set = set(patch_idx.tolist())
                        candidates_in_pool = sink_patch_set & selected_set
                        if candidates_in_pool:
                            pool_capa = capa_score[b, patch_idx]  # [N], same order as patch_idx
                            capa_median = pool_capa.median()
                            contributing = {p for p in candidates_in_pool if capa_score[b, p] >= capa_median}
                            sink_patch_set -= contributing  # only demote non-contributing sinks
                else:
                    sink_patch_set = set()

                selected_list = patch_idx.tolist()
                is_sink = torch.tensor([p in sink_patch_set for p in selected_list], device=hidden_states.device)
                sel_scores = scores_b[patch_idx]

                non_sink_order = torch.argsort(torch.where(is_sink, torch.full_like(sel_scores, float('-inf')), sel_scores), descending=True)
                ordered_local = non_sink_order  # non-sink tokens first (by score desc), sink tokens last

                n_non_sink = int((~is_sink).sum().item())
                tier_sizes_actual = _fit_tier_sizes(n4, n2, n1, len(selected_list), n_non_sink)

            ordered_patch_idx = patch_idx[ordered_local]  # [N] patch indices, tier-ready order
            ordered_features = hidden_states[b, 1:, :][ordered_patch_idx]  # [N, C], skip CLS at hidden_states[:,0]

            if self.compressor is not None:
                original_dtype = ordered_features.dtype
                if self.quant_mode == 'stratified':
                    # Reuse the per-bit-width MSECompressors already built once
                    # in load_model() (self.compressor.compressors) -- building
                    # a fresh StratifiedCompressor here would rebuild each
                    # LloydMaxCodebook (a ~200-iteration CPU-only numerical
                    # solve) from scratch on *every single image*, which is
                    # the dominant cost of this whole method (confirmed by
                    # profiling: near-0% GPU utilization during that step).
                    # Tier sizes vary per image (sink adjustment), but the
                    # codebook per bit-width does not, so just re-slice.
                    quantized = torch.empty_like(ordered_features, dtype=torch.float32)
                    offset = 0
                    for n_tok, bits in tier_sizes_actual:
                        if n_tok == 0:
                            continue
                        group = ordered_features[offset:offset + n_tok, :].float()
                        comp = self.compressor.compressors[bits].compress(group.unsqueeze(0).unsqueeze(0))
                        recon = self.compressor.compressors[bits].decompress(comp).squeeze(0).squeeze(0)
                        quantized[offset:offset + n_tok, :] = recon
                        offset += n_tok
                    ordered_features = quantized.to(dtype=original_dtype)
                else:
                    # uniform: every selected token gets the same bit-width,
                    # so sink status doesn't matter here -- self.compressor is
                    # already a single-bit MSECompressor built in load_model().
                    compressed = self.compressor.compress(ordered_features.unsqueeze(0).unsqueeze(0))
                    reconstructed = self.compressor.decompress(compressed)
                    ordered_features = reconstructed.squeeze(0).squeeze(0).to(dtype=original_dtype)

            cls_feature = hidden_states[b, 0:1, :]
            batch_out.append(torch.cat([cls_feature, ordered_features], dim=0))  # [1+N, C]

        return torch.stack(batch_out)  # [B, 1+N, C]

    def token_prune_merge_advanced(self, images, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 10/03/2024 using the key*key matrix to calculate the cosine similarity
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]  

        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)


                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = weighted_avg + x_others[b, i, :]  
                updated_x_others[b, i, :] = updated_center 
            

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features
    
    
    def token_prune_merge_advanced_plus(self, images, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 24/03/2024 using the spacially smapled tokens to supplement the pruned tokens
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]  

        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        
        # # # print("idx: ", idx)
        if if_adaptive:
            step_length = int(1/reduction_ratio)
            arithmetic_sequence = torch.arange(0, 575, int(step_length/3)).to(device=self.device)
            original_tensor_1d = idx.flatten().to(device=self.device)
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
            concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
            idx = concatenated_tensor
            # # print("idx_new: ", idx)
        else:
            # # this is for training
            step_length = int(1/reduction_ratio)
            new_idx = torch.zeros((idx.size(0), idx.size(1)*2), dtype=torch.long).to(device=self.device)
            for i in range(idx.size(0)):
                arithmetic_sequence = torch.arange(int(step_length/2), 575, int(step_length)).to(device=self.device)
                original_tensor_1d = idx[i].flatten().to(device=self.device)
                filtered_sequence = arithmetic_sequence
                # filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
                concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
                new_idx[i] = concatenated_tensor
            idx = new_idx

        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = x_others[b, i, :]  + weighted_avg 
                updated_x_others[b, i, :] = updated_center 
            
        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features

    
 
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            token_method = os.environ.get('LLAVA_TOKEN_METHOD', 'original')

            if token_method == 'prumerge':
                image_features = self.token_prune_merge_advanced(images, if_adaptive=True, reduction_ratio=1/8)
            elif token_method == 'prumerge_plus':
                image_features = self.token_prune_merge_advanced_plus(images, if_adaptive=True, reduction_ratio=1/8)
            elif token_method == 'prumerge_quant':
                image_features = self.token_prune_select(images)
            elif token_method == 'prumerge_quant_merge':
                image_features = self.token_prune_merge_select(images)
            elif token_method == 'prunesid_quant':
                # Quantization (incl. sink-aware tiering) already happens
                # inside this method, since it needs the sink flags to decide
                # tier assignment per-token -- skip the generic post-hoc
                # compressor block below (would double-quantize otherwise).
                return self.token_prunesid_quant_select(images)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

            if self.compressor is not None:
                original_dtype = image_features.dtype
                if self.quant_mode == 'stratified':
                    image_features = self.compressor.compress_decompress(image_features).to(dtype=original_dtype)
                else:
                    compressed = self.compressor.compress(image_features.unsqueeze(1))
                    reconstructed = self.compressor.decompress(compressed)
                    image_features = reconstructed.squeeze(1).to(dtype=original_dtype)

                # diff = (image_features_q.float() - image_features.float()).abs()
                # print(f"Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
                # exit()


        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
