"""PruneSID selection + our (n4,n2,n1) bit-tier labeling on top of it.
PruneSID picks WHICH N tokens to keep (diversity-aware); we then rank just
those N tokens by CLS-attention score (same signal PruneSID itself uses
internally) to decide who gets 4-bit vs 2-bit vs 1-bit, same scheme as the
original PruMerge tier viz, so the two are directly comparable.

Sink handling matches the method (sink mode 'pin' in clip_encoder.py
token_prunesid_quant_select): tokens scoring above 100x the median are sink
candidates, and at most the two highest candidates are sinks. Any flagged
candidate that is also in the PruneSID-selected N is pinned to the lowest
(1-bit) tier regardless of its attention rank, and gets a distinct highlight
border in the rendered tiles. Set SINK_MAX_CANDIDATES = 576 to show every
token above the threshold instead (the "no cap" variant of the paper's
sensitivity table).

Per-budget PNGs are rendered for all 8 searched anchor budgets (ANCHOR_CONFIGS,
imported live from paper_data.py so this never drifts from the actual search
results). A subset (PAPER_FIGURE_BUDGETS) is additionally composed into a
single multi-panel PDF suitable for the paper.
"""
import importlib.util
import os
import sys

# Make the repo-root `llava` package importable when run from paper/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib
# Embed real (subset) TrueType fonts instead of matplotlib's PDF default
# (Type 3 bitmap-ish fonts) -- journal PDF checkers flag Type 3 fonts as
# non-embeddable/non-searchable text.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw

from paper_data import ANCHOR_CONFIGS
from llava.model.multimodal_encoder.clip_encoder import _detect_sink_count, _fit_tier_sizes

spec = importlib.util.spec_from_file_location(
    "prunesid_clip_encoder",
    # Reference PruneSID implementation (https://github.com/ZhengyaoFang/PruneSID),
    # used here only to reproduce its token selection for the visualization.
    os.path.join(os.environ.get("PRUNESID_ROOT", os.path.join(REPO_ROOT, "third_party", "PruneSID")),
                 "prunesid", "prunesid_llava", "clip_encoder.py"),
)
prunesid_clip_encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prunesid_clip_encoder)
CLIPVisionTower_PruneSID = prunesid_clip_encoder.CLIPVisionTower_PruneSID

from transformers import CLIPImageProcessor, CLIPVisionModel

VISION_TOWER_NAME = "openai/clip-vit-large-patch14-336"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GRID = 24
# MME artwork sample shown in the paper (Figure 5); override with $VIZ_IMAGE.
IMAGE_PATH = os.environ.get(
    "VIZ_IMAGE",
    os.path.join(REPO_ROOT, "playground/data/eval/MME/MME_Benchmark_release_version/artwork/images/41036.jpg"),
)
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "figures", "token_tier_viz")
PAPER_FIGURE_PATH = os.path.join(_HERE, "figures", "figure5_prunesid_tier_viz.pdf")

# (n4, n2, n1) per anchor budget, read live from paper_data.py's search results
# (the actual budget-optimal LUT, not a hand-copied snapshot).
BUDGET_CONFIGS = {
    budget: (cfg["n4"], cfg["n2"], cfg["n1"]) for budget, cfg in ANCHOR_CONFIGS.items()
}

# 2 budgets picked for the paper figure (low / high) -- trimmed from the
# original 4 (25,600 / 106,000 / 273,000 / 589,824) 2026-08-26, paper is
# tight on space; low vs. high still shows the tier-mix contrast clearly.
PAPER_FIGURE_BUDGETS = [25_600, 273_000]

TIER_COLORS = {
    4: (220, 40, 40, 130),
    2: (240, 190, 30, 130),
    1: (60, 120, 220, 130),
    0: (30, 30, 30, 150),
}
LEGEND_LABELS = {4: "4-bit", 2: "2-bit", 1: "1-bit", 0: "pruned"}

SINK_BORDER_COLOR = (255, 0, 255, 255)  # bright magenta, drawn on top of the tier fill
SINK_BORDER_WIDTH = 3
SINK_RATIO = 100.0
SINK_MAX_CANDIDATES = 2  # same cap of two sinks as the pipeline (LLAVA_SINK_MAX_CANDIDATES)


class MiniTower:
    def __init__(self, vision_tower, device, dtype, need_token_num):
        self.vision_tower = vision_tower
        self.device = device
        self.dtype = dtype
        self.need_token_num = need_token_num
        self.select_layer = -2
        self.select_feature = "patch"

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        return image_features


MiniTower.forward = CLIPVisionTower_PruneSID.forward


def tier_map_for_budget(tower_factory, cls_attn_sum, n4, n2, n1):
    """Run PruneSID selection for this budget, then apply the actual pin
    method (clip_encoder.py token_prunesid_quant_select), but with
    SINK_MAX_CANDIDATES uncapped (see module docstring): detect sink
    candidates among the full 576 patches, pin any that got selected to the
    lowest tier regardless of rank, rank everyone else normally.
    Returns (patch_tier[576], selected_sorted, sink_patch_set)."""
    N = n4 + n2 + n1
    tower = tower_factory(N)
    _, batch_indices = tower.forward(tower.pixel_values)
    idx = batch_indices[0].cpu()
    idx = idx[idx != 0] - 1
    selected = idx.tolist()

    full_sorted_scores, full_sorted_idx = torch.sort(cls_attn_sum, descending=True)
    sink_count = _detect_sink_count(full_sorted_scores, SINK_MAX_CANDIDATES, SINK_RATIO)
    sink_patch_set = set(full_sorted_idx[:sink_count].tolist())

    is_sink = [p in sink_patch_set for p in selected]
    non_sink_order = sorted(
        range(len(selected)),
        key=lambda i: (is_sink[i], -cls_attn_sum[selected[i]].item()),
    )  # non-sink tokens first (by score desc), sink tokens last
    n_non_sink = sum(1 for s in is_sink if not s)
    tier_sizes_actual = _fit_tier_sizes(n4, n2, n1, len(selected), n_non_sink)

    ordered_patches = [selected[i] for i in non_sink_order]
    patch_tier = [0] * 576
    offset = 0
    for n_tok, bits in tier_sizes_actual:
        for p in ordered_patches[offset:offset + n_tok]:
            patch_tier[p] = bits
        offset += n_tok

    selected_sink_set = {p for p in selected if p in sink_patch_set}
    return patch_tier, selected, selected_sink_set


def render_tile(base_img, patch_tier, counts, budget, n4, n2, n1, N, sink_set=(), legend=True):
    img = base_img.convert("RGBA").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = img.size[0] / GRID
    for patch_idx in range(576):
        row, col = divmod(patch_idx, GRID)
        color = TIER_COLORS[patch_tier[patch_idx]]
        x0, y0 = col * scale, row * scale
        draw.rectangle([x0, y0, x0 + scale, y0 + scale], fill=color)

    # Sink-pinned tokens get a bright outline on top of their tier fill --
    # visually separates "pinned to 1-bit because it's a detected sink" from
    # "landed in 1-bit naturally by low attention rank".
    for patch_idx in sink_set:
        row, col = divmod(patch_idx, GRID)
        x0, y0 = col * scale, row * scale
        draw.rectangle(
            [x0 + 1, y0 + 1, x0 + scale - 1, y0 + scale - 1],
            outline=SINK_BORDER_COLOR, width=SINK_BORDER_WIDTH,
        )

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    if legend:
        draw2 = ImageDraw.Draw(composited)
        legend_y = 8
        for tier in (4, 2, 1, 0):
            c = TIER_COLORS[tier]
            draw2.rectangle([8, legend_y, 28, legend_y + 16], fill=c[:3])
            draw2.text((34, legend_y), f"{LEGEND_LABELS[tier]}  n={counts[tier]}", fill=(255, 255, 255))
            legend_y += 20
        draw2.rectangle([8, legend_y, 28, legend_y + 16], outline=SINK_BORDER_COLOR[:3], width=SINK_BORDER_WIDTH)
        draw2.text((34, legend_y), f"sink token  n={len(sink_set)}", fill=(255, 255, 255))
        legend_y += 20
        draw2.text((8, legend_y + 4), f"PruneSID+tier  S={budget} (n4={n4}, n2={n2}, n1={n1}, N={N})", fill=(255, 255, 0))
    return composited


def main():
    # PruneSID's torch.pca_lowrank is randomized -- seed it (same 42
    # convention as LLAVA_PCA_SEED in the main pipeline) so this figure is
    # reproducible run-to-run.
    seed = int(os.environ.get("LLAVA_PCA_SEED", "42"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    processor = CLIPImageProcessor.from_pretrained(VISION_TOWER_NAME)
    vt = CLIPVisionModel.from_pretrained(VISION_TOWER_NAME, torch_dtype=torch.float16).to(DEVICE)
    vt.requires_grad_(False)
    vt.eval()

    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    pixel_values = processor(raw_image, return_tensors="pt")["pixel_values"].to(DEVICE, dtype=torch.float16)

    # Same multi-head CLS-attention score PruneSID itself uses internally
    # (HF's proper output_attentions, averaged/summed across all heads --
    # not the single hand-rolled Q/K hook from the earlier sink test).
    with torch.no_grad():
        out = vt(pixel_values, output_hidden_states=True, output_attentions=True)
    attn_weights = out.attentions[-2]  # penultimate layer, same layer PruneSID reads
    cls_attn_sum = attn_weights[:, :, 0, 1:].sum(dim=1)[0].float().cpu()  # [576]

    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    denorm = pixel_values[0].cpu().float() * std + mean
    base_img = Image.fromarray(
        (denorm.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    ).resize((672, 672), Image.NEAREST)

    def tower_factory(need_token_num):
        t = MiniTower(vt, DEVICE, torch.float16, need_token_num=need_token_num)
        t.pixel_values = pixel_values
        return t

    paper_tiles = {}  # budget -> (patch_tier, counts, n4, n2, n1, N, sink_set)

    for S, (n4, n2, n1) in BUDGET_CONFIGS.items():
        N = n4 + n2 + n1
        patch_tier, selected, sink_set = tier_map_for_budget(tower_factory, cls_attn_sum, n4, n2, n1)
        # Count from the actual assigned tiers (not the requested n4/n2/n1) --
        # _fit_tier_sizes can cap n4/n2 below the requested amount if too few
        # non-sink tokens are available, so this reflects what really happened.
        counts = {t: sum(1 for p in selected if patch_tier[p] == t) for t in (4, 2, 1)}
        counts[0] = 576 - len(selected)

        composited = render_tile(base_img, patch_tier, counts, S, n4, n2, n1, N, sink_set=sink_set, legend=True)
        out_path = os.path.join(OUT_DIR, f"prunesid_tier_S{S}.png")
        composited.save(out_path)
        print(f"saved {out_path}  (sink tokens: {len(sink_set)})")

        if S in PAPER_FIGURE_BUDGETS:
            paper_tiles[S] = (patch_tier, counts, n4, n2, n1, N, sink_set)

    make_paper_figure(base_img, paper_tiles)


def make_paper_figure(base_img, paper_tiles):
    """Compose PAPER_FIGURE_BUDGETS into a 1x2 grid vector PDF for the paper."""
    budgets = sorted(paper_tiles)
    ncols = 2
    nrows = (len(budgets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows))
    axes = axes.flatten()
    for ax in axes[len(budgets):]:
        ax.axis("off")

    for ax, S in zip(axes, budgets):
        patch_tier, counts, n4, n2, n1, N, sink_set = paper_tiles[S]
        tile = render_tile(base_img, patch_tier, counts, S, n4, n2, n1, N, sink_set=sink_set, legend=False)
        ax.imshow(tile)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"$B_{{\\mathrm{{budget}}}}$ = {S:,} bits\nN = {N}  (4-bit:{n4}, 2-bit:{n2}, 1-bit:{n1}, sink:{len(sink_set)})", fontsize=14)

    tier_order = (4, 2, 1, 0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=[c / 255 for c in TIER_COLORS[t][:3]], edgecolor="none")
        for t in tier_order
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none",
                                  edgecolor=[c / 255 for c in SINK_BORDER_COLOR[:3]], linewidth=2))
    fig.legend(
        handles,
        [LEGEND_LABELS[t] for t in tier_order] + ["sink"],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.02),
    )
    # fig.suptitle("PruneSID token selection + sink-aware bit-tier assignment across budgets", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    fig.savefig(PAPER_FIGURE_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(PAPER_FIGURE_PATH.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {PAPER_FIGURE_PATH}")


if __name__ == "__main__":
    main()
