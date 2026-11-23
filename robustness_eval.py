"""Robustness of pretrained face-verification models to image degradations (LFW pairs).
Models: facenet-pytorch InceptionResnetV1 pretrained on VGGFace2 and on CASIA-WebFace.
Protocol: standard LFW 10-fold verification pairs; the SAME degradation is applied to both images of a pair;
threshold chosen on 9 folds and tested on the 10th; no labels are used at inference.
Run: python robustness_eval.py   (GPU recommended; ~15 min on a Colab T4)."""
import io, json, time, numpy as np, torch
from PIL import Image, ImageFilter
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score
from facenet_pytorch import InceptionResnetV1

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = fetch_lfw_pairs(subset="10_folds", color=True, resize=1.0)
P = d.pairs; y = d.target.astype(int)
if P.max() <= 1.5: P = P * 255.0
P = np.clip(P, 0, 255).astype(np.uint8)          # (6000, 2, H, W, 3)
N = len(y); print("pairs", P.shape, "same-rate", y.mean(), flush=True)

def blur(im, s):  return im.filter(ImageFilter.GaussianBlur(radius=s))
def noise(im, s, rng):
    a = np.asarray(im).astype(np.float32); a += rng.normal(0, s, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def lowres(im, f):
    w, h = im.size; return im.resize((max(2, w // f), max(2, h // f)), Image.BICUBIC).resize((w, h), Image.BICUBIC)
def mask(im, frac):                               # occlude the lower part of the face (mask-like)
    a = np.asarray(im).copy(); h = a.shape[0]; a[int(h * (1 - frac)):] = 0; return Image.fromarray(a)

SEV = {
    "gaussian_blur_radius": [1, 2, 3, 4, 6],
    "gaussian_noise_std":   [10, 20, 30, 45, 60],
    "jpeg_quality":         [50, 30, 15, 8, 3],
    "downsample_factor":    [2, 3, 4, 6, 8],
    "lower_occlusion_frac": [0.1, 0.2, 0.3, 0.4, 0.5],
}
def apply(kind, level, im, rng):
    if kind == "gaussian_blur_radius": return blur(im, level)
    if kind == "gaussian_noise_std":   return noise(im, level, rng)
    if kind == "jpeg_quality":         return jpeg(im, level)
    if kind == "downsample_factor":    return lowres(im, level)
    if kind == "lower_occlusion_frac": return mask(im, level)
    return im

@torch.no_grad()
def embed_all(model, kind=None, level=None, bs=200):
    rng = np.random.default_rng(0)
    out = np.zeros((N, 2, 512), np.float32)
    for side in (0, 1):
        for s in range(0, N, bs):
            ims = []
            for a in P[s:s + bs, side]:
                im = Image.fromarray(a)
                if kind: im = apply(kind, level, im, rng)
                ims.append(np.asarray(im.resize((160, 160), Image.BILINEAR)))
            x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev)
            x = (x - 127.5) / 128.0
            e = torch.nn.functional.normalize(model(x), dim=1)
            out[s:s + bs, side] = e.cpu().numpy()
    return out

def verify(E):
    sim = (E[:, 0] * E[:, 1]).sum(1)
    folds = np.arange(N) // (N // 10); accs = []
    for k in range(10):
        tr, te = folds != k, folds == k
        ths = np.unique(sim[tr]); best = max(ths[:: max(1, len(ths) // 400)], key=lambda t: ((sim[tr] > t) == y[tr]).mean())
        accs.append(((sim[te] > best) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs)), float(roc_auc_score(y, sim))

res = {}
for name in ("vggface2", "casia-webface"):
    m = InceptionResnetV1(pretrained=name).eval().to(dev); t0 = time.time()
    res[name] = {"clean": dict(zip(("acc", "acc_std", "auc"), verify(embed_all(m))))}
    print(name, "clean", res[name]["clean"], flush=True)
    for kind, levels in SEV.items():
        res[name][kind] = {}
        for lv in levels:
            a, s, u = verify(embed_all(m, kind, lv)); res[name][kind][str(lv)] = {"acc": a, "acc_std": s, "auc": u}
            print(name, kind, lv, round(a, 4), round(u, 4), flush=True)
    print(name, "minutes", round((time.time() - t0) / 60, 1), flush=True)
json.dump(res, open("results.json", "w"), indent=1)
print("RESULT_JSON", json.dumps(res))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.6), sharey=True)
    for i, (kind, levels) in enumerate(SEV.items()):
        for name, c in (("vggface2", "#1f77b4"), ("casia-webface", "#d62728")):
            ys = [res[name]["clean"]["acc"]] + [res[name][kind][str(l)]["acc"] for l in levels]
            ax[i].plot(range(len(ys)), ys, marker="o", color=c, label=name)
        ax[i].set_title(kind); ax[i].set_xticks(range(len(levels) + 1)); ax[i].set_xticklabels(["clean"] + [str(l) for l in levels], fontsize=7); ax[i].grid(alpha=.3)
    ax[0].set_ylabel("LFW 10-fold accuracy"); ax[0].legend(); plt.tight_layout(); plt.savefig("robustness.png", dpi=130)
except Exception as e: print("plot failed", e)
