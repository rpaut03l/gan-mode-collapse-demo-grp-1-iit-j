# Deep Learning :: Generative Adversarial Networks (GANs) — PRACTICE

> Part 3 of 3 — **Practice / Code** | [Theory](./DL_GANs_Theory.md) | [Numericals](./DL_GANs_Numericals.md)

<a id="top"></a>

---

## Table of Contents

1. [Tools & tech stack (the toolbox)](#tools)
2. [Dependencies & environment setup](#deps)
3. [Library cheatsheet (what each import does)](#libs)
4. [Code 1 — Simple GAN on MNIST (every line explained)](#simple-gan)
5. [Code 2 — DCGAN (convolutional, every line explained)](#dcgan)
6. [Code 3 — the training tricks, as code](#tricks)
7. [Logging with W&B](#wandb)
8. [Debugging cheatsheet (it broke, now what)](#debug)
9. [Training-recipe cheatsheet](#recipe)
10. [Exam/viva hacks for the lab](#hacks)

[Skip to bottom](#bottom)

---

<a id="tools"></a>
## 1. Tools & tech stack (the toolbox)

```
  LAYER            TOOL                  WHY
  ----------------------------------------------------------------
  framework        PyTorch               clean autograd, GAN-friendly
  (alt)            TensorFlow/Keras      also fine; tf.GradientTape
  data/vision      torchvision           datasets (MNIST), transforms
  tensors/math     numpy                 cpu math, plotting prep
  plotting         matplotlib            show generated grids
  progress         tqdm                  training bars
  experiment log   Weights & Biases      track loss curves, image samples
  compute          Kaggle T4 / Colab     free GPU
  pretrained GANs  HuggingFace, torch.hub StyleGAN/BigGAN demos
  metrics          torchmetrics / pytorch-fid   FID, IS out of the box
```
> Your stack: **PyTorch on Kaggle T4**, log to **W&B** (`rohitspatel0008`), share notebook to your Kaggle (`rohitpatel0008`). Single self-contained cell preferred.

[Back to top](#top)

---

<a id="deps"></a>
## 2. Dependencies & environment setup

```bash
# CPU/GPU PyTorch (Kaggle/Colab already have it; locally:)
pip install torch torchvision

# extras
pip install numpy matplotlib tqdm
pip install wandb               # experiment logging (optional)
pip install pytorch-fid         # FID metric (optional)
```

Versions that play nicely (any recent works; pin if reproducing):
```
  python >= 3.9
  torch   >= 2.0
  torchvision >= 0.15
  numpy   >= 1.24
  matplotlib >= 3.7
```

Quick GPU check (run first, always):
```python
import torch
print(torch.__version__)                 # which torch
print(torch.cuda.is_available())         # True on Kaggle T4 / Colab GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)
```
- `torch.cuda.is_available()` → tells you if a GPU is wired up. If `False` on Kaggle, switch the accelerator to **GPU T4** in settings.
- `device` string is reused everywhere to send tensors/models to the GPU.

[Back to top](#top)

---

<a id="libs"></a>
## 3. Library cheatsheet (what each import does)

```python
import torch                      # core tensors + autograd (the engine)
import torch.nn as nn             # layers: Linear, Conv2d, BatchNorm, activations
import torch.optim as optim       # optimisers: Adam, SGD
from torch.utils.data import DataLoader   # batches + shuffling
import torchvision                # vision datasets & utilities
import torchvision.transforms as T        # image preprocessing pipeline
from torchvision.utils import make_grid   # tile many images into one grid
import matplotlib.pyplot as plt   # display the grid
```

| Import | One-liner |
|---|---|
| `torch` | tensors, GPU, automatic differentiation (`.backward()`) |
| `nn` | building blocks of networks; `nn.Module` is the base class |
| `optim` | gradient-descent algorithms (we use `Adam`) |
| `DataLoader` | feeds data in shuffled minibatches |
| `torchvision.datasets` | ready MNIST/CIFAR downloads |
| `transforms (T)` | `ToTensor`, `Normalize`, resize, etc. |
| `make_grid` | stitches a batch of images into one picture to view |
| `matplotlib` | renders that picture |

[Back to top](#top)

---

<a id="simple-gan"></a>
## 4. Code 1 — Simple GAN on MNIST (every line explained)

A fully self-contained, single-cell GAN (MLP generator + MLP discriminator) on 28×28 MNIST digits. Read the inline comments — every line has a job.

```python
# ============ SIMPLE GAN ON MNIST (single cell) ============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# ---- 0. config & device ----
device   = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if present
z_dim    = 64        # length of the noise vector z (G's input)
img_dim  = 28*28     # MNIST flattened = 784 pixels (G's output, D's input)
batch    = 128       # samples per minibatch
lr       = 3e-4      # learning rate (Adam)
epochs   = 50        # passes over the dataset
torch.manual_seed(0) # reproducibility: same random numbers every run

# ---- 1. data ----
# Normalize to [-1,1] because the Generator ends in Tanh (also [-1,1]); they must match.
transform = T.Compose([
    T.ToTensor(),                 # PIL image -> tensor, scales pixels to [0,1]
    T.Normalize((0.5,), (0.5,)),  # (x-0.5)/0.5 -> maps [0,1] to [-1,1]
])
dataset = torchvision.datasets.MNIST(
    root=".", train=True, transform=transform, download=True)  # downloads MNIST once
loader  = DataLoader(dataset, batch_size=batch, shuffle=True)   # shuffled minibatches

# ---- 2. the two networks ----
class Generator(nn.Module):           # G: noise z -> fake image
    def __init__(self):
        super().__init__()            # required: init the nn.Module machinery
        self.net = nn.Sequential(     # a stack of layers run in order
            nn.Linear(z_dim, 256),    # z(64) -> 256 hidden units
            nn.LeakyReLU(0.1),        # activation; leak keeps small negative gradient
            nn.Linear(256, 512),      # widen
            nn.LeakyReLU(0.1),
            nn.Linear(512, img_dim),  # 512 -> 784 pixels
            nn.Tanh(),                # squashes output to [-1,1] (matches data range)
        )
    def forward(self, z):             # how data flows through G
        return self.net(z)            # returns a fake flattened image

class Discriminator(nn.Module):       # D: image -> probability it's real
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 256),  # 784 pixels -> 256
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),        # 256 -> single score
            nn.Sigmoid(),             # squashes score to [0,1] = P(real)
        )
    def forward(self, x):
        return self.net(x)            # returns probability in [0,1]

G = Generator().to(device)            # move model weights onto GPU/CPU
D = Discriminator().to(device)

# ---- 3. loss & optimisers ----
criterion = nn.BCELoss()              # Binary Cross-Entropy: matches GAN's log-loss
opt_G = optim.Adam(G.parameters(), lr=lr)  # updates ONLY G's weights
opt_D = optim.Adam(D.parameters(), lr=lr)  # updates ONLY D's weights
fixed_noise = torch.randn(64, z_dim, device=device)  # frozen z to watch G improve

# ---- 4. training loop ----
for epoch in range(epochs):
    for real, _ in loader:                       # _ = labels, we don't need them
        real = real.view(-1, img_dim).to(device) # flatten 28x28 -> 784, to device
        bs   = real.size(0)                       # actual batch size (last batch smaller)

        # ----- (A) train Discriminator: maximise log D(real) + log(1 - D(fake)) -----
        z     = torch.randn(bs, z_dim, device=device)  # fresh random noise
        fake  = G(z)                                    # G makes fakes
        D_real = D(real).view(-1)                       # D's score on reals
        D_fake = D(fake.detach()).view(-1)              # .detach(): don't backprop into G here
        # BCE with label 1 for reals, 0 for fakes:
        lossD_real = criterion(D_real, torch.ones_like(D_real))   # push reals -> 1
        lossD_fake = criterion(D_fake, torch.zeros_like(D_fake))  # push fakes -> 0
        lossD = (lossD_real + lossD_fake) / 2           # average the two
        opt_D.zero_grad()                               # clear old gradients
        lossD.backward()                                # compute new gradients
        opt_D.step()                                    # update D's weights

        # ----- (B) train Generator: maximise log D(fake)  (non-saturating) -----
        output = D(fake).view(-1)                       # re-score the SAME fakes (no detach)
        lossG  = criterion(output, torch.ones_like(output))  # G wants D to say "real"(1)
        opt_G.zero_grad()
        lossG.backward()                                # gradients flow through D into G
        opt_G.step()                                    # update ONLY G (opt_G touches G params)

    # ---- 5. peek at progress each epoch ----
    print(f"epoch {epoch+1}/{epochs}  lossD {lossD.item():.3f}  lossG {lossG.item():.3f}")

# ---- 6. show final samples ----
with torch.no_grad():                                # no gradients needed for viewing
    samples = G(fixed_noise).view(-1, 1, 28, 28)     # reshape flat -> image
    grid = make_grid(samples, normalize=True)        # tile 64 images, rescale for display
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0).cpu())            # CHW -> HWC for matplotlib, to cpu
    plt.axis("off"); plt.show()
```

**The 6 lines that trip everyone up:**
- `fake.detach()` (D step): freezes G so D's update doesn't change G. **Forget this → G gets wrong gradients.**
- No detach in G step: gradients must flow *through* D into G. (D's weights aren't updated because `opt_G` only holds G's params.)
- `opt_*.zero_grad()` **before** each `backward()`: PyTorch *accumulates* gradients; skip this and they pile up.
- `torch.ones_like` / `zeros_like`: the labels 1 (real) and 0 (fake) for BCE.
- `T.Normalize((0.5,),(0.5,))` + `Tanh`: both put data in `[-1,1]`. **Mismatch = garbage / never converges.**
- `torch.no_grad()` at sampling: saves memory, no graph built.

[Back to top](#top)

---

<a id="dcgan"></a>
## 5. Code 2 — DCGAN (convolutional, every line explained)

Convolutions beat MLPs on images. This is the DCGAN recipe (`SBR-LRT-A` from the theory mnemonic). Shown for 64×64 single-channel images.

```python
# ============ DCGAN core networks ============
import torch
import torch.nn as nn

# ---- Generator: noise(z_dim x 1 x 1) -> image(channels x 64 x 64) ----
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, g_feat=64, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # block: ConvTranspose2d UPSAMPLES (doubles H,W each time)
            # args: (in_ch, out_ch, kernel=4, stride=2, padding=1)
            self._block(z_dim, g_feat*8, 4, 1, 0),   # z -> 4x4    (special: stride1,pad0)
            self._block(g_feat*8, g_feat*4, 4, 2, 1), # 4 -> 8
            self._block(g_feat*4, g_feat*2, 4, 2, 1), # 8 -> 16
            self._block(g_feat*2, g_feat,   4, 2, 1), # 16 -> 32
            nn.ConvTranspose2d(g_feat, channels, 4, 2, 1),  # 32 -> 64 (final, no BN)
            nn.Tanh(),                                 # pixels -> [-1,1]
        )
    def _block(self, i, o, k, s, p):                  # reusable upsample block
        return nn.Sequential(
            nn.ConvTranspose2d(i, o, k, s, p, bias=False), # bias=False: BN handles bias
            nn.BatchNorm2d(o),                             # stabilise training (the 'B')
            nn.ReLU(),                                     # ReLU in G (the 'R')
        )
    def forward(self, z):                              # z shape: (batch, z_dim, 1, 1)
        return self.net(z)

# ---- Discriminator: image(channels x 64 x 64) -> score ----
class DCDiscriminator(nn.Module):
    def __init__(self, channels=1, d_feat=64):
        super().__init__()
        self.net = nn.Sequential(
            # first layer: NO BatchNorm (DCGAN rule)
            nn.Conv2d(channels, d_feat, 4, 2, 1),      # 64 -> 32
            nn.LeakyReLU(0.2),                         # LeakyReLU in D (the 'L')
            self._block(d_feat,   d_feat*2, 4, 2, 1),  # 32 -> 16
            self._block(d_feat*2, d_feat*4, 4, 2, 1),  # 16 -> 8
            self._block(d_feat*4, d_feat*8, 4, 2, 1),  # 8 -> 4
            nn.Conv2d(d_feat*8, 1, 4, 2, 0),           # 4 -> 1 (single score)
            nn.Sigmoid(),                              # -> P(real) in [0,1]
        )
    def _block(self, i, o, k, s, p):                   # reusable downsample block
        return nn.Sequential(
            nn.Conv2d(i, o, k, s, p, bias=False),
            nn.BatchNorm2d(o),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.net(x)

# ---- weight init recommended by DCGAN paper (mean 0, std 0.02) ----
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)      # the famous N(0, 0.02) init

# ---- usage ----
device = "cuda" if torch.cuda.is_available() else "cpu"
G = DCGenerator().to(device);  G.apply(init_weights)
D = DCDiscriminator().to(device); D.apply(init_weights)

# DCGAN optimiser settings (memorise): Adam, lr=2e-4, betas=(0.5, 0.999)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# noise has shape (batch, z_dim, 1, 1) for ConvTranspose:
z = torch.randn(128, 100, 1, 1, device=device)
fake = G(z)               # -> (128, 1, 64, 64)
score = D(fake)           # -> (128, 1, 1, 1)
```

**DCGAN rules visible in the code (`SBR-LRT-A`):**
- **S**trided convs replace pooling — `stride=2` everywhere does the resizing.
- **B**atchNorm in both nets (except D's first layer & G's last layer).
- **R**emove fully-connected hidden layers — it's all conv.
- **L**eakyReLU in D, **R**eLU in G, **T**anh on G's output.
- **A**dam with `lr=2e-4, beta1=0.5`.

The training loop is **identical** to the simple GAN (Code 1, section A & B) — only the network classes change. Swap in `DCGenerator`/`DCDiscriminator`, make noise shape `(bs,100,1,1)`, and resize MNIST to 64×64 in the transform.

[Back to top](#top)

---

<a id="tricks"></a>
## 6. Code 3 — the training tricks, as code

```python
# (1) one-sided label smoothing: use 0.9 instead of 1.0 for "real" -> D less over-confident
lossD_real = criterion(D_real, torch.full_like(D_real, 0.9))

# (2) add noise to D's inputs (instance noise) -> smoother training early on
real_noisy = real + 0.05 * torch.randn_like(real)

# (3) train D k times per G step (here k=1; raise if D is too weak)
K = 1
for _ in range(K):
    ... # D update

# (4) TTUR: different learning rates (D faster than G)
opt_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.5,0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.999))

# (5) WGAN-GP gradient penalty (sketch) -> enforces 1-Lipschitz critic
def gradient_penalty(D, real, fake, device):
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)   # random mix coeff
    mixed = (eps*real + (1-eps)*fake).requires_grad_(True)    # point between real & fake
    score = D(mixed)
    grad  = torch.autograd.grad(outputs=score, inputs=mixed,
                                grad_outputs=torch.ones_like(score),
                                create_graph=True, retain_graph=True)[0]
    grad  = grad.view(grad.size(0), -1)
    gp    = ((grad.norm(2, dim=1) - 1)**2).mean()             # push ||grad|| toward 1
    return gp
# then:  lossD = lossD_fake - lossD_real + LAMBDA * gp     (LAMBDA = 10)
```

[Back to top](#top)

---

<a id="wandb"></a>
## 7. Logging with W&B

```python
import wandb
wandb.init(project="gan-mnist", entity="rohitspatel0008")  # your W&B org

wandb.log({"lossD": lossD.item(), "lossG": lossG.item(), "epoch": epoch})  # curves

# log a sample grid as an image every few epochs
if epoch % 5 == 0:
    with torch.no_grad():
        grid = make_grid(G(fixed_noise).view(-1,1,28,28), normalize=True)
    wandb.log({"samples": wandb.Image(grid)})
```
- Watch `lossD` and `lossG` **trade places** — healthy GANs oscillate, they don't flatline.
- If `lossD → 0` and `lossG → ∞`: D won, G starved → lower D's LR or train G more.

[Back to top](#top)

---

<a id="debug"></a>
## 8. Debugging cheatsheet (it broke, now what)

```
SYMPTOM                          LIKELY CAUSE                  FIX
-----------------------------------------------------------------------------
all outputs look identical       MODE COLLAPSE                 WGAN-GP, minibatch
                                                               discrimination, lower G lr
G loss explodes, D loss -> 0     D too strong (vanishing grad) non-sat loss (already),
                                                               train G more, lower D lr
nothing learns / noise stays     forgot fake.detach() in D     add .detach() in D step
                                 step OR data not in [-1,1]    match Normalize + Tanh
NaN losses                       lr too high / bad init        lr=2e-4, N(0,0.02) init,
                                                               check for /0 in custom loss
gradients zero in G              detached fake reused in G     re-run D(fake) WITHOUT detach
images blurry not sharp          MLP on images                use DCGAN (convs)
loss looks great, images bad     loss != image quality         judge by eyes + FID, not loss
CUDA out of memory               batch too big                 lower batch, use AMP
```

> **Golden rule:** in a GAN, **loss numbers lie.** Always look at generated images and/or FID. A "nice" loss curve can hide mode collapse.

[Back to top](#top)

---

<a id="recipe"></a>
## 9. Training-recipe cheatsheet

```
  data         normalise to [-1,1]; G ends in Tanh
  noise z      N(0,1), dim 64 (MLP) or 100 (DCGAN), shape (bs,zdim,1,1) for conv
  optimiser    Adam, lr=2e-4, betas=(0.5,0.999)
  batch        64-128
  init         N(0, 0.02) for conv/BN (DCGAN)
  loss         non-saturating BCE  (or WGAN-GP for stability)
  arch         DCGAN: strided/transposed conv, BN, LeakyReLU(D)/ReLU(G), Tanh out
  ratio        k=1 D-step per G-step (raise k if D too weak)
  watch        sample grid + FID, NOT the loss number
  stop         when D(real)~D(fake)~0.5 and samples look real
```

[Back to top](#top)

---

<a id="hacks"></a>
## 10. Exam/viva hacks for the lab

1. **"Why `.detach()`?"** → to update D without backpropagating into G; keeps the two updates independent.
2. **"Why Tanh + Normalize(0.5,0.5)?"** → both force `[-1,1]`; ranges must match or training fails.
3. **"Why Adam β1=0.5?"** → DCGAN found the default 0.9 momentum too aggressive → oscillation; 0.5 stabilises.
4. **"Why LeakyReLU in D?"** → avoids dead neurons / zero gradients so the critic keeps learning.
5. **"Why no BN on D's first / G's last layer?"** → keeps input/output statistics intact (DCGAN finding).
6. **"How do you know it converged?"** → samples look real, `D≈0.5` on both, FID stops dropping.
7. **"Loss is low but images bad — explain"** → GAN loss is a *game score*, not a quality measure; use FID/eyes.
8. **Show the loop verbally**: sample noise → fake → train D (real=1, fake=0, detach) → train G (fake=1) → repeat.
9. **Cross-reference your real pipeline** — your production MLOps repo packages models the same way (Docker/K8s deploy): [rptl_gn_mlops](https://github.com/rpaut03l/rptl_gn_mlops/tree/mlops-pipeline). Good "I've shipped this" line in a viva.

[Back to top](#top)

<a id="bottom"></a>

---
[Back to top](#top) | [<- Theory](./DL_GANs_Theory.md) | [<- Numericals](./DL_GANs_Numericals.md)
