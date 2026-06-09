# Deep Learning :: Generative Adversarial Networks (GANs) — THEORY

> Subject: Deep Learning (F3 — Generative Models) | Text: Goodfellow et al., *Deep Learning*, Ch. 20 | Original paper: Goodfellow et al., 2014
> Part 1 of 3 — **Theory** | [Numericals](./DL_GANs_Numericals.md) | [Practice](./DL_GANs_Practice.md)

<a id="top"></a>

---

## Table of Contents

1. [The 30-second story (ELI5)](#eli5)
2. [What problem are we even solving?](#problem)
3. [The two players: G and D](#players)
4. [Notation — every symbol used](#notation)
5. [The architecture (big ASCII map)](#architecture)
6. [The adversarial game (the minimax)](#minimax)
7. [The two loss functions you must know](#losses)
8. [The optimal discriminator (full derivation)](#optimal-d)
9. [Why it secretly minimises JS divergence](#jsd)
10. [The training algorithm (step by step)](#algorithm)
11. [What "convergence" means (Nash equilibrium)](#nash)
12. [What goes wrong (the 4 demons)](#demons)
13. [The GAN zoo (variants you must name)](#zoo)
14. [How do we score a GAN? (IS, FID)](#metrics)
15. [Mnemonics block](#mnemonics)
16. [Cheatsheet (one screen)](#cheatsheet)
17. [Exam hacks](#exam-hacks)
18. [Where to go next](#links)

[Skip to bottom](#bottom)

---

<a id="eli5"></a>
## 1. The 30-second story (ELI5)

Imagine two kids.

- **Robby the Forger** wants to paint fake money. At first his "money" looks like a potato.
- **Dia the Detective** has a stack of *real* money and Robby's *fakes* all mixed together. Her only job: point at each note and shout "REAL!" or "FAKE!"

Every round:
- Dia gets better at *catching* Robby (she learns what fakes look like).
- Robby looks at *what gave him away* and fixes it, so his next batch fools her a little more.

They keep fighting. After thousands of rounds, Robby's fakes are SO good that Dia can only guess — she's right 50% of the time, like flipping a coin. **At that point Robby is a master forger: his fake money is indistinguishable from real money.**

That's a GAN.

```
   noise -->  [ Robby = Generator G ]  --> fake image
                                              \
   real images ----------------------------->  [ Dia = Discriminator D ] --> "real or fake?"
                                              /
                                  feedback flows BACKWARD to make both smarter
```

**Generator = forger. Discriminator = detective. They train by competing.** That single sentence is 80% of the exam.

[Back to top](#top)

---

<a id="problem"></a>
## 2. What problem are we even solving?

We want a machine that can **generate brand-new data** that looks like our training data — new faces, new handwriting, new songs.

Formally: we have real data drawn from an unknown distribution `p_data(x)`. We want to learn a model distribution `p_g(x)` so that `p_g ≈ p_data`. Then we can **sample** from `p_g` to make new things.

Old way (explicit density): write down a formula for `p_g(x)` and maximise likelihood. Hard for images (the formula is monstrous).

**GAN way (implicit density):** never write the formula. Instead, learn a *function G* that turns easy random noise `z` into a sample. We only need to be able to *draw samples*, not evaluate a probability. We judge quality with a second network instead of a likelihood number.

```
generative-model family tree

                       Generative models
                      /                  \
         Explicit density            Implicit density
        (you can compute p(x))      (you can only sample)
          /         \                       |
   Tractable    Approximate              **GAN**  <-- you are here
   (PixelRNN,   (VAE: uses a
    Autoreg.)    lower bound)
```

**Key idea to memorise:** GANs are *implicit, likelihood-free* generative models. They sidestep the intractable likelihood by replacing it with a learned critic (D).

[Back to top](#top)

---

<a id="players"></a>
## 3. The two players: G and D

| Player | Name | Input | Output | Wants |
|---|---|---|---|---|
| `G` | Generator | noise vector `z` | a fake sample `G(z)` | D to say "real" for its fakes |
| `D` | Discriminator | a sample `x` (real or fake) | scalar in `[0,1]` = P(real) | label reals 1, fakes 0 |

- `G` is a **counterfeiter** turning random scribble into believable data.
- `D` is a **binary classifier** (a normal neural net ending in sigmoid).
- They have **opposite goals** → "adversarial".

`G` never sees real data directly. It learns *only* through D's gradient — like learning to cook for someone you never meet, hearing only "too salty / perfect" through a wall.

[Back to top](#top)

---

<a id="notation"></a>
## 4. Notation — every symbol used

| Symbol | Meaning |
|---|---|
| `x` | a real data sample (e.g. an image) |
| `z` | latent **noise** vector (the "seed"), usually `z ~ N(0, I)` or `U(-1,1)` |
| `p_data(x)` | true data distribution (unknown, we only have samples) |
| `p_z(z)` | prior on the noise (we choose it, e.g. Gaussian) |
| `p_g(x)` | distribution of samples produced by G (implicit) |
| `G(z; θ_g)` | generator network with parameters `θ_g` |
| `D(x; θ_d)` | discriminator network with parameters `θ_d`, outputs P(real) |
| `V(D, G)` | value function (the thing G and D fight over) |
| `E[.]` | expectation (average over the distribution) |
| `θ_g, θ_d` | weights of G and D |
| `m` | minibatch size |
| `α` (or `η`) | learning rate |
| `k` | # of D updates per 1 G update (often `k=1`) |
| `JSD` | Jensen–Shannon divergence |
| `KL` | Kullback–Leibler divergence |
| `∇` | gradient (vector of partial derivatives) |
| `λ` | penalty weight (e.g. gradient penalty in WGAN-GP) |

> Read `x ~ p_data` as "x is sampled from p_data". Read `E_{x~p_data}[f(x)]` as "average of f(x) when x comes from real data".

[Back to top](#top)

---

<a id="architecture"></a>
## 5. The architecture (big ASCII map)

```
                          THE GAN LOOP
   .......................................................................

   p_z(z)            +-------------------+
   random noise ---> |   GENERATOR  G    | ---> G(z)  (fake sample)
   z ~ N(0, I)       |  (deconv / MLP)   |          \
                     +-------------------+           \
                            ^                          \
                            | gradient to fool D        v
                            |                     +-----------------+
                            |                     |                 |
   p_data(x)                |   real x ---------> | DISCRIMINATOR D | --> D(.) in [0,1]
   real samples ------------|-------------------> |  (conv / MLP +  |       |
                            |                     |    sigmoid)     |       |
                            |                     +-----------------+       |
                            |                            ^                  |
                            |                            |                  v
                            |                       gradient to     loss (BCE):
                            |                       classify better  reals->1, fakes->0
                            |                            |
                            +-------- backprop ----------+

   .......................................................................
   Two optimisers: one updates only theta_d (D), one updates only theta_g (G).
   When D processes a FAKE during G's update, gradients pass THROUGH D into G,
   but D's weights are NOT changed in that step.
```

Inside each block (typical DCGAN shapes for 64x64 images):

```
GENERATOR  (noise -> image): "grow the picture"
  z[100] -> ConvT -> 4x4x512 -> ConvT -> 8x8x256 -> ConvT -> 16x16x128
         -> ConvT -> 32x32x64 -> ConvT -> 64x64x3 -> Tanh   (pixels in [-1,1])

DISCRIMINATOR (image -> score): "shrink the picture"
  64x64x3 -> Conv -> 32x32x64 -> Conv -> 16x16x128 -> Conv -> 8x8x256
          -> Conv -> 4x4x512 -> Conv -> 1 -> Sigmoid   (prob it is real)
```

Notice the mirror symmetry: **G upsamples, D downsamples.** Transposed-conv grows; strided-conv shrinks.

[Back to top](#top)

---

<a id="minimax"></a>
## 6. The adversarial game (the minimax)

Both nets share **one** scoreboard, the **value function**:

```
  min  max  V(D, G) =  E_{x~p_data}[ log D(x) ]  +  E_{z~p_z}[ log(1 - D(G(z))) ]
   G    D
                       \_____ D wants this BIG _____/   \____ D wants this BIG too ___/
                       (call reals real)                (call fakes fake)
```

Read it as a tug-of-war over the same number `V`:

- **D maximises V**: push `D(x)→1` on reals (first term big) and `D(G(z))→0` on fakes (second term big).
- **G minimises V**: it can only touch the second term, and it wants `D(G(z))→1`, which makes `log(1 - D(G(z)))` very negative → V small.

```
   D pulls V up   <===========[ V ]===========>   G pulls V down
   (better cop)                                    (better forger)
```

> One-liner for the exam: **G and D play a two-player zero-sum minimax game on the value function V(D,G).**

[Back to top](#top)

---

<a id="losses"></a>
## 7. The two loss functions you must know

Everything reduces to **Binary Cross-Entropy (BCE)** with labels: real = 1, fake = 0.

**(a) Discriminator loss** — maximise V, i.e. minimise `-V`:

```
  L_D = - E_{x~p_data}[ log D(x) ]  -  E_{z~p_z}[ log(1 - D(G(z))) ]
        \___ reals should score 1 ___/   \___ fakes should score 0 ___/
```

**(b) Generator loss — two flavours:**

*Saturating (the original minimax form):*
```
  L_G(minimax) = E_{z}[ log(1 - D(G(z))) ]      (G minimises this)
```
Problem: early on, fakes are obvious, `D(G(z))≈0`, so `log(1-0)=0` and its gradient is almost flat → **G barely learns** ("vanishing gradient" / saturation).

*Non-saturating (what everyone actually uses):*
```
  L_G(NS) = - E_{z}[ log D(G(z)) ]              (G minimises this)
```
Same fixed point, but **strong gradients when fakes are bad** — exactly when G needs to learn most.

```
   gradient strength for G when D is confident fakes are fake:

   saturating  : ~ 0   (flat, useless)   -----______________
   non-sat     : large (steep, useful)   \
                                           \___________
```

> Exam line: *"We flip `min log(1-D(G(z)))` into `max log D(G(z))` to avoid early-training gradient saturation."* — Goodfellow's "non-saturating" trick.

[Back to top](#top)

---

<a id="optimal-d"></a>
## 8. The optimal discriminator (full derivation)

**Goal:** with G fixed, find the best D.

Write V as integrals (turn expectation into integral over x; use change of variables so the noise term becomes an integral over G's output distribution `p_g`):

```
  V(D,G) = INT p_data(x) log D(x) dx  +  INT p_g(x) log(1 - D(x)) dx
         = INT [ p_data(x) log D(x) + p_g(x) log(1 - D(x)) ] dx
```

For each fixed `x`, maximise the integrand. Let `a = p_data(x)`, `b = p_g(x)`, `y = D(x)`:

```
  f(y) = a log y + b log(1 - y)
  f'(y) = a/y  -  b/(1 - y)  = 0
  =>  a(1 - y) = b y
  =>  y = a / (a + b)
```

So the **optimal discriminator** is:

```
   D*(x) =        p_data(x)
           ------------------------
           p_data(x)  +  p_g(x)
```

Sanity check: where G already matches data (`p_g = p_data`), `D*(x) = 1/2` — D is forced to **guess**. That 0.5 is the famous "GAN has converged" signal.

[Back to top](#top)

---

<a id="jsd"></a>
## 9. Why it secretly minimises JS divergence

Plug `D*` back into V. After algebra:

```
  C(G) = V(D*, G)
       = - log 4  +  KL( p_data || (p_data+p_g)/2 )  +  KL( p_g || (p_data+p_g)/2 )
       = - log 4  +  2 * JSD( p_data || p_g )
```

Because `JSD ≥ 0` and `= 0` only when `p_data = p_g`:

```
   minimum of C(G) = - log 4 ≈ -1.386,  achieved iff  p_g = p_data
```

**Meaning:** training G (against the optimal D) is the same as **pushing the Jensen–Shannon divergence between the fake and real distributions to zero.** The global optimum is when the generator perfectly copies the data distribution.

```
  JSD measures "how far apart are the two distributions"
     far apart  ->  big JSD  ->  big loss  ->  keep training
     identical  ->  JSD = 0   ->  loss = -log4  ->  done
```

> This is WHY GANs work and the most-tested theory result. Memorise: **optimal value = -log 4**, and **GAN ⇔ minimise JSD**.

[Back to top](#top)

---

<a id="algorithm"></a>
## 10. The training algorithm (step by step)

This is **Algorithm 1** from the 2014 paper. Learn the shape, not the wording.

```
for each training iteration:

  # ---- (A) update D for k steps (k often = 1) ----
  for k steps:
      1. sample minibatch of m noise vectors {z_1..z_m} from p_z
      2. sample minibatch of m real samples  {x_1..x_m} from p_data
      3. compute fakes:  G(z_i)
      4. ASCEND D's gradient (improve the cop):
         grad_theta_d  (1/m) * SUM [ log D(x_i) + log(1 - D(G(z_i))) ]
      5. theta_d <- theta_d + alpha * grad   (gradient ASCENT, maximise V)

  # ---- (B) update G once ----
      6. sample minibatch of m noise vectors {z_1..z_m} from p_z
      7. DESCEND G's gradient (improve the forger):
         non-saturating:  minimise  -(1/m) * SUM log D(G(z_i))
      8. theta_g <- theta_g - alpha * grad   (gradient DESCENT)

repeat until D(real) ~ D(fake) ~ 0.5
```

Order matters: **D first, G second** (G needs a halfway-decent critic to learn from). In code you do D-ascent and G-descent (most frameworks just minimise both `L_D` and `L_G`, which is equivalent).

```
  one iteration:   [ train D ] -> [ train G ] -> repeat
                     learn what    learn to
                     fakes look     fool the
                     like           updated D
```

[Back to top](#top)

---

<a id="nash"></a>
## 11. What "convergence" means (Nash equilibrium)

GAN training is **not** plain minimisation — it's finding a **saddle point / Nash equilibrium** of a two-player game.

```
  Normal NN:  roll DOWNHILL to the valley bottom (one loss, one minimum).

  GAN:        balance on a SADDLE.
              along G's direction it's a min,
              along D's direction it's a max.

                  D-axis (max)
                     ^
                     |     . saddle point = equilibrium
              -------+--------
                     |
                  G-axis (min)
```

At equilibrium: `p_g = p_data`, `D*(x) = 1/2` everywhere, `V = -log 4`. Neither player can improve by changing only their own weights → **Nash equilibrium**. This is why GANs are *unstable*: you're not descending, you're balancing.

[Back to top](#top)

---

<a id="demons"></a>
## 12. What goes wrong (the 4 demons)

| Demon | What happens | Why | Fix |
|---|---|---|---|
| **Mode collapse** | G outputs the same few samples (e.g. only "3"s on MNIST) | G found one image that always fools D and keeps making it | minibatch discrimination, **unrolled GAN**, **WGAN**, **PacGAN**, feature matching |
| **Vanishing gradient** | G stops learning | D got *too good*, `D(G(z))≈0`, saturating loss flatlines | **non-saturating loss**, **WGAN** (no log) |
| **Non-convergence / oscillation** | losses bounce forever, never settle | it's a game, not a descent; players chase each other | **TTUR** (diff. LRs), lower LR, label smoothing |
| **Instability / imbalance** | one net crushes the other | D learns faster than G (or vice versa) | balance update steps `k`, tune capacities, spectral norm |

```
  MODE COLLAPSE picture: real data has many modes, G covers only one.

   real p_data:   /\   /\   /\   /\        (4 clusters)
   G's p_g:                  /\            (collapsed to 1)
```

> Mnemonic for the demons: **"MOVIN"** — **M**ode collapse, **O**scillation (non-convergence), **V**anishing gradient, **IN**stability/imbalance.

[Back to top](#top)

---

<a id="zoo"></a>
## 13. The GAN zoo (variants you must name)

```
                      vanilla GAN (2014)
                           |
      +--------------------+---------------------+-------------------+
      |                    |                     |                   |
  better arch          better loss          add control         image2image
  DCGAN (conv)         WGAN (Wasserstein)    cGAN (labels)       Pix2Pix (paired)
  ProGAN/StyleGAN      WGAN-GP (grad pen.)   InfoGAN (latent)    CycleGAN (unpaired)
  BigGAN, SAGAN        LSGAN (least sq.)     ACGAN               ESRGAN (super-res)
```

| Variant | One-line idea | Key trick |
|---|---|---|
| **DCGAN** | first stable conv GAN | strided/transposed conv, BatchNorm, no FC |
| **cGAN** | control what is generated | feed label `y` to both G and D |
| **WGAN** | fix vanishing gradient | use **Wasserstein (Earth-Mover)** distance; D→"critic" (no sigmoid); weight clipping |
| **WGAN-GP** | fix WGAN's clipping hack | replace clipping with **gradient penalty** (`λ=10`) |
| **LSGAN** | smoother gradients | least-squares loss instead of BCE |
| **InfoGAN** | disentangled features | maximise mutual info between latent code and output |
| **Pix2Pix** | image→image (paired) | cGAN + L1 loss, U-Net G, PatchGAN D |
| **CycleGAN** | image→image (unpaired) | two GANs + **cycle-consistency** loss |
| **ProGAN** | high-res, stable | grow resolution progressively 4→8→…→1024 |
| **StyleGAN(1/2/3)** | photoreal faces, style control | mapping network + AdaIN style injection |
| **BigGAN** | high-res class-conditional | huge batch + truncation trick |
| **SAGAN** | global structure | self-attention layers |
| **ESRGAN** | super-resolution | perceptual + adversarial loss |

**WGAN value (most-tested variant):**
```
  W(p_data, p_g) = sup_{||f||_L <= 1}  E_{x~p_data}[f(x)] - E_{x~p_g}[f(x)]
```
The "critic" `f` must be **1-Lipschitz** (enforced by weight clipping in WGAN, by gradient penalty in WGAN-GP). Wasserstein distance gives **smooth, non-vanishing gradients even when distributions don't overlap** — that's the whole point.

[Back to top](#top)

---

<a id="metrics"></a>
## 14. How do we score a GAN? (IS, FID)

You can't use accuracy (there's no label). The two standard scores:

**Inception Score (IS)** — *higher is better.*
```
  IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )
```
- `p(y|x)`: an Inception classifier's label prediction for one generated image — want it **confident** (sharp).
- `p(y)`: average over many images — want it **uniform/diverse** (all classes appear).
- High IS = sharp AND diverse. Weakness: ignores real data entirely.

**Fréchet Inception Distance (FID)** — *lower is better, the modern default.*
```
  FID = || mu_r - mu_g ||^2  +  Tr( S_r + S_g - 2 (S_r S_g)^(1/2) )
```
- Run real and fake images through Inception, grab feature vectors.
- Fit a Gaussian to each: real `(mu_r, S_r)`, fake `(mu_g, S_g)`.
- FID = distance between those two Gaussians. **0 = identical.**
- Catches mode collapse (IS doesn't); compares to real data.

```
   IS : "are my images sharp & varied?"     higher better
   FID: "how close are fakes to reals?"      lower  better  <- preferred
```

[Back to top](#top)

---

<a id="mnemonics"></a>
## 15. Mnemonics block

```
+---------------------------------------------------------------+
|  GAN MEMORY KIT                                               |
+---------------------------------------------------------------+
|  ROLES:   "Forger fools FBI"                                  |
|           G = forGer (generates),  D = Detective (decides)   |
|                                                               |
|  GOAL:    "G goes LOW, D goes HIGH"  on the same V            |
|           min_G  max_D  V(D,G)                                |
|                                                               |
|  D* :     "DATA over DATA-plus-GENERATED"                     |
|           D*(x) = p_data / (p_data + p_g)                     |
|                                                               |
|  OPTIMUM: "minus log four, then no more"                      |
|           V* = -log 4  <=>  p_g = p_data  <=>  D = 1/2        |
|                                                               |
|  WHAT IT MINIMISES: "Just Solve Divergence" = JSD             |
|                                                               |
|  LOSS FIX: "Non-Sat = Don't Saturate"                         |
|           use max log D(G(z)), not min log(1-D(G(z)))         |
|                                                               |
|  FAILURES: "MOVIN" = Mode collapse, Oscillation,              |
|             Vanishing grad, INstability                       |
|                                                               |
|  DCGAN:   "SBR-LRT-A"                                         |
|           Strided conv, BatchNorm, Remove FC,                |
|           LeakyReLU(D)/ReLU(G), Tanh out, Adam(2e-4, b1=.5)  |
|                                                               |
|  METRICS: "IS up, FID down"                                   |
+---------------------------------------------------------------+
```

[Back to top](#top)

---

<a id="cheatsheet"></a>
## 16. Cheatsheet (one screen)

```
VALUE FUNCTION
   min_G max_D  V = E_{x~pdata}[log D(x)] + E_{z~pz}[log(1 - D(G(z)))]

LOSSES (minimise these in code)
   L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
   L_G(non-sat) = -E[log D(G(z))]          <- use this
   L_G(saturating) = E[log(1 - D(G(z)))]   <- avoid (vanishing grad)

OPTIMAL D     D*(x) = pdata(x) / (pdata(x) + pg(x))
GLOBAL OPT    V* = -log4,  reached iff  pg = pdata,  D = 1/2
EQUIVALENT TO minimising  2*JSD(pdata || pg)

WGAN          W = sup_{||f||_L<=1}  E_pdata[f] - E_pg[f]
              critic (no sigmoid); WGAN clip weights; WGAN-GP grad penalty lambda=10

METRICS       IS = exp(E_x KL(p(y|x)||p(y)))   higher better
              FID = ||mu_r-mu_g||^2 + Tr(Sr+Sg-2(SrSg)^1/2)  lower better

CONV SIZES    conv:  O = floor((W - K + 2P)/S) + 1
              convT: O = (W-1)*S - 2P + K (+output_padding)

DEFAULTS      z~N(0,I) dim 100; Adam lr=2e-4 beta1=0.5 beta2=0.999;
              batch 64-128; G out Tanh -> normalise data to [-1,1]
```

[Back to top](#top)

---

<a id="exam-hacks"></a>
## 17. Exam hacks

1. **If asked "derive optimal D"** → set `f(y)=a log y + b log(1-y)`, `f'=0`, get `y=a/(a+b)`. Always worth full marks.
2. **If asked the global optimum value** → answer `-log 4` and state `p_g = p_data`, `D=1/2`. Mention JSD.
3. **"Why non-saturating loss?"** → early training fakes are obvious, `D(G(z))→0`, the `log(1-D(G(z)))` term saturates (flat gradient); flipping to `max log D(G(z))` keeps gradients strong.
4. **"What is mode collapse and a fix?"** → G maps many `z` to one output; fix with WGAN / minibatch discrimination / unrolled GAN.
5. **"GAN vs VAE"** → VAE = explicit (maximises a likelihood lower bound, blurry, stable); GAN = implicit (adversarial, sharper, unstable). Both are deep generative models.
6. **"Why WGAN?"** → JSD gives zero/vanishing gradient when supports don't overlap; Wasserstein distance is continuous & gives usable gradients everywhere. Critic must be 1-Lipschitz.
7. **"How to evaluate?"** → FID (lower better) is the modern standard; mention IS and its blindness to mode collapse.
8. **Always label your diagram**: noise `z` → G → fake → D ← real, gradients backward. Draw it; partial credit is free.
9. **Zero-sum / minimax / Nash equilibrium / saddle point** — drop these exact words; they're keyword-scored.
10. **Numbers to memorise**: `D*=1/2`, `V*=-log4≈-1.386`, WGAN-GP `λ=10`, DCGAN `lr=2e-4, β1=0.5`.

[Back to top](#top)

---

<a id="links"></a>
## 18. Where to go next

- **Worked numbers** (BCE plug-ins, D* values, JSD, FID math, conv-size sums): [DL_GANs_Numericals.md](./DL_GANs_Numericals.md)
- **Code from scratch** (simple GAN + DCGAN, every line explained, libs & deps): [DL_GANs_Practice.md](./DL_GANs_Practice.md)
- Repo home: [TS-02](https://github.com/rpaut03l/TS-02)
- Related generative models (VAE) in the same F3 unit — cross-link your VAE theory file here once written.

<a id="bottom"></a>

---
[Back to top](#top) | [Numericals ->](./DL_GANs_Numericals.md) | [Practice ->](./DL_GANs_Practice.md)
