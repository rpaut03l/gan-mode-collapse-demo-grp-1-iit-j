# Deep Learning :: Generative Adversarial Networks (GANs) — NUMERICALS

> Part 2 of 3 — **Numerical Analysis** | [Theory](./DL_GANs_Theory.md) | [Practice](./DL_GANs_Practice.md)

<a id="top"></a>

---

## Table of Contents

1. [Tiny refresher: the formulas you'll plug into](#formulas)
2. [Q1 — Discriminator BCE loss (numbers)](#q1)
3. [Q2 — Optimal discriminator value D*](#q2)
4. [Q3 — Value function at the optimum (−log4)](#q3)
5. [Q4 — Jensen–Shannon divergence by hand](#q4)
6. [Q5 — Saturating vs non-saturating gradient](#q5)
7. [Q6 — KL divergence between two discrete dists](#q6)
8. [Q7 — Wasserstein distance (1-D example)](#q7)
9. [Q8 — FID computed by hand (1-D)](#q8)
10. [Q9 — Inception Score (toy)](#q9)
11. [Q10 — Conv & transposed-conv output sizes](#q10)
12. [Q11 — Count DCGAN parameters](#q11)
13. [Formula cheatsheet](#cheatsheet)
14. [Exam hacks](#exam-hacks)

[Skip to bottom](#bottom)

---

<a id="formulas"></a>
## 1. Tiny refresher: the formulas you'll plug into

```
  V(D,G) = E_pdata[log D(x)] + E_pz[log(1 - D(G(z)))]
  D*(x)  = pdata(x) / (pdata(x) + pg(x))
  V*     = -log 4
  KL(P||Q) = SUM_i P_i log(P_i / Q_i)
  JSD(P||Q)= 1/2 KL(P||M) + 1/2 KL(Q||M),  M = (P+Q)/2
  BCE(y,p) = -[ y log p + (1-y) log(1-p) ]
  conv out : O = floor((W - K + 2P)/S) + 1
  convT out: O = (W-1)*S - 2P + K
```
(All `log` are natural log unless stated. `log 2 = 0.6931`, `log 4 = 1.3863`, `log 10 = 2.3026`.)

[Back to top](#top)

---

<a id="q1"></a>
## 2. Q1 — Discriminator BCE loss (numbers)

**Given:** a batch of 2 reals and 2 fakes. D outputs:
- reals: `D(x1)=0.9`, `D(x2)=0.8`
- fakes: `D(G(z1))=0.3`, `D(G(z2))=0.2`

**Find** the discriminator loss `L_D` (averaged).

```
  L_D = -(1/m) SUM [ log D(x_i) + log(1 - D(G(z_i))) ]     (m matched pairs = 2)

  real terms:  log 0.9 = -0.1054 ;  log 0.8 = -0.2231
  fake terms:  log(1-0.3)=log0.7=-0.3567 ; log(1-0.2)=log0.8=-0.2231

  pair1 = (-0.1054) + (-0.3567) = -0.4621
  pair2 = (-0.2231) + (-0.2231) = -0.4462

  L_D = -(1/2)[ -0.4621 + -0.4462 ] = -(1/2)(-0.9083) = 0.4542
```
**Answer:** `L_D ≈ 0.454`. (Lower = D is doing well; a perfect D would push reals→1, fakes→0, loss→0.)

[Back to top](#top)

---

<a id="q2"></a>
## 3. Q2 — Optimal discriminator value D*

**Given** at a point `x`: `p_data(x) = 0.6`, `p_g(x) = 0.2`. **Find** `D*(x)`.

```
  D*(x) = pdata / (pdata + pg) = 0.6 / (0.6 + 0.2) = 0.6 / 0.8 = 0.75
```
**Answer:** `D*(x)=0.75`. (Real density 3× the fake density → D is 75% sure it's real.)

**Follow-up:** where `p_g(x) = p_data(x) = 0.4`:
```
  D*(x) = 0.4 / (0.4 + 0.4) = 0.5
```
→ **0.5**, the "converged / can't tell" value.

[Back to top](#top)

---

<a id="q3"></a>
## 4. Q3 — Value function at the optimum (show it equals −log4)

**Claim:** when `p_g = p_data`, `D*=1/2` everywhere, and `V = -log 4`.

```
  V(D*,G) = E_pdata[log(1/2)] + E_pg[log(1 - 1/2)]
          = log(1/2) + log(1/2)         (expectations of a constant)
          = -log2 + -log2
          = -2 log 2
          = -log 4
          = -1.3863
```
**Answer:** `V* = -log 4 ≈ -1.386`. This is the global minimum of the generator's objective.

[Back to top](#top)

---

<a id="q4"></a>
## 5. Q4 — Jensen–Shannon divergence by hand

**Given** two 2-outcome distributions:
- `P = [0.5, 0.5]` (real), `Q = [0.9, 0.1]` (fake).

**Step 1 — midpoint** `M = (P+Q)/2`:
```
  M = [ (0.5+0.9)/2 , (0.5+0.1)/2 ] = [0.7, 0.3]
```
**Step 2 — KL(P||M):**
```
  = 0.5 log(0.5/0.7) + 0.5 log(0.5/0.3)
  = 0.5 log(0.7143)  + 0.5 log(1.6667)
  = 0.5(-0.3365)     + 0.5(0.5108)
  = -0.1683 + 0.2554 = 0.0871
```
**Step 3 — KL(Q||M):**
```
  = 0.9 log(0.9/0.7) + 0.1 log(0.1/0.3)
  = 0.9 log(1.2857)  + 0.1 log(0.3333)
  = 0.9(0.2513)      + 0.1(-1.0986)
  = 0.2262 - 0.1099 = 0.1163
```
**Step 4 — JSD:**
```
  JSD = 1/2 (0.0871) + 1/2 (0.1163) = 0.0436 + 0.0582 = 0.1017
```
**Answer:** `JSD(P||Q) ≈ 0.102 nats`. And the GAN objective at optimal D would be `C(G) = -log4 + 2(0.1017) = -1.3863 + 0.2035 = -1.1828`. As Q → P, JSD → 0 and C(G) → −log4.

> Note: JSD is **symmetric** and **bounded** by `log 2 ≈ 0.693` (in nats). KL is neither.

[Back to top](#top)

---

<a id="q5"></a>
## 6. Q5 — Saturating vs non-saturating gradient (why we switch)

Early training, a bad fake gives `D(G(z)) = 0.01`.

**Saturating loss** `L = log(1 - D(G(z)))`. Gradient wrt `D(G(z))` (call it `d`):
```
  dL/dd = -1 / (1 - d) = -1 / (1 - 0.01) = -1.0101   (small magnitude near d=0)
```
Now a *good* fake `d=0.5`: `-1/(1-0.5) = -2`. The gradient is **weakest exactly when the fake is worst** — backwards from what we want.

**Non-saturating loss** `L = -log D(G(z))`:
```
  dL/dd = -1/d = -1/0.01 = -100     (huge when fake is bad -> learns fast)
```
At `d=0.5`: `-1/0.5 = -2`. **Strong push when the fake is bad** — correct direction.

**Answer:** non-saturating gives ~100× the gradient at `d=0.01`, so G actually moves early in training.

[Back to top](#top)

---

<a id="q6"></a>
## 7. Q6 — KL divergence between two discrete dists

**Given** `P=[0.7,0.2,0.1]`, `Q=[0.5,0.3,0.2]`. Compute `KL(P||Q)`.
```
  KL = 0.7 log(0.7/0.5) + 0.2 log(0.2/0.3) + 0.1 log(0.1/0.2)
     = 0.7 log(1.4)     + 0.2 log(0.6667)  + 0.1 log(0.5)
     = 0.7(0.3365)      + 0.2(-0.4055)     + 0.1(-0.6931)
     = 0.2355 - 0.0811 - 0.0693
     = 0.0851 nats
```
**Answer:** `KL(P||Q) ≈ 0.085`. (Try `KL(Q||P)` to see it's **not** equal — KL is asymmetric.)

[Back to top](#top)

---

<a id="q7"></a>
## 8. Q7 — Wasserstein distance (1-D example)

For 1-D distributions, the **Wasserstein-1 (Earth-Mover)** distance equals the area between their CDFs, or simply, for point masses, the "dirt-moving cost".

**Given** real mass all at `x=0`, fake mass all at `x=θ` (a classic non-overlapping case).
```
  JSD(real, fake) = log 2  (constant for ALL theta != 0)  -> gradient w.r.t theta = 0  (BAD)
  W(real, fake)   = |theta|                                -> gradient = sign(theta)   (GOOD)
```
**Answer:** even though the supports never overlap, `W = |θ|` decreases smoothly to 0 as `θ→0`, giving a usable gradient. JSD is stuck at `log2` and gives **no** gradient. **This single example is the entire motivation for WGAN.**

```
  loss vs theta:
    JSD : ___________   (flat at log2, no slope -> G can't learn)
    W   :     \    /    (V-shape, slope points to 0 -> G learns)
               \  /
                \/  at theta=0
```

[Back to top](#top)

---

<a id="q8"></a>
## 9. Q8 — FID computed by hand (1-D)

In 1-D, covariances are scalars, so:
```
  FID = (mu_r - mu_g)^2 + ( s_r + s_g - 2 sqrt(s_r * s_g) )
```
**Given** real features `~ N(mu=2, var=4)`, fake features `~ N(mu=2.5, var=1)`.
```
  mean term  : (2 - 2.5)^2 = 0.25
  var  term  : 4 + 1 - 2*sqrt(4*1) = 5 - 2*sqrt(4) = 5 - 2*2 = 5 - 4 = 1
  FID        : 0.25 + 1 = 1.25
```
**Answer:** `FID = 1.25`. If fakes matched reals exactly (`mu=2, var=4`): mean term 0, var term `4+4-2*sqrt(16)=8-8=0` → **FID=0**. Lower is better.

[Back to top](#top)

---

<a id="q9"></a>
## 10. Q9 — Inception Score (toy)

**Given** a generator producing images that the Inception classifier maps to (3-class) label dists:
- img A: `p(y|A)=[0.9,0.05,0.05]`
- img B: `p(y|B)=[0.05,0.9,0.05]`

Marginal `p(y) = average = [0.475, 0.475, 0.05]`.

KL for each image `KL(p(y|x)||p(y))`:
```
  img A: 0.9 log(0.9/0.475) + 0.05 log(0.05/0.475) + 0.05 log(0.05/0.05)
       = 0.9 log(1.8947)    + 0.05 log(0.1053)     + 0.05 log(1)
       = 0.9(0.6391)        + 0.05(-2.2513)        + 0
       = 0.5752 - 0.1126 + 0 = 0.4626

  img B: by symmetry = 0.4626

  E_x[KL] = (0.4626 + 0.4626)/2 = 0.4626
  IS = exp(0.4626) = 1.588
```
**Answer:** `IS ≈ 1.59`. Confident (sharp) + both classes appear (diverse) → IS above 1. A generator that output only class-1 images would have `p(y|x)=p(y)` → KL 0 → IS = 1 (worst diversity).

[Back to top](#top)

---

<a id="q10"></a>
## 11. Q10 — Conv & transposed-conv output sizes

**Conv (D, downsampling):** input `W=64`, kernel `K=4`, stride `S=2`, padding `P=1`.
```
  O = floor((W - K + 2P)/S) + 1 = floor((64 - 4 + 2)/2) + 1
    = floor(62/2) + 1 = 31 + 1 = 32
```
**Answer:** `64 -> 32`. (Each strided conv halves spatial size — that's the DCGAN downsampling block.)

**Transposed conv (G, upsampling):** input `W=4`, `K=4`, `S=2`, `P=1`.
```
  O = (W - 1)*S - 2P + K = (4-1)*2 - 2*1 + 4 = 6 - 2 + 4 = 8
```
**Answer:** `4 -> 8`. (Each transposed conv doubles size — the DCGAN upsampling block.)

**Chain for DCGAN G:** `4 ->8 ->16 ->32 ->64`. For D: `64 ->32 ->16 ->8 ->4`. Mirror images.

[Back to top](#top)

---

<a id="q11"></a>
## 12. Q11 — Count DCGAN parameters (one conv layer)

Params in a conv layer = `(K * K * C_in * C_out) + C_out` (the `+C_out` is the bias; **0 if BatchNorm replaces bias**).

**Given** a G layer: `K=4`, `C_in=512`, `C_out=256`, with BatchNorm (no bias).
```
  weights = 4*4*512*256 = 16 * 131072 = 2,097,152
  bias    = 0  (folded into BatchNorm)
  BatchNorm params = 2 * C_out = 2*256 = 512   (gamma + beta)
  total   = 2,097,152 + 512 = 2,097,664
```
**Answer:** ≈ **2.10 M** parameters in that single layer. (Stack 4–5 such layers and you see why DCGANs are millions of params.)

[Back to top](#top)

---

<a id="cheatsheet"></a>
## 13. Formula cheatsheet

```
  BCE        -(y log p + (1-y) log(1-p))
  L_D        -E[log D(x)] - E[log(1-D(G(z)))]
  D*         pdata/(pdata+pg)
  V*         -log4 = -1.3863      (pg=pdata, D=1/2)
  C(G)       -log4 + 2*JSD(pdata||pg)
  KL(P||Q)   SUM P log(P/Q)        (asymmetric, unbounded)
  JSD        1/2 KL(P||M)+1/2 KL(Q||M), M=(P+Q)/2  (symmetric, <= log2)
  W-1 (1D)   area between CDFs ; point masses: |dirt moved|
  FID        ||mu_r-mu_g||^2 + Tr(Sr+Sg-2(SrSg)^1/2)   (1D: var term s_r+s_g-2 sqrt(s_r s_g))
  IS         exp(E_x KL(p(y|x)||p(y)))
  conv       O=floor((W-K+2P)/S)+1
  convT      O=(W-1)S-2P+K
  params     K*K*Cin*Cout (+Cout bias)   BN adds 2*Cout

  useful logs: log2=0.6931  log4=1.3863  log10=2.3026  log0.5=-0.6931
```

[Back to top](#top)

---

<a id="exam-hacks"></a>
## 14. Exam hacks

1. **D* sums are free marks** — always `pdata/(pdata+pg)`; plug and divide.
2. **Show `V*=-log4`** by substituting `D=1/2`: `log(1/2)+log(1/2)=-log4`. Two lines.
3. **JSD/KL**: write `M=(P+Q)/2` first; compute each KL term-by-term in a small table; keep 4 decimals.
4. **WGAN motivation Q** → always use the non-overlapping point-mass example: JSD=log2 (flat), W=|θ| (sloped).
5. **FID/IS direction** — write "FID lower better, IS higher better" before computing; graders look for it.
6. **Conv sizes** — memorise: stride-2 K-4 P-1 halves (D) / doubles (convT, G). 64↔32↔16↔8↔4.
7. **Keep natural log** unless told otherwise; state your log base. Carry `log2=0.6931`.
8. **Sanity-check every answer**: `D*∈[0,1]`, `JSD∈[0,log2]`, `FID≥0`, `IS≥1`.

[Back to top](#top)

<a id="bottom"></a>

---
[Back to top](#top) | [<- Theory](./DL_GANs_Theory.md) | [Practice ->](./DL_GANs_Practice.md)
