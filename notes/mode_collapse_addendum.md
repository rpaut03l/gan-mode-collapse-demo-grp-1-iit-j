# Mode Collapse Presentation — Addendum (Applications, Differentiator Demo, Mechanism Visual, Trained-Model Demo)

Extends the main plan. Covers your four points and gives slide specs + a block to append to the PDF.

---

## Point 1 — Application use cases (where GANs are used, and where collapse bites)

GANs show up wherever we need to *generate* realistic data:

| Application | What the GAN does | Example model |
|---|---|---|
| Face / image synthesis | invents photoreal images from scratch | StyleGAN |
| Image-to-image translation | sketch->photo, summer->winter, horse->zebra | Pix2Pix, CycleGAN |
| Super-resolution | upscales a blurry image to sharp | SRGAN / ESRGAN |
| Data augmentation | makes synthetic training data for rare cases | medical imaging GANs |
| Inpainting | fills missing/erased parts of an image | context-encoder GANs |
| Anomaly detection | learns "normal", flags what doesn't fit | AnoGAN |
| Drug / molecule design | proposes new candidate molecules | MolGAN |
| Audio / voice synthesis | generates speech and sound | GAN vocoders |

**The hook for the slide — why collapse matters in each:** a collapsed GAN destroys the *one thing* these applications need most — variety.
- Data augmentation that only generates common cases is useless for the rare disease you were trying to cover.
- A face generator that makes only one kind of face is biased and worthless.
- Molecule design that proposes one scaffold finds no new drugs.

**One-line takeaway:** "Every GAN application is built on diversity — mode collapse is the failure that quietly removes it."

---

## Point 3 — Differentiator: the "ask for one of us, get someone else" demo

This is what makes the talk memorable and clearly *ours*. We use a **conditional GAN** idea: you feed in a label (which group member you want), and the generator should produce that person.

- **Healthy conditional GAN:** ask for Pujan -> get Pujan, ask for Disha -> get Disha. The label controls the output.
- **Collapsed conditional GAN:** ask for *anyone* -> always get the same person (e.g. Sharvan). The generator ignores the label and produces one identity. That is mode collapse on the identity axis.

**The slide (this is the differentiator):**

```
   REQUESTED (the label / condition)   ->   COLLAPSED GENERATOR OUTPUT

       Pujan   ----------------------->        Sharvan
       Disha   ----------------------->        Sharvan
       Jeenal  ----------------------->        Sharvan
       Rohit   ----------------------->        Sharvan
```
Caption: "We asked for four different people. The collapsed generator gave us Sharvan every time."

Optionally show the healthy version next to it (each request returns the right person) so the contrast is obvious.

**How to actually build it — two options:**
- *Illustrative (recommended for a 10-min talk):* use the group's own photos as the "requested" column and route them all to Sharvan's photo in the "output" column. Label the slide as an illustration of the concept. Zero training, maximum clarity, and it's funny.
- *Real (only if you want the wow):* train a small conditional face GAN on the group's photos, or use an existing face toolkit, and deliberately induce collapse (small latent dim, over-trained D). Heavier; not needed to make the point.

**Two cautions:** (1) get a quick OK from the group members whose photos you use — it's their faces. (2) Keep it clearly a class illustration, not anything that looks like a real impersonation tool.

**Note on your shorthand:** I read "G.C." as the *conditioning label* given to the generator (the requested identity). If you meant something else, the structure above still works — just relabel the left column.

---

## Point 4 — The dominating visual: how/why mode collapse forms

The centerpiece image (built separately, theme-safe, drop the SVG straight into the PDF): two distributions side by side. Gray bars = the real data's modes; violet = what the generator produces. Left, violet sits in every mode. Right, violet over-pumps one mode and the other modes are empty dashed outlines.

**The mechanism, in four steps (put these as a caption strip under the visual):**
1. The generator finds the one mode that most easily fools the current discriminator, and piles all its output there.
2. The discriminator adapts and learns to reject that mode.
3. Instead of *spreading out* to cover other modes, the generator **hops** to a different single mode.
4. The discriminator chases; the generator hops again -> it oscillates between a few modes and never covers them all.

**Why it happens (the root causes, one line each):**
- Nothing in the basic GAN loss *rewards* variety — fooling the judge once is enough.
- The discriminator judges one sample at a time, so it can't punish "all of these are identical."
- The Jensen-Shannon objective can give weak or misleading gradients, so the generator has no strong push to cover everything.

This is the slide to spend the most time on — it earns the "dominating feature" label because it connects the picture (empty modes) to the cause (the hop loop).

### SVG source for the centerpiece (paste into the PDF / Kimi prompt verbatim)

```svg
<svg width="100%" viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg">
  <!-- legend -->
  <rect x="206" y="36" width="13" height="13" rx="3" fill="#888780"/>
  <text x="224" y="47" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">real data modes</text>
  <rect x="330" y="36" width="13" height="13" rx="3" fill="#7C5CFF"/>
  <text x="348" y="47" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">generator output</text>
  <!-- panel titles -->
  <text x="182" y="84" text-anchor="middle" font-size="16" fill="#E8E8F0" font-family="Inter,sans-serif">what we want</text>
  <text x="182" y="102" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">covers every mode</text>
  <text x="492" y="84" text-anchor="middle" font-size="16" fill="#E8E8F0" font-family="Inter,sans-serif">mode collapse</text>
  <text x="492" y="102" text-anchor="middle" font-size="12" fill="#FBBF24" font-family="Inter,sans-serif">stuck on one mode</text>
  <!-- healthy panel: gray data bars + violet generator bars on all 4 modes -->
  <rect x="73"  y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="137" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="200" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="264" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="82"  y="192" width="20" height="78" rx="3" fill="#7C5CFF"/>
  <rect x="146" y="192" width="20" height="78" rx="3" fill="#7C5CFF"/>
  <rect x="209" y="192" width="20" height="78" rx="3" fill="#7C5CFF"/>
  <rect x="273" y="192" width="20" height="78" rx="3" fill="#7C5CFF"/>
  <line x1="55" y1="270" x2="315" y2="270" stroke="#9AA0B5" stroke-width="0.5"/>
  <text x="92"  y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m1</text>
  <text x="156" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m2</text>
  <text x="219" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m3</text>
  <text x="283" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m4</text>
  <!-- collapsed panel: gray data bars; one tall violet spike; 3 empty dashed outlines -->
  <rect x="378" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="442" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="505" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="569" y="180" width="38" height="90" rx="4" fill="#888780" opacity="0.5"/>
  <rect x="387" y="192" width="20" height="78" rx="3" fill="none" stroke="#7C5CFF" stroke-width="1" stroke-dasharray="3 3"/>
  <rect x="451" y="150" width="20" height="120" rx="3" fill="#7C5CFF"/>
  <rect x="514" y="192" width="20" height="78" rx="3" fill="none" stroke="#7C5CFF" stroke-width="1" stroke-dasharray="3 3"/>
  <rect x="578" y="192" width="20" height="78" rx="3" fill="none" stroke="#7C5CFF" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="360" y1="270" x2="620" y2="270" stroke="#9AA0B5" stroke-width="0.5"/>
  <text x="397" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m1</text>
  <text x="461" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m2</text>
  <text x="524" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m3</text>
  <text x="588" y="287" text-anchor="middle" font-size="12" fill="#9AA0B5" font-family="Inter,sans-serif">m4</text>
</svg>
```

---

## Point 5 — Trained G and D demo, framed like RAG poisoning

The angle: mode collapse is to a GAN what poisoning is to RAG — the model gets hijacked into producing **one output regardless of input**. Present it as a measurable before/after, exactly like a security demo.

**Define a single metric (the "attack-success-rate" equivalent):** `Mode Coverage` = how many of the data's modes the generator actually produces. On MNIST there are 10 modes (digits 0-9).

```
   GAN STATE          MODE COVERAGE        VERDICT
   healthy model      10 / 10 (100%)       diverse, working
   collapsed model     1 / 10 ( 10%)       hijacked to one output
```

**Two ways to show it (pick one for the live slot):**

- **Training mode (watch it happen live):** show sample grids at epoch 10 / 50 / 100 and the coverage metric falling 10 -> 4 -> 1. Same shape as watching an attack succeed over time.
- **Testing / inference mode (clean A/B):** load two saved checkpoints — one healthy, one collapsed — sample from each, show "all digits" vs "all 3s". Fastest and most reliable for a presentation.

**How to get a collapsed model on purpose** (build on the GAN code from the practice file):
- shrink the noise dimension, over-train D relative to G, drop any diversity safeguards — collapse appears quickly on MNIST.

```python
# measure mode coverage with a pre-trained digit classifier
import torch
@torch.no_grad()
def mode_coverage(generator, classifier, z_dim, n=1000, device="cpu"):
    z = torch.randn(n, z_dim, device=device)          # sample noise
    imgs = generator(z)                                # generate fakes
    preds = classifier(imgs).argmax(dim=1)             # which digit is each?
    modes = preds.unique().numel()                     # how many distinct digits
    return modes, modes / 10.0                          # count, fraction of 10

# healthy_G -> ~ (10, 1.0)   |   collapsed_G -> ~ (1, 0.1)
```

**Framing line for the slide:** "Healthy: 10/10 modes. Collapsed: 1/10. The generator was hijacked into one answer — the GAN version of a poisoned response."

---

## Append this to your Kimi PDF prompt (4 new slides)

Insert these into the SLIDE CONTENT list; renumber as needed. Keep the same dark theme and violet=G / cyan=D coding.

```
NEW SLIDE A — APPLICATIONS (after the intro)
  Headline: "Where GANs are used"
  4 short items: image synthesis (StyleGAN); image-to-image (Pix2Pix/CycleGAN);
  super-resolution (ESRGAN); data augmentation for rare cases.
  Bottom line (amber): "Every use case depends on variety — collapse removes it."

NEW SLIDE B — DOMINATING VISUAL: HOW IT FORMS  (make this the visual centerpiece)
  Headline: "How mode collapse forms"
  Use the provided SVG verbatim as the main graphic (gray data modes vs violet
  generator; one mode over-produced, the rest empty dashed outlines).
  Caption strip, 4 steps: 1) G piles onto the easiest mode that fools D.
  2) D learns to reject it. 3) G hops to another single mode instead of spreading.
  4) D chases, G hops again -> never covers all modes.

NEW SLIDE C — DIFFERENTIATOR: ASK FOR ONE OF US, GET SOMEONE ELSE
  Headline: "Same demo, our faces"
  Two columns: REQUESTED (Pujan, Disha, Jeenal, Rohit) -> COLLAPSED OUTPUT
  (Sharvan, Sharvan, Sharvan, Sharvan). Use the group's photos.
  Caption: "We asked for four people. The collapsed generator gave us Sharvan every time."
  (Mark as an illustration; get group OK for the photos.)

NEW SLIDE D — MEASURED COLLAPSE, RAG-POISONING STYLE
  Headline: "A trained generator, hijacked"
  Metric cards: Healthy = 10/10 modes (100%); Collapsed = 1/10 modes (10%).
  Optionally a small healthy-vs-collapsed sample grid (all digits vs all 3s).
  Line: "Mode collapse is to a GAN what poisoning is to RAG — one output, whatever the input."
```
