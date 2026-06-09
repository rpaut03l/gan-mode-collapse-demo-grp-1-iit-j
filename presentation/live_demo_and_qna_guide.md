# Live Demo & Q&A Master Guide — Mode Collapse in GANs

> Read this once and you'll know how every part of the demo works and how to answer almost any question the audience or professor throws at the group.

<a id="top"></a>

---

## Table of Contents

1. [What the live demo actually shows](#what)
2. [Be honest: what's real vs illustrated](#honest)
3. [The screen, part by part](#screen)
4. [Every control explained](#controls)
5. [Every metric explained](#metrics)
6. [What the colours mean](#colours)
7. [How to narrate the demo (tie-in to the talk)](#narrate)
8. [Q&A bank — everything you might be asked](#qa)
9. [If you genuinely don't know an answer](#dontknow)
10. [Who owns which questions](#owners)

[Skip to bottom](#bottom)

---

<a id="what"></a>
## 1. What the live demo actually shows

The demo is a grid of generated "samples" (shown as common animals). As training progresses, you watch the generator **lose variety** — different cells stop producing different animals and all converge onto one (a cat). A side panel tracks **mode coverage** (how many of the 10 animal-types are still being produced) and it falls from ~10/10 to 1/10. Then one click (`Reveal the fix`) shows diversity returning.

In one sentence: **it makes the abstract idea of "the generator collapses onto one output" something you can literally watch happen and then watch get fixed.**

[Back to top](#top)

---

<a id="honest"></a>
## 2. Be honest: what's real vs illustrated

This matters for academic integrity and for the inevitable "is this a real model?" question. The honest answer:

- The demo is an **illustration / simulation of the dynamics**, not a live-trained neural network running in the browser.
- The **behaviour it shows is faithful** to what real GANs do: diversity collapsing onto a few modes, coverage dropping, losses oscillating instead of converging, and diversity returning after a fix like minibatch discrimination or WGAN.
- A real trained GAN would need model weights and a runtime that don't fit cleanly in a standalone file; the practice notes contain real, runnable PyTorch code that produces the same effect on MNIST.

**Sayable line if asked:** "It's a faithful simulation of the training dynamics so it runs anywhere — the actual collapse, the coverage drop, and the recovery all mirror real GAN behaviour, and we have the runnable PyTorch version in our notes."

[Back to top](#top)

---

<a id="screen"></a>
## 3. The screen, part by part

```
  +-----------------------------------------------------------+
  |  Live demo - watching a GAN collapse                      |
  |  [Collapse run] [Healthy run]   [Play] [Reveal fix] [Reset]|
  |  Epoch  |==============o-----------------|   42            |
  |                                                            |
  |   dog cat fox owl pig cow ram  +----------------------+    |
  |   cat dog cat hen fox cat pig  | Mode coverage   7/10 |    |
  |   cat cat cat fox cat cat dog  | Generator loss  1.24 |    |
  |   ...  (grid of animals) ...   | Discrim. loss   0.41 |    |
  |                                | Epoch           42   |    |
  |  Mode collapse is to a GAN what poisoning is to RAG.      |
  +-----------------------------------------------------------+
```

- **Left** = the generator's output grid (the "samples").
- **Right** = the live metrics panel.
- **Top** = the controls.
- **Bottom** = the one-line framing.

[Back to top](#top)

---

<a id="controls"></a>
## 4. Every control explained

| Control | Key | What it does | When to use it |
|---|---|---|---|
| **Collapse run** | `C` | Sets the demo to the *failing* scenario: as epochs advance, the generator narrows onto one digit. | Default; the main story. |
| **Healthy run** | `H` | Sets the demo to the *working* scenario: the generator keeps producing all ten digits no matter how far you train. | To contrast "good vs bad" side by side. |
| **Play / Pause** | `space` | Auto-advances training epoch by epoch so the grid animates. Pressing again pauses. | During the detection section — press Play and let it collapse. |
| **Apply a fix** | `1` `2` `3` | Three buttons, each applies a fix and explains the technique: `1` = **WGAN** (loss fix), `2` = **minibatch discrimination**, `3` = **unrolled GAN** (both structural). The collapsed cells re-diversify and coverage climbs back to 10/10. | Sharvan presses `1` (loss fix); Pujan presses `2` (structural fix). |
| **Epoch slider** | `← →` | Manually scrub to any point in training. Lets you pause exactly on the moment of collapse. | To freeze on a dramatic frame, or step slowly. |
| **Reset** | `R` | Clears the fix, returns to epoch 0, and re-seeds the grid with fresh samples. | Before each run / between rehearsals. |

**The "epoch" word:** an *epoch* is one full pass over the training data. Here the epoch number (0–120) is just the **training-progress axis** — higher epoch = "we've trained longer." Collapse appears as the epoch climbs.

[Back to top](#top)

---

<a id="metrics"></a>
## 5. Every metric explained

| Metric | Meaning | What to watch |
|---|---|---|
| **Mode coverage** (`n / 10`) | How many of the 10 animal-types the generator is currently producing across the whole grid. | Falls 10 → 1 during collapse; climbs back after the fix. This is the headline number. |
| **Verdict pill** | A one-word health label tied to coverage: `diverse` (≥8), `narrowing` (4–7), `hijacked` (≤3), `recovered` (after fix). | "Hijacked" is your RAG-poisoning moment. |
| **Generator loss** | A stand-in for how hard the generator is working to fool the discriminator. | Rises as collapse deepens — the generator is stuck. |
| **Discriminator loss** | A stand-in for how easily the discriminator tells real from fake. | Falls during collapse — the judge wins easily against repetitive fakes. |
| **Epoch** | The current training-progress step (0–120). | Just the timeline. |

**Important point for Q&A:** in a real GAN the losses **oscillate and don't settle** — that's normal, because it's a two-player game, not a single descent. The demo reflects this (the numbers wobble).

[Back to top](#top)

---

<a id="colours"></a>
## 6. What the colours mean

- **Generator (indigo / violet):** the generator and its normal, varied output.
- **Discriminator (cyan):** the judge network.
- **Collapsed (orange):** a cell that has collapsed onto the single repeated mode (a cat in the demo — the collapse target; the choice of animal is arbitrary).
- **Healthy / recovered (green), hijacked (red):** the coverage verdict states.

Consistent rule: **violet = generator, cyan = discriminator, orange = the problem, green = healthy.** The same coding runs through the slide deck.

[Back to top](#top)

---

<a id="narrate"></a>
## 7. How to narrate the demo (tie-in to the talk)

- **Rohit (what breaks):** press **Play** on a Collapse run and let it finish. "Early on the animals are varied — coverage near 10/10. As we train, watch cells turn orange and every one become a cat — coverage falls toward 1/10. Same output for very different inputs — that's collapse." Leave it collapsed.
- **Sharvan (loss fix):** press **`1`** (WGAN). "The loss fix: switching to the Wasserstein distance gives a real gradient again, so the generator recovers all the modes."
- **Disha (diagnosis):** press **`R`** then scrub the slider to ~110. "The loss just wobbles and looks fine — that's deceptive. The honest signals are coverage at 1/10 and a high FID."
- **Pujan (structural fix):** press **`2`** (minibatch discrimination). "A structural fix recovers diversity without touching the loss."
- **If asked to compare:** press **Healthy run** to show a generator that never collapses, then back to **Collapse run**.

[Back to top](#top)

---

<a id="qa"></a>
## 8. Q&A bank — everything you might be asked

### About the concept
| Question | Short answer |
|---|---|
| What is a GAN? | Two networks: a generator that makes fakes from noise and a discriminator that judges real vs fake. They train by competing. |
| What is mode collapse? | The generator loses output variety — many different inputs produce the same few (or one) output. |
| What is a "mode"? | A distinct type/cluster in the data; each animal type in the demo (or each digit 0–9 in MNIST) is one mode. |
| Complete vs partial collapse? | Complete = basically one output; partial = a few types while missing many. |

### About the mechanism
| Question | Short answer |
|---|---|
| Why does it happen? | Nothing in the basic loss rewards variety; the discriminator judges one sample at a time; and the gradients don't push the generator to spread out. |
| Why doesn't the discriminator just fix it? | It sees one sample at a time, so it can't notice "all of these are identical" — unless we give it batch information (minibatch discrimination). |
| What's the cat-and-mouse loop? | The generator locks onto one mode, the discriminator learns to reject it, so the generator hops to another single mode instead of spreading — and it oscillates forever. |
| Is it the generator's or discriminator's fault? | Neither alone — it's a failure of the adversarial game's dynamics and objective. |

### About detection
| Question | Short answer |
|---|---|
| How do you detect it? | Eyeball the samples, count how many classes appear, "walk" the noise (big input change, tiny output change = collapse), and check metrics. |
| Why can't you just watch the loss? | The training loss is deceptive — GAN losses oscillate, and a collapsed generator can still show a normal-looking loss. Judge diversity, not the loss. |
| What metrics? | FID (lower better), Inception Score (higher better), precision/recall for GANs, and mode coverage. |
| IS vs FID? | IS rewards images that are sharp and diverse (higher better); FID compares fakes to real data and catches collapse because diversity drops (lower better, the modern default). |
| What is mode coverage (in the demo)? | The count of distinct data modes the generator actually produces — here, distinct animals out of 10. |

### About fixes
| Question | Short answer |
|---|---|
| How do you fix it? | Two routes: a <b>loss fix</b> (WGAN / Earth Mover's distance) or <b>structural fixes</b> (minibatch discrimination, unrolled GAN, PacGAN, feature matching). |
| Why does JSD cause collapse? | The original loss minimises Jensen–Shannon divergence. When the fake and real distributions barely overlap (what happens at collapse), JSD is almost constant, so its gradient is nearly zero — the generator gets no signal to spread out. |
| What is the Earth Mover's / Wasserstein distance? | The minimum "cost" of moving probability mass to turn one distribution into the other. It varies smoothly even with no overlap, so it gives a usable gradient everywhere — the basis of WGAN. |
| What is minibatch discrimination? | Letting the discriminator compare samples across a whole batch so it can detect and punish lack of variety. |
| What is an unrolled GAN? | The generator trains against several look-ahead steps of the discriminator, so it can't exploit the current discriminator by collapsing onto one mode. |
| Why does WGAN help? | The Wasserstein distance gives smoother, non-vanishing gradients and is more robust to collapse. The critic must be 1-Lipschitz (weight clipping, or gradient penalty in WGAN-GP). |
| Does WGAN fully solve it? | It reduces it and is more robust, but it's not a guaranteed cure. |
| Simplest fix to try first? | Minibatch discrimination, or switch the loss to WGAN-GP. |

### About the demo itself
| Question | Short answer |
|---|---|
| Is this a real trained model? | It's a faithful simulation of the dynamics so it runs anywhere; the collapse, coverage drop, and recovery all mirror real GAN behaviour. We have runnable PyTorch in our notes. |
| What do the orange tiles mean? | Cells that have collapsed onto the single repeated mode (all cats). |
| Why does it collapse to a cat? | Arbitrary collapse target — a real collapse can land on any mode. It also nods to the analogy of a student who only ever draws a cat. |
| What does the epoch slider do? | Scrubs through training so we can pause on any moment, including the instant of collapse. |
| Why do the losses wobble instead of going to zero? | GAN training is a two-player game, so losses oscillate rather than converge — that's expected, not a bug. |
| What do the three fix buttons represent? | `1` applies WGAN (the loss fix), `2` minibatch discrimination and `3` unrolled GAN (structural fixes); each shows the technique and recovers coverage. |

### Metrics / light maths
| Question | Short answer |
|---|---|
| What's the optimal discriminator? | `D*(x) = p_data(x) / (p_data(x) + p_g(x))`. |
| What's the global optimum value? | `-log 4`, reached when the generator distribution equals the data distribution. |
| What does the GAN loss actually minimise? | The Jensen–Shannon divergence between the real and generated distributions. |
| Why the "RAG poisoning" analogy? | Both are a system hijacked into one output regardless of input — it frames collapse as a measurable failure. |

### Comparisons
| Question | Short answer |
|---|---|
| GAN vs VAE? | VAE optimises an explicit likelihood bound (more stable, blurrier); GAN is implicit and adversarial (sharper, less stable). Both are deep generative models. |
| GAN vs diffusion models? | Diffusion iteratively denoises — more stable and currently state-of-the-art for image quality, but slower to sample; GANs generate in one shot and are faster. |
| Mode collapse vs overfitting? | Overfitting = memorising training data; collapse = lack of output variety. Different problems. |
| Mode collapse vs vanishing gradient? | Vanishing gradient = the generator stops learning because the discriminator is too strong; collapse = the generator learns but only narrow outputs. |

### Applications
| Question | Short answer |
|---|---|
| Where are GANs used? | Face/image synthesis (StyleGAN), image-to-image (Pix2Pix, CycleGAN), super-resolution (ESRGAN), data augmentation, anomaly detection, drug design. |
| Real-world cost of collapse? | Data augmentation that misses rare cases, a face generator biased to one face, molecule design that finds no diversity. |

### Curveballs
| Question | Short answer |
|---|---|
| Can mode collapse ever be good? | No — narrow generation defeats the purpose of a generative model. |
| How long until collapse happens? | It varies and can be sudden; common with an imbalanced generator/discriminator or a too-small latent dimension. |
| Does a bigger model fix it? | Not directly — it's about the training objective and dynamics, not raw capacity. |
| If the data is unlabeled, how do you count modes? | Use a separate classifier or cluster the features; for MNIST we use the digit labels. |
| Is mode collapse unique to GANs? | It's specific to the adversarial setup; other generative models have different failure modes. |

[Back to top](#top)

---

<a id="dontknow"></a>
## 9. If you genuinely don't know an answer

Don't guess wildly. Acknowledge it, give the closest correct idea, and offer to follow up:
> "Good question — the core reason is [the closest thing you do know]; I'd want to check the exact details before saying more."

This reads as honest and competent, which is better than a confident wrong answer.

[Back to top](#top)

---

<a id="owners"></a>
## 10. Who owns which questions

So nobody freezes, each person is the first responder for their area (Pujan routes the question to the right person):

| Member | Owns questions about |
|---|---|
| **Jeenal** | What a GAN is, the two-player game, Nash equilibrium |
| **Rohit** | What collapse is, the mechanism, complete vs partial, "is the demo real?" |
| **Sharvan** | The loss math, why JSD fails, WGAN / Earth Mover's distance |
| **Disha** | Detection, deceptive loss, IS & FID, mode coverage |
| **Pujan** | Structural fixes (minibatch discrimination, unrolled GAN), the summary, routing Q&A |

[Back to top](#top)

<a id="bottom"></a>

---
Everything here matches the live demo and the slide deck. If a control or colour ever looks different, the demo file is the source of truth.
