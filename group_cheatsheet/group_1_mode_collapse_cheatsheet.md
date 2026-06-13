# Mode Collapse in GANs — Group 1 Cheatsheet

A simple, read-once guide so **everyone can present their own part in their own words**.
Each person: read your section, get the idea in your head, then say it naturally. You don't need to memorise — just understand.

Deck: `Mode_Collapse_GANs_Final_v4` · 15 slides · ~9 minutes total · Group 1
Repo: https://github.com/rpaut03l/gan-mode-collapse-demo-grp-1-iit-j

---

## The one idea the whole talk hangs on (read this first, everyone)

> A generator is supposed to copy **all** the variety in real data.
> But it is only graded on **"did you fool the judge?"** — never on "did you cover everything?"
> So it takes the lazy shortcut: find **one** easy thing that fools the judge, and keep making that.

Every slide is just this idea playing out. If you hold this sentence, your part comes out naturally.

---

## Who presents what (running order)

| # | Slide | Presenter | Time |
|---|-------|-----------|------|
| 1 | Title | All | — |
| 2 | Team & Flow | All | — |
| 3 | GAN Recap | **Jeenal** | ~1.5 min |
| 4 | Nash Equilibrium | **Jeenal** | |
| 5 | What Is Mode Collapse? | **Rohit** | ~2 min |
| 6 | How Mode Collapse Forms | **Rohit** | |
| 7 | Complete vs Partial Collapse | **Rohit** | |
| 8 | Mode Collapse on Our Faces | **Rohit** | |
| 9 | Why the Loss Fails | **Sharvan** | ~2 min |
| 10 | WGAN: The Wasserstein Fix | **Sharvan** | |
| 11 | Watch It Collapse (LIVE DEMO) | **Disha** | ~2 min |
| 12 | Detecting Collapse: IS & FID | **Disha** | |
| 13 | Structural Fixes | **Pujan** | ~2 min |
| 14 | Takeaway | **Pujan** | |
| 15 | Q&A | All | |

> Note on the demo: in this deck, the live demo on slide 11 is run by **Disha** (Demo_v2). Please confirm this is what we all agreed — earlier drafts had Rohit driving it.

---

## JEENAL — set up the game (slides 3–4)

**Slide 3 · GAN Recap**
Say it simply: "A GAN is two networks playing a game. The **generator** takes random noise and tries to make fake data that looks real. The **discriminator** is the judge that tries to tell real from fake. They compete, and that competition is what makes both get better."
- Key terms: generator G(z), discriminator D, noise z, minimax game.
- Remember: *two players, one game.*

**Slide 4 · Nash Equilibrium**
Say it simply: "The finish line of the game is when the generator's fakes match the real data so well that the judge can only guess — 50/50, meaning D(x) = 0.5. That's the ideal result, but in real training it's very hard to reach."
- Key terms: Nash equilibrium, generated distribution = real distribution (p_g = p_data), D(x) = 0.5.
- Remember: *50/50 — the judge is just guessing.*

---

## ROHIT — show what breaks and why (slides 5–8)

**Slide 5 · What Is Mode Collapse?**
Say it simply: "A healthy generator makes all the variety in the data. Mode collapse is when it gets lazy and only makes one thing — like a kid who can draw ten animals but only ever draws the cat. It still produces output, but the variety is gone, no matter the input noise."
- Key terms: mode (one type), variety / diversity, one mode.
- Remember: *the lazy cat-only kid.*

**Slide 6 · How Mode Collapse Forms**
Say it simply: "Why does it happen? The generator finds one easy output that fools the judge and piles everything there. The judge learns to reject it, so the generator hops to one new easy output. They keep going back and forth — it oscillates and gets stuck, never covering all the modes."
- Key terms: exploitative loop, easy 'sweet spot', judge adapts, oscillation.
- Remember: *cheat → caught → hop → repeat.*

**Slide 7 · Complete vs Partial Collapse**
Say it simply: "Two strengths. **Complete** collapse: almost every input maps to one near-identical output. **Partial** collapse: it captures a few types but still misses many. Partial is sneaky because it looks varied at a glance."
- Key terms: latent vectors z, complete vs partial.
- Remember: *one thing vs a few things.*

**Slide 8 · Mode Collapse on Our Faces**
Say it simply: "Now make it personal. Ask the collapsed model for Jeenal, Disha, Sharvan, Pujan — five different people — and it hands back the same face every time. Different inputs, one boring output."
- Remember: *all paths lead to Rohit.*

---

## SHARVAN — the loss problem and the fix (slides 9–10)

**Slide 9 · Why the Loss Fails**
Say it simply: "The standard GAN loss is based on **JSD**. When the fake and real distributions don't overlap — which is exactly what happens during collapse — JSD flattens to a constant (log 2 ≈ 0.693). A flat loss means the gradient is almost zero, so the generator gets no signal about which way to improve. The dilemma: a weak judge gives noisy feedback, a strong judge makes the gradient vanish."
- Key terms: JSD (Jensen–Shannon divergence), saturation, log(2) ≈ 0.693, vanishing gradient.
- Remember: *flat loss = no signal.*

**Slide 10 · WGAN: The Wasserstein Fix**
Say it simply: "WGAN swaps JSD for the **Wasserstein (Earth Mover's) distance** — the minimum cost of moving probability mass to turn one distribution into the other. It changes smoothly even when distributions don't overlap, so there's always a usable gradient. The catch: the critic has to be **1-Lipschitz**, enforced by weight clipping or a gradient penalty (WGAN-GP)."
- Key terms: Wasserstein distance, transport cost, 1-Lipschitz critic, weight clipping, WGAN-GP.
- Remember: *smooth slope instead of a flat cliff.*

---

## DISHA — diagnose it and run the demo (slides 11–12)

**Slide 11 · Watch It Collapse — LIVE DEMO (you drive this)**
Say it simply: "Let's watch it live. This is a 2D toy dataset with 10 Gaussian modes. I press play, and as the generator trains it loses coverage — the mode-coverage score drops from 10/10 to 1/10. That's collapse happening in real time."
- Key terms: mode coverage, 10 Gaussian modes, 10/10 → 1/10.
- Remember: *press play, ten to one.*
- Tip: if the live demo lags, just talk over the two pictures on the slide (full coverage → one mode). Nobody will know.

**Slide 12 · Detecting Collapse: IS & FID**
Say it simply: "Don't trust the training loss — a collapsed generator can show a low, healthy-looking loss while making no variety, so loss does not equal quality. Instead we use metrics. **Inception Score (IS)**: higher means sharper and more diverse, but it ignores the real data. **FID**: lower means closer to the real data, and it directly catches collapse because a collapsed model has a very different covariance from real data."
- Key terms: deceptive loss, Inception Score (IS), FID, covariance.
- Remember: *loss lies — FID tells the truth.*
- Note: the numbers on the slide (IS ~15.4, FID ~8.2) are example values, not measured results — present them as illustrations.

---

## PUJAN — the fixes and the wrap-up (slides 13–14)

**Slide 13 · Structural Fixes**
Say it simply: "We can also fix it without changing the loss. **Minibatch discrimination**: let the discriminator look at a whole batch at once, so it can spot and punish 'everything looks the same.' **Unrolled GAN**: let the generator look a few steps ahead at how the discriminator will react, so it can't win by collapsing onto one mode."
- Key terms: minibatch discrimination, batch similarity, unrolled GAN, surrogate loss, k steps.
- Remember: *let the judge see the whole batch; make the artist think ahead.*

**Slide 14 · Takeaway**
Say it simply: "Three pillars. One — the core mechanism: collapse trades variety for an easy win. Two — detection: never trust loss curves alone, measure coverage with FID and IS. Three — fixes: change the loss (WGAN) or the architecture (minibatch discrimination, unrolled GAN). One line to remember: mode collapse is the GAN version of a poisoned response — one output no matter the input."
- Remember: *trade-off → diagnose → fix.*

---

## Q&A — quick answers (anyone can field these)

- **What is a 'mode'?** A distinct type or cluster in the data — one animal, or one digit in MNIST.
- **Why does JSD cause collapse?** When the distributions don't overlap, JSD is constant (log 2), so the gradient is ~0 and there's no learning signal.
- **What is Earth Mover's / Wasserstein distance?** The minimum cost of moving probability mass to turn one distribution into another; it stays smooth even with no overlap.
- **Collapse vs vanishing gradient?** Vanishing gradient = the generator stops learning at all; collapse = it learns, but only one or a few modes.
- **Why not trust the loss?** GAN losses oscillate, and a collapsed model can still look fine — so we judge diversity with metrics, not the loss.
- **IS vs FID?** IS higher = sharp and diverse (but ignores real data); FID lower = closer to real data (the modern default, and it catches collapse).
- **Does WGAN fully solve collapse?** It greatly reduces it and is more stable, but it is not a guaranteed cure.

---

## Mini-glossary (plain words)

| Term | In plain words |
|------|----------------|
| Generator / Discriminator | The artist who makes fakes / the judge who spots fakes |
| Latent z | The random "idea seed" the generator starts from |
| Mode | One type/cluster in the data (one animal) |
| Mode coverage | How many types the model still makes (10/10 vs 1/10) |
| Mode collapse | All output piles onto one (or a few) modes |
| Complete vs partial | Only one type survives vs a few survive |
| Minimax game | The two networks fight over one score |
| Nash equilibrium | The balanced finish line where the judge guesses 50/50 |
| JSD | The standard GAN distance; goes flat when distributions don't overlap |
| Vanishing gradient | Flat loss → almost no signal to learn from |
| Wasserstein distance | "Cost of moving mass"; stays smooth, gives a usable gradient |
| 1-Lipschitz / WGAN-GP | The rule that keeps the WGAN critic well-behaved |
| IS / FID | Quality scores; IS higher = better, FID lower = better |
| Minibatch discrimination | Judge looks at the whole batch to catch sameness |
| Unrolled GAN | Generator plans ahead so it can't cheat with one mode |

---

## Validation notes (what we checked)

- **Content is technically accurate** across all 15 slides — definitions, the JSD/Wasserstein argument, IS/FID behaviour, and the structural fixes are all correct.
- **Timing**: 5 presenters, ~9 minutes total as printed on the Team & Flow slide.
- **Flag 1 — demo owner**: the live demo (slide 11) is assigned to **Disha** in this deck. Earlier plans had Rohit driving it. Please confirm who actually presses play.
- **Flag 2 — example numbers**: IS ~15.4 and FID ~8.2 are illustrative values, not measured on a stated dataset. Say "for example" when you mention them.
- **Small accuracy note**: "JSD = log 2 ≈ 0.693" on slide 9 is the maximum divergence for non-overlapping distributions — this is correct (it is separate from the GAN global optimum value of −log 4).
