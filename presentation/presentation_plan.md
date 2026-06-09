# Group Presentation Plan — Mode Collapse in GANs (Deep Learning)

**Team (5):** Jeenal · Rohit · Sharvan · Disha · Pujan
**Format:** 5–10 min talk + Q&A | ~8–9 slides | one section per person

---

## 1. The story we are telling (in one breath)

A GAN has two networks fighting: a **Generator** that makes fakes and a **Discriminator** that catches fakes. Mode collapse is when the Generator gets *lazy* — instead of learning to make every kind of real thing, it finds **one** thing that fools the judge and makes that same thing again and again. It "wins" by being narrow, not by being good.

We want the audience to leave with three things: **what** it is, **why** it happens, **how** we fix it.

---

## 2. Who does what (task split)

| # | Member | Section | Slides | Talk time |
|---|--------|---------|--------|-----------|
| 1 | **Jeenal** | Hook + 1-slide GAN recap (G vs D) + set up the problem | 1–2 | ~1.5 min |
| 2 | **Rohit** | What mode collapse IS + concrete example (the picture) | 3–4 | ~2 min |
| 3 | **Sharvan** | WHY it happens + complete vs partial collapse | 5–6 | ~2 min |
| 4 | **Disha** | How to DETECT / diagnose it | 7 | ~1.5 min |
| 5 | **Pujan** | How to FIX it (toolbox) + key takeaway, then leads Q&A | 8–9 | ~2 min |

**Shared jobs (assign now):**
- **Deck builder:** one person owns the master slide file so styling stays consistent (suggest Rohit or Disha).
- **Timekeeper:** signals at the 8-min mark so we land inside 10 min.
- **Q&A:** everyone preps answers for their own section; Pujan fields the first question and routes others to the right person.

---

## 3. Prep timeline (D = presentation day)

| When | What | Owner |
|------|------|-------|
| D-5 | Each person drafts their section's bullet points + 1 visual | All |
| D-3 | Merge into one deck, fix flow & timing | Deck builder |
| D-2 | Full run-through #1 (time it), trim to fit 10 min | All |
| D-1 | Run-through #2 + Q&A drill (ask each other the questions below) | All |
| D | Arrive early, load deck, decide speaking order | All |

---

## 4. The content, section by section

### Section 1 — Jeenal: GAN recap + the problem (slides 1–2)

A GAN is two networks playing a game:
- **Generator (G):** takes random noise, turns it into a fake sample (e.g. a digit image).
- **Discriminator (D):** looks at a sample and says "real" or "fake".

They train by competing: G tries to fool D, D tries not to be fooled. When training goes well, G learns to produce the **full variety** of real data.

```
   noise --> [ Generator ] --> fake --\
                                        > [ Discriminator ] --> real? / fake?
   real data ------------------------>/
```

**The catch (hand off to Rohit):** sometimes G stops producing variety and gets stuck on a few outputs. That failure has a name.

---

### Section 2 — Rohit: What mode collapse IS (slides 3–4)

A "mode" = one distinct type in the data. In handwritten digits there are **10 modes** (0,1,2,...,9). A healthy generator produces all 10. A collapsed generator produces only one or two.

**The plain version:** imagine a student who must draw many different animals to pass. They notice the examiner always accepts a great drawing of a cat — so they just draw a cat every single time. They pass, but they never actually learned to draw the other animals. The generator does the same: it finds one output that reliably fools the judge and repeats it.

```
   What we WANT (covers all modes):     What collapse gives (one mode):

   0 1 2 3 4 5 6 7 8 9                   3 3 3 3 3 3 3 3 3 3
   (all digits appear)                   (only one digit, over and over)
```

So mode collapse = **the generator produces very little variety**; many different noise inputs map to nearly the same output.

---

### Section 3 — Sharvan: Why it happens (slides 5–6)

Three reasons, simply:

1. **G only cares about fooling D right now.** Nothing in the basic setup *rewards* variety. If one output fools the judge, G has no reason to make anything else.
2. **The cat-and-mouse loop.** G locks onto one output. D learns to reject it. So G jumps to *another* single output. D adapts again. G keeps hopping between a few outputs but never covers them all.
3. **D judges one sample at a time.** It can't notice "all of these are identical," so it can't directly punish lack of variety.

**Two flavours:**
- **Complete collapse:** G produces basically one output.
- **Partial collapse:** G produces a few types but misses many.

```
   real data has 4 clusters:   /\   /\   /\   /\
   collapsed generator covers:           /\          (misses the other three)
```

---

### Section 4 — Disha: How to detect it (slide 7)

Signs that mode collapse is happening:
- **Eyeball the samples** — they all look the same or come in very few varieties.
- **Count the classes** — for labelled data, check whether all categories show up (should see all 10 digits, not just one).
- **Walk the noise** — change the input noise a lot; if the output barely changes, that's collapse.
- **Metrics** — low *recall* (it misses real modes), high **FID** that won't drop, low diversity score.

One line for the slide: **"Same output for very different inputs = collapse."**

---

### Section 5 — Pujan: How to fix it + takeaway (slides 8–9)

The toolbox (pick a couple to mention; don't list all):
- **Minibatch discrimination** — let D look at a whole batch at once so it *can* spot "these are all identical" and punish it. Pushes G to diversify.
- **Feature matching** — G aims to match the statistics of real data, not just fool D.
- **Unrolled GAN** — G "looks ahead" at how D will react, so it can't cheaply exploit the current D.
- **WGAN / WGAN-GP** — a better distance measure gives smoother gradients and is more robust to collapse.
- **PacGAN** — D sees several samples packed together, so it notices missing variety.

**Key takeaway slide:** "Mode collapse = the generator trades variety for an easy win. We catch it by checking output diversity, and we fix it by forcing the generator to cover all the modes."

---

## 5. Mnemonics (optional, for the speakers)

- **Cause, in 3 words:** *"narrow beats broad"* — G wins by being narrow.
- **Fixes — "FUM-WP":** **F**eature matching, **U**nrolled GAN, **M**inibatch discrimination, **W**GAN-GP, **P**acGAN.

---

## 6. Q&A bank (drill these before the presentation)

| Likely question | Short answer |
|---|---|
| What exactly is a "mode"? | A distinct type/cluster in the data — e.g. each digit 0–9 is one mode. |
| Mode collapse vs overfitting? | Overfitting = memorising training examples. Collapse = lack of *output variety*; many inputs give the same output. Different problems. |
| Mode collapse vs vanishing gradient? | Vanishing gradient = G stops learning because D got too strong. Collapse = G *does* learn, but only narrow outputs. Different failure modes. |
| Why doesn't the discriminator just fix it? | D judges one sample at a time, so it can't see that all outputs are identical — unless we give it batch info (minibatch discrimination). |
| Does WGAN completely solve it? | It reduces it and is more robust, but it's not a guaranteed cure. |
| How do you measure diversity? | Number of modes covered, recall (precision-recall for GANs), FID (lower better), diversity of samples. |
| Real-world consequence? | A face generator that makes only one kind of face; data augmentation that misses rare but important cases. |
| Simplest fix to try first? | Minibatch discrimination, or switch the loss to WGAN-GP. |
| Is partial collapse always bad? | Usually yes — the model is unreliable because it ignores parts of the data it was meant to learn. |

**If someone doesn't know an answer:** acknowledge it, give the closest correct idea, and offer to follow up — don't guess wildly. ("Good question — the core reason is X; the exact numbers we'd need to check.")

---

## 7. Quick checklist before going live

- [ ] All 10 modes story / picture on a slide (it's the clearest visual)
- [ ] Deck under 9 slides, timed under 10 minutes
- [ ] One visual per section, minimal text
- [ ] Each speaker knows their cue and hand-off line
- [ ] Q&A bank reviewed by everyone
