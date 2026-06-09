# Mode Collapse in GANs — Presentation Kit

A complete kit for a Deep Learning group presentation on **mode collapse in GANs**: an interactive live demo, a presenter runbook, a Q&A guide, and full study notes.

**Topic:** Mode Collapse in GANs (Deep Learning)
**Team:** Jeenal · Rohit · Sharvan · Disha · Pujan

---

## What's inside

```
gan-mode-collapse-demo/
├── README.md                          <- you are here
├── demo/
│   └── mode_collapse_live_demo.html   <- interactive live demo (open in a browser)
├── presentation/
│   ├── presenter_runbook.html         <- who says what, demo cues, timing (print -> PDF)
│   ├── presentation_plan.md           <- task split + section content
│   └── live_demo_and_qna_guide.md     <- every control explained + Q&A bank
├── notes/
│   ├── DL_GANs_Theory.md              <- GANs from scratch to mastery
│   ├── DL_GANs_Numericals.md          <- worked numerical problems
│   ├── DL_GANs_Practice.md            <- runnable PyTorch (simple GAN + DCGAN)
│   └── mode_collapse_addendum.md      <- applications, differentiator, mechanism visual
└── assets/
    └── mode_collapse_mechanism.svg    <- the dominating "how it forms" diagram
```

---

## The live demo

Open `demo/mode_collapse_live_demo.html` in any browser (works offline). The grid shows ten animals the generator should produce; on a **Collapse run** you watch them all turn into a cat while **mode coverage** falls from 10/10 to 1/10, and **Reveal the fix** brings the variety back.

| Control | Key |
|---|---|
| Play / pause training | `space` |
| Reveal the fix | `F` |
| Healthy vs Collapse run | `H` / `C` |
| Scrub epoch | `← →` |
| Reset | `R` |

The ten animals: dog, **cat (collapse target)**, mouse, rabbit, fox, bear, panda, bird, frog, monkey.

> The demo is a faithful simulation of GAN training dynamics so it runs anywhere. The real, runnable PyTorch version is in `notes/DL_GANs_Practice.md`.

---

## How to present

1. Print `presentation/presenter_runbook.html` to PDF and give each person their card.
2. Open the live demo in a separate browser tab, full-screen.
3. Follow the slide ↔ demo map in the runbook: Disha plays the collapse during detection, Pujan reveals the fix.
4. Keep a 15-second screen recording of a Play → collapse → fix run as a backup.

---

## Study notes

Start with `notes/DL_GANs_Theory.md` (cross-linked to the numericals and practice files) for the full GANs deep-dive, including the mode-collapse mechanism.

---

*Class project — for educational use.*
