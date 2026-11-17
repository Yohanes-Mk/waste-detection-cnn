# Genuinely unseen external images

Five images, **none from TACO, none used in training/validation/test-split selection or tuning** — sourced independently from Wikimedia Commons specifically for the Phase 5 qualitative "unseen data" check. Downloaded 2026-11-16.

| File | Source | License |
|---|---|---|
| `Litter.JPG` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Litter.JPG) | CC BY 3.0 / GFDL |
| `Plastic_Bottles_-_Waste_(7992944072).jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Plastic_Bottles_-_Waste_(7992944072).jpg) (orig. Flickr, Tony Webster) | CC BY 2.0 |
| `TRASH,_BLACK_PLASTIC_BAG.JPG` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:TRASH,_BLACK_PLASTIC_BAG.JPG) | CC BY-SA 4.0 |
| `Filthy_Habit_by_SillyPuttyEnemies.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Filthy_Habit_by_SillyPuttyEnemies.jpg) | CC BY-SA 3.0 |
| `Ditch_litter_5622.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ditch_litter_5622.jpg) | CC BY-SA 3.0 |

## Known distribution shift vs. TACO training data

- TACO photos are mostly single or few litter items, deliberately photographed against varied-but-uncluttered backgrounds.
- `Plastic_Bottles_-_Waste_...jpg` is a dense, repetitive close-up pile of bottles (a recycling-collection photo) — compositionally very different from any TACO training image.
- `Litter.JPG` (beach) and `Ditch_litter_5622.jpg` are heavily cluttered multi-object scenes with far more simultaneous litter items than typical TACO photos.
- These are qualitative-only checks (no ground-truth boxes exist for them) — kept separate from the metric-bearing locked TACO test-set evaluation (`outputs/test_metrics.json`).
