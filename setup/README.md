# setup/

One-command box provisioning + baselines for flint, so spinning a fresh GPU box is never a fight again.

## Usage (on a fresh box)
```bash
# get the repo onto the box (git clone, or: tar czf - . | ssh box 'tar xzf - -C /root/flint')
bash setup/setup_box.sh                 # venv + torch + Granite-4.1-3b + gpt-fast  (~few min)
bash setup/baselines.sh 2.0e12          # measure B=1 baselines (use your GPU's peak-bw)
```

## peak-bw per GPU (HBM peak, for the MBU %)
| GPU | peak-bw |
|---|---|
| A100-40 | `1.55e12` |
| A100-80 / **H100-PCIe** | `2.0e12` |
| H100-SXM | `3.35e12` |
| H200 | `4.8e12` |

## Notes
- **Use a modern-driver box (>= 550 / CUDA 12.4+).** Then torch's default wheel + latest torchao just work.
- **Old driver (<= 535)?** the default torch wheel crashes at runtime (`CUDA driver too old`). Pin a matching cu wheel — see the NOTE at the bottom of `setup_box.sh` (`--index-url .../cu121` + `torchao==0.7.0`). This is the torch-version hell we hit on the cheap A100-40; avoid it by renting modern-driver boxes.
- gpt-fast is pinned to the **pre-FlexAttention** commit (SDPA) — it compiles cleanly on any torch 2.x, unlike the newer FlexAttention build.
