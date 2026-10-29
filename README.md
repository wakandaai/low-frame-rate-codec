# Probing Low Frame Rate Degradation in Neural Audio Codecs

This repository contains the code for the paper **"Probing Low Frame Rate
Degradation in Neural Audio Codecs"** (Interspeech 2026), by Alex Gichamba and
Moise Busogi, Carnegie Mellon University Africa.

📄 Paper: TODO add arXiv / proceedings link &nbsp;|&nbsp;
🔊 Project page & audio samples: https://wakandaai.github.io/low-frame-rate-codec/

It is a fork of the [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec)
and retains DAC's MIT license. Our changes add a controlled frame-rate ablation
on top of the original training framework; the underlying codec architecture is
unchanged.

## Summary

Low frame rates make autoregressive speech synthesis cheaper, since generation
cost scales with sequence length, but codecs have been reported to collapse
below ~12.5 Hz. We ask whether that collapse is intrinsic to the frame rate or
an artifact of how codecs are trained.

We find it is the latter. The standard recipe crops training clips to a fixed
*duration* (0.38 s), which at low frame rates yields only a handful of tokens
per clip (as few as 2 tokens at 6.25 Hz), so the decoder never learns to model
audio across token boundaries. Fixing the token *count* instead removes the
6.25 Hz cliff entirely (WER 107% → 15%) and keeps speech intelligible down to
1.6 Hz at 192 bps, with no architectural changes. Phonemic collision and
codebook saturation are ruled out as fundamental barriers.

## Relationship to upstream DAC

All credit for the codec architecture, losses, and training framework goes to
the original authors of DAC (Kumar et al., NeurIPS 2023). This fork modifies
only the experimental configuration to sweep frame rate and training clip
length. The original DAC README content (install, encode/decode, Docker, etc.)
still applies and is preserved below the citation section. TODO: keep or trim
the upstream usage docs as you prefer.

## Installation

```bash
pip install -e ".[dev]"
```

See the original DAC instructions further down for Docker-based setup.

## Data

All ablation models are trained on **LibriSpeech `train-clean-100`** at 16 kHz
and evaluated on `test-clean`. The data configuration is in
`conf/librispeech_16khz.yaml`.

## Reproducing the frame-rate ablation

Frame rate is set by the encoder stride product, which must satisfy
`prod(strides) = fs / fr` with `fs = 16000`. Each variant changes the per-block
strides while holding everything else fixed (`n_q = 12`, `|V| = 1024`, so
`R = 120 * fr` bps).

The two training regimes in the paper differ only in how clips are cropped, and
are distinguished by config filename in `conf/new_ablations/frame_rate/`:

- **Fixed clip duration** (`T_clip = 0.38 s`, the DAC default) — files named
  `librispeech_16khz_<fr>hz.yaml`. Reproduces the quality cliff; token count `K`
  shrinks as frame rate drops.
- **Fixed token count** (`K = 19`, i.e. `T_clip = 19 / fr`) — files named
  `librispeech_16khz_<fr>hz_extended_duration.yaml`. Removes the cliff.

The ultra-low rates (3.125 Hz, 1.6 Hz) are provided only in the
`_extended_duration` form, since they appear only in the matched-token results.

| Frame rate | Fixed `T_clip` (cliff) | Fixed `K` (fix) |
|---|---|---|
| 100 Hz   | `librispeech_16khz_100hz.yaml`   | `librispeech_16khz_100hz_extended_duration.yaml`   |
| 25 Hz    | `librispeech_16khz_25hz.yaml`    | `librispeech_16khz_25hz_extended_duration.yaml`    |
| 12.5 Hz  | `librispeech_16khz_12-5hz.yaml`  | `librispeech_16khz_12-5hz_extended_duration.yaml`  |
| 6.25 Hz  | `librispeech_16khz_6-25hz.yaml`  | `librispeech_16khz_6-25hz_extended_duration.yaml`  |
| 3.125 Hz | —                                | `librispeech_16khz_3-125hz_extended_duration.yaml` |
| 1.6 Hz   | —                                | `librispeech_16khz_1-6hz_extended_duration.yaml`   |

Train a single variant (single GPU):

```bash
export CUDA_VISIBLE_DEVICES=0
# 6.25 Hz, fixed clip duration -> reproduces the cliff
python scripts/train.py \
    --args.load conf/new_ablations/frame_rate/librispeech_16khz_6-25hz.yaml \
    --save_path runs/6-25hz/

# 6.25 Hz, fixed token count -> recovers performance
python scripts/train.py \
    --args.load conf/new_ablations/frame_rate/librispeech_16khz_6-25hz_extended_duration.yaml \
    --save_path runs/6-25hz_extended/
```

Notes:
- The 50 Hz DAC baseline (marked † in the paper tables) is the standard DAC
  configuration. TODO: confirm which config file you used for it (it is not in
  the `frame_rate/` folder above).
- A `librispeech_16khz_6-25hz_v2.yaml` is also present. TODO: note whether the
  reported 6.25 Hz numbers come from this or the base config.
- `run.sh`, `run2.sh`, `run3.sh` are the SLURM/launch wrappers used on Bridges2.

## Evaluation

Reconstruction quality is measured with five metrics:

- **WER** — transcribe reconstructions with MMS-1B and score against references
- **STOI** — short-time objective intelligibility
- **MCD** — mel-cepstral distortion
- **SPK-SIM** — speaker-embedding cosine similarity (ESPnet-SPK / WavLM)
- **UTMOS** — reference-free MOS predictor

TODO: add the command(s) to run your evaluation pipeline, e.g.
`python scripts/evaluate.py --ckpt runs/<variant>/ ...`, and note any external
dependencies (MMS-1B, UTMOS, ESPnet-SPK).

## Citation

```bibtex
@inproceedings{gichamba2026probing,
  title     = {Probing Low Frame Rate Degradation in Neural Audio Codecs},
  author    = {Gichamba, Alex and Busogi, Moise},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
  address   = {Sydney, Australia}
}
```

## Acknowledgments

This work is part of the African Engineering and Technology Network (Afretec),
managed by Carnegie Mellon University Africa with financial support from the
Mastercard Foundation. It used Bridges2 at the Pittsburgh Supercomputing Center
through allocation CIS250700 from the ACCESS program.

## License

MIT, inherited from the upstream Descript Audio Codec. See [LICENSE](LICENSE).

---

<!-- TODO: the original DAC README content (Usage / Weights / Docker / Training
/ Testing / Results) can be pasted below this line if you want to preserve it. -->