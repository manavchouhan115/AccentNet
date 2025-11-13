# AccentNet — Neural Accent Conversion

AccentNet is an end-to-end accent conversion system that separates *what is being said* from *how it is spoken*. By disentangling linguistic content, speaker identity, prosody, and accent-specific cues, the project converts a source utterance into a target accent while preserving the original speaker’s voice and rhythm.

The work is motivated by the communication barriers: multilingual regions such as India or Singapore frequently experience comprehension gaps caused by accent diversity, and curated parallel data covering multiple accents is scarce. AccentNet addresses the problem by fusing transfer learning for robust encoders with a FastSpeech2-style acoustic decoder and a neural vocoder.

## High-Level Architecture

The diagram in `AccentNet_Architecture.png` summarizes the pipeline:

!['AccentNet_Architecture'](AccentNet_Architecture.png)

1. **Content Encoder** extracts linguistic features (ContentVec-sized 768-d vectors per frame).
2. **Prosody Extractor** captures F0, energy, and duration cues from the source utterance.
3. **Speaker Encoder** (SpeechBrain ECAPA-TDNN) produces a 192-d vector that preserves the original voice.
4. **Accent Encoder** (Conformer + adversarial gradient reversal) learns speaker-invariant 256-d accent embeddings from a separate pool of target-accent speech.
5. **Acoustic Decoder** (`Decoder/FastSpeech2Accent`) fuses the embeddings and prosody features to predict target-accent mel spectrograms.
6. **Vocoder** (SpeechBrain HiFi-GAN) synthesizes the final waveform.

## Repository Layout

```
AccentEncoder/         # Standalone Conformer-based accent encoder + data utilities
ContentEncoder/        # Notebooks and helpers for ContentVec feature prep
Decoder/               # FastSpeech2Accent model, datasets, configs, training & inference scripts
Prosody/               # Prosody encoder + extraction helpers (PyWorld + librosa)
SpeakerEncoder/        # SpeechBrain ECAPA-TDNN wrapper for speaker embeddings
AccentNet_Architecture.png
```


## Data Sources & Preparation

Datasets emphasized in the SMC report and scripts:

- **VCTK** (British/Indian English speakers, 44h)
- **L2-ARCTIC** (non-native accents with transcript pairs)
- **AccentDB Extended** (four Indian-English accent groups)

The target is to obtain accent-balanced splits where the same transcript can be aligned across accents whenever possible.

### Build the master manifest

Inside `AccentEncoder`:

```bash
python scripts/prepare_manifest.py \
  --root /path/to/data_root \
  --output data/manifest.csv
```

`prepare_manifest.py` records utterance IDs, speaker IDs, accent labels, split assignments, transcript text, and relative WAV paths spanning VCTK, L2-ARCTIC, and AccentDB. Split strategy is speaker-based (train/val/test sets are disjoint).

### Audio normalization and mel features

1. (Optional) `scripts/resample_audio.py` to create a 16 kHz workspace (`data/audio16`).
2. `scripts/extract_mels.py` caches log-mel spectrogram tensors per utterance:

```bash
python scripts/extract_mels.py \
  --manifest data/manifest.csv \
  --audio-root data/audio16 \
  --output-dir data/features/mels
```

The cached tensors live under `data/features/mels/{split}/{utt_id}.pt` and are used throughout the project.

## Component Details

### Accent Encoder (`AccentEncoder/accent_encoder`)

- **Backbone:** 4 Conformer blocks with depthwise convolution, self-attention, and feed-forward stacks.
- **Pooling:** Multi-head self-attentive statistics pooling → 256-d normalized embedding.
- **Heads:** Accent classifier and adversarial speaker classifier using a Gradient Reversal Layer to promote speaker invariance.
- **Training:** `scripts/train_accent_encoder.py` wires up dataloaders, TensorBoard logging, cosine-warmup LR scheduling, and checkpointing.

Example training run:

```bash
python scripts/train_accent_encoder.py \
  --manifest data/manifest.csv \
  --features data/features/mels \
  --batch-size 32 --total-steps 20000
```

After training, `scripts/extract_embeddings.py` (and related helpers such as `compute_centroids.py`) export accent embeddings for each utterance to be consumed by the decoder.

### Content Encoder (`ContentEncoder/`)

`content_encoder_utils.py` contains utilities for scanning accent-specific directories, constructing balanced train/val/test splits, normalizing audio to 16 kHz, and converting transformer embeddings (e.g., ContentVec/HuBERT derivatives) into pooled or frame-level datasets. Notebooks (`accentNet.ipynb`, `content_encoder_analysis.ipynb`) document experiments on accent discrimination and embedding quality.

### Speaker Encoder (`SpeakerEncoder/`)

`speechEncoder.py` wraps SpeechBrain’s `spkrec-ecapa-voxceleb` model:

- Ensures mono 16 kHz input using Torchaudio.
- Returns 192-d embeddings per utterance.
- Includes helpers for batch processing file lists and computing per-speaker centroids.

`SpeakerEncoder/speaker_similarity_speechbrain_ecapa.ipynb` evaluates cosine similarity matrices to verify intra-speaker clustering (see PDF slides).

### Prosody Encoder (`Prosody/`)

- `prosody_encoder.py` defines a CNN + BiGRU network that transforms concatenated mel/F0/energy features into frame-level and global prosody embeddings.
- `extract_features` uses PyWorld (Harvest + Stonemask) and librosa to compute:
  - Log-mel spectrograms.
  - Log-normalized F0 contours with gap interpolation.
  - Log energy traces.
- `test_prosody_extraction.ipynb` validates synchronization against content embeddings (5 ms hop resolution, truncated tails, etc.).

### Acoustic Decoder & Vocoder (`Decoder/`)

`FastSpeech2Accent` adapts FastSpeech2 to accept:

- Content embeddings (`cv.npy`), broadcast speaker embeddings (`spk.npy`), and target accent embeddings (`accent.pt`).
- Source prosody features (pitch, energy, duration) injected via `AccentVarianceAdaptor`.
- Outputs 80-bin mel spectrograms and refined mel predictions via a PostNet.

Supporting modules:

- `dataset.py` for legacy metadata-driven pairing.
- `manifest_dataset.py` for JSONL manifests aligning source and target entries at the utterance level (handles padding, duration quantization, optional mel targets).
- `multi_encoder_input.py` projects concatenated embeddings to the model hidden size.
- `variance_adaptor.py` handles duration expansion and pitch/energy quantization.
- `utils.py` provides a padded collate function for variable-length batches.

The YAML config at `Decoder/config/model.yaml` captures default dimensions and training hyperparameters.

## End-to-End Workflow

1. **Collect & Organize Audio**
   - Download VCTK, L2-ARCTIC, AccentDB Extended.
   - Mirror the directory structure expected by `prepare_manifest.py`.

2. **Build Manifest & Mel Cache**
   - Run `prepare_manifest.py`, `extract_mels.py`, and (if needed) `resample_audio.py`.

3. **Train Encoders**
   - Content encoder experiments live in the notebooks; export embeddings as `cv.npy`.
   - Run `SpeakerEncoder/speechEncoder.py` (or the notebook) to generate `spk.npy`.
   - Use `Prosody/prosody_encoder.py` (or the companion notebook) to save per-frame pitch/energy/duration statistics (`lf0.npy`, `lf0i.npy`, `duration.npy`).
   - Train the accent encoder and export target-accent embeddings (`accent.pt`).

4. **Assemble Embedding Manifests**
   - Arrange each utterance directory with files `{cv.npy, spk.npy, accent.pt, lf0.npy, lf0i.npy, duration.npy, mel.npy}` under a root such as `Decoder/embeddings_clean`.
   - Create a JSONL manifest listing metadata and relative paths.

5. **Train the Acoustic Decoder**
   - With metadata files `data/train.txt` / `data/val.txt` (or explicit utterance lists), run:

```bash
python Decoder/train_accent.py \
  --config Decoder/config/model.yaml \
  --manifest Decoder/embeddings_clean/manifest.jsonl \
  --embeddings_dir Decoder/embeddings_clean \
  --source_accent indian --target_accent english \
  --epochs 100 --batch_size 16
```

   - The script supports manifest mode (recommended) or legacy metadata triplets.

6. **Inference & Waveform Synthesis**

```bash
python Decoder/convert_accent.py \
  --config Decoder/config/model.yaml \
  --checkpoint Decoder/checkpoints/model_epoch_XX.pt \
  --manifest Decoder/embeddings_clean/manifest.jsonl \
  --embeddings_root Decoder/embeddings_clean \
  --source_accent indian --target_accent english \
  --source_speaker accentdb_telugu_s01 --target_speaker p256 \
  --utterance_id 135 \
  --output_dir Decoder/outputs/inference \
  --save_mel
```

The script fetches matching source/target embeddings, runs the decoder, and synthesizes audio with SpeechBrain’s HiFi-GAN (default `speechbrain/tts-hifigan-ljspeech`).

## Environment & Dependencies

- Python ≥ 3.9, PyTorch ≥ 2.0.
- Accent encoder requirements: see `AccentEncoder/requirements.txt`.
- Prosody & decoder stages additionally use `librosa`, `pyworld`, `soundfile`, `speechbrain`, and `torchaudio`.
- Install extras as needed, e.g.:

```bash
pip install -r AccentEncoder/requirements.txt
pip install speechbrain librosa pyworld soundfile
```

GPU acceleration is highly recommended for the Conformer encoder and FastSpeech2 decoder training loops.

