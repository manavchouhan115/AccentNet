# Accent Conversion

- **identity_encoder** – GE2E-based speaker embedding network.
- **mel_synthesizer** – Tacotron implementation that conditions on embeddings.
- **signal_vocoder** – WaveRNN vocoder for waveform reconstruction.
Fine-tuning two parallel stacks on accent-specific speech allows the system to keep the speaker identity while forcing the generated audio into the target accent.


1. **identity_encoder** ingests raw audio and produces normalized embeddings.
2. **mel_synthesizer** (Tacotron) conditions on embeddings + text to emit mel spectrograms.
3. **signal_vocoder** (WaveRNN) upsamples mels into high-quality waveforms.
