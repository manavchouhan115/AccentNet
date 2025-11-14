# Accent Conversion

<img width="871" height="421" alt="Base_Architecture" src="https://github.com/user-attachments/assets/c5025bfe-7da8-4ca8-befb-7c555611e881" />

1. **identity_encoder** ingests raw audio and produces normalized embeddings.
2. **STT** uses Whisper to convert the input speech to text
3. **mel_synthesizer** (Tacotron) conditions on embeddings + text to emit mel spectrograms.
4. **signal_vocoder** (WaveRNN) upsamples mels into high-quality waveforms.
