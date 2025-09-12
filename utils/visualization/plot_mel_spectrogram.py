import numpy as np
import matplotlib.pyplot as plt
import librosa


def make_mel_spec_image(mel_spec: np.ndarray, save_fpath: str):
    """
    Saves a Mel spectrogram image from a given Mel spectrogram array.

    Parameters
    ----------
    mel_spec: np.ndarray
        The Mel spectrogram (shape: [num_frames, num_mel_bins]).
    save_fpath: str
        Path to save the image file.
    """
    S = mel_spec.T  # transposed `mel_spec` to go from (T, n_mels) into (n_mels, T) shape

    fig = plt.figure(figsize=(8, 8), dpi=100)  # 8 inches * 100 dpi = 800 pixels

    # Display Mel spectrogram
    librosa.display.specshow(S, y_axis="mel", cmap="magma")

    plt.colorbar(label="dB")
    plt.xlabel("Time Frames")  # Show raw frame count
    plt.ylabel("Mel Frequency")  # N.B: it is in Mel scale --> Mel = 2595 * log10(1 + Hz / 700)
    plt.title(f"Mel Spectrogram ({S.shape[0]} Mels, {S.shape[1]} Frames)")

    # Save as 800x800 image
    plt.savefig(save_fpath, dpi=100, bbox_inches='tight', pad_inches=0.2)
    plt.close()

