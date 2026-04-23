import matplotlib.pyplot as plt
import numpy as np

def plot_waveforms_and_spectrograms(data, compressed, decompressed, fs, labels, save_path):
    """Generates the 4x2 waveform and spectrogram grid."""
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))

    for i in range(4):
        # Column 1: Waveforms
        axs[i, 0].plot(data[:1000, i], label='Original', alpha=0.4, color='blue')
        comp_idx = 0 if i < 2 else 1
        axs[i, 0].plot(compressed[:1000, comp_idx], label=f'Compressed (PC{comp_idx+1})', color='green', alpha=0.6, linewidth=1)   
        axs[i, 0].plot(decompressed[:1000, i], label='Decompressed', color='orange', alpha=0.8)

        axs[i, 0].set_title(f"Channel {labels[i]} Waveform Transformation")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].legend(loc='upper right')

        # Column 2: Difference Spectrograms
        diff = data[:, i] - decompressed[:, i]
        axs[i, 1].specgram(diff + 1e-10, Fs=fs, NFFT=1024, cmap='magma')
        axs[i, 1].set_title(f"Channel {labels[i]} Difference Spectrogram")
        axs[i, 1].set_xlabel("Time (s)")
        axs[i, 1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig) # Close to free up memory

def plot_perceptual_analysis(metrics, binaural_cues, spatial_data, save_path):
    """Generates the 2x2 perceptual and localization analysis grid."""
    # Unpack metrics
    o_e, c_e, d_e, freqs, psd, thresh = metrics
    itds, ilds = binaural_cues
    mag_o, az_o, mag_d, az_d, el_d = spatial_data

    fig = plt.figure(figsize=(16, 14))

    # 1. Masking Threshold
    ax_m = fig.add_subplot(2, 2, 1)
    ax_m.semilogy(freqs, psd + 1e-10, label='Signal PSD')
    ax_m.semilogy(freqs, thresh + 1e-10, 'r:', label='Masking Threshold')
    ax_m.set_title("Directional Masking Analysis")
    ax_m.set_xlabel("Frequency (Hz)"); ax_m.set_ylabel("Power/Freq (dB/Hz)"); ax_m.legend()

    # 2. Total Acoustic Energy
    ax_e = fig.add_subplot(2, 2, 2)
    ax_e.plot(o_e[:1500], label='Original', alpha=0.6)
    ax_e.plot(c_e[:1500], label='Compressed (Scaled)', alpha=0.8)
    ax_e.plot(d_e[:1500], label='Decompressed', alpha=0.6)
    ax_e.set_title("Total Acoustic Energy Path (L2 Norm)")
    ax_e.set_xlabel("Time (Samples)"); ax_e.set_ylabel("Energy Magnitude"); ax_e.legend()

    # 3. ITD & ILD
    ax_c = fig.add_subplot(2, 2, 3)
    jitter = np.random.normal(0, 1, size=len(itds))
    angles = np.full_like(itds, np.degrees(az_d)) + jitter
    ax_c.scatter(angles, itds, color='blue', alpha=0.4, label='ITD (s)')
    ax_c.scatter(angles, ilds, color='green', alpha=0.4, label='ILD (dB)')
    ax_c.set_xlim([-180, 180])
    ax_c.set_xlabel("Decompressed Azimuth (Degrees)"); ax_c.set_ylabel("Cue Value")
    ax_c.set_title("Binaural Cue Distribution (HRTF-Style)"); ax_c.legend()

    # 4. 3D Polar Vector
    ax_p = fig.add_subplot(2, 2, 4, projection='polar')
    ax_p.annotate('', xy=(az_o, mag_o), xytext=(0,0), arrowprops=dict(facecolor='blue', width=2))
    ax_p.annotate('', xy=(az_d, mag_d), xytext=(0,0), arrowprops=dict(facecolor='red', alpha=0.5, width=2))
    ax_p.text(az_o, mag_o + 0.15, 'Original', color='blue', ha='center', va='center', weight='bold')
    ax_p.text(az_d + 0.2, mag_d + 0.2, 'Decompressed', color='red', ha='right', va='center', weight='bold')
    ax_p.set_title(f"Spatial Localization (Elevation: {np.degrees(el_d):.1f}°)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)