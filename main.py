import os, time
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
from compressor import SpatialCodec
import spatial_utils as utils
import visualization as viz

def run_project(input_file):
    print(f"--- Processing: {input_file} ---")
    data, fs = sf.read(input_file)
    if data.shape[1] < 4: raise ValueError("Input must be 4-channel Ambisonic")

    # 1. Processing
    start_time = time.time()
    codec = SpatialCodec(n_components=2)
    compressed = codec.compress(data)
    decompressed = codec.decompress(compressed)
    latency = (time.time() - start_time) * 1000
    binaural = utils.simple_binaural_render(decompressed)

    # 2. Rendering for Demo
    binaural_out = utils.simple_binaural_render(decompressed)
    # Save the binaural demo to the processed folder
    output_path = 'spatial_codec/data/processed/output_binaural.wav'
    sf.write(output_path, binaural_out, fs)

    # 3. Metrics & Terminal Output
    mag_o, az_o, el_o = utils.calculate_energy_vector(data)
    mag_c, az_c, el_c = utils.calculate_energy_vector(compressed)
    mag_d, az_d, el_d = utils.calculate_energy_vector(decompressed)
    
    orig_br = (fs * 16 * 4) / 1000
    comp_br = (fs * 16 * 2) / 1000 
    
    # Keeping terminal output identical
    print(f"\n--- Codec Metrics ---")
    print(f"Compression Ratio:    {codec.get_compression_ratio(data.shape):.1f}:1")
    print(f"Original Bitrate:   {orig_br:.1f} kbps")
    print(f"Compressed Bitrate: {comp_br:.1f} kbps")
    print(f"Decompressed Bitrate:      {comp_br:.1f} kbps")
    print(f"SNR:                  {utils.calculate_snr(data, decompressed):.2f} dB")
    print(f"Latency:              {latency:.2f} ms")
    print(f"Original Azimuth:     {np.degrees(az_o):.1f}°")
    print(f"Compressed Azimuth:   {np.degrees(az_c):.1f}°")
    print(f"Decompressed Azimuth: {np.degrees(az_d):.1f}°")

    # --- IMAGE 1: Waveforms & Difference Spectrograms ---
    labels = ['W (Omni)', 'X (Front)', 'Y (Left)', 'Z (Up)']
    path1 = 'spatial_codec/data/processed/output_image_1.png'
    viz.plot_waveforms_and_spectrograms(data, compressed, decompressed, fs, labels, path1)

    # --- IMAGE 2: Perceptual & Localization Analysis ---
    metrics = utils.get_3d_perceptual_metrics(data, compressed, decompressed, fs)
    binaural_cues = utils.calculate_moving_cues(binaural)
    spatial_data = (mag_o, az_o, mag_d, az_d, el_d)
    path2 = 'spatial_codec/data/processed/output_image_2.png'
    viz.plot_perceptual_analysis(metrics, binaural_cues, spatial_data, path2)

    print(f"--- Reports saved to spatial_codec/data/processed/ ---")

if __name__ == "__main__":
    run_project('spatial_codec/data/processed/music_aac_decoded.wav')