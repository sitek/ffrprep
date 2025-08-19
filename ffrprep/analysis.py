import numpy as np
import mne


def compute_power(avg_evoked, f_low=90, f_high=110, t_low=0.1, t_high=0.2):
    # Compute the spectrogram (time-frequency representation)
    power = mne.time_frequency.tfr_multitaper(avg_evoked,
                                              freqs=np.arange
                                              (f_low, f_high, 1),
                                              n_cycles=2,
                                              time_bandwidth=4.0,
                                              average=True,
                                              return_itc=False)

    # Select the time and frequency bands
    time_mask = (power.times >= t_low) & (power.times <= t_high)
    freq_mask = (power.freqs >= f_low) & (power.freqs <= f_high)

    # Extract the power within the specified bands
    selected_power = power.data[:, freq_mask, :][:, :, time_mask]

    # Average power over the selected time and frequency bands
    average_power = selected_power.mean(axis=2).mean(axis=1)[0]
    return average_power

def rms_snr():
    
