import numpy as np
import scipy.signal as sg
import scipy.stats
import librosa

from parselmouth.praat import call
from features_extractor import VoiceFeaturesExtractor

# TODO: complete the acquisition of cpp 

class SpectralExtractor(VoiceFeaturesExtractor):
    """    A class to extract spectral features from audio samples.
    Inherits from VoiceFeaturesExtractor.
    This class provides methods to compute alpha ratio, Hammarberg index,
    spectral slopes.
    """
    def __init__(self, audio_path: str, gender: str, time_step: float = 0.01, verbose: bool = True, **kwargs):
        """
        Initializes the SpectralExtractor by inheriting from VoiceFeaturesExtractor.
        """
        super().__init__(audio_path, gender, time_step=time_step, verbose=verbose, **kwargs)
        
    def get_alpha_ratio(self,
                        samples: np.ndarray = None,
                        sampling_rate: int = None,
                        band1: tuple = (50, 1000), 
                        band2: tuple = (1000, 5000)):
        """
        Compute the ratio of the summed energy from band1 to band2 in dB.
        :param band1: Frequency range for the first band (default is (50, 1000))
        :param band2: Frequency range for the second band (default is (1000, 5000))
        :param samples:  Audio samples to analyze (if None, will use the whole audio signal)
        :param sampling_rate: The sampling rate of the audio (if None, will use the sampling rate of the sound)
        :return: The alpha ratio in dB
        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        f, Pxx, _ = self.get_spectral_features(samples=samples, sampling_rate=sampling_rate)
        idx_band1 = np.where((f >= band1[0]) & (f <= band1[1]))
        idx_band2 = np.where((f >= band2[0]) & (f <= band2[1]))
        # sanity check
        if len(idx_band1[0]) == 0 or len(idx_band2[0]) == 0:
            if self.verbose:
                print("No frequencies found in the specified bands, returning NaN for alpha ratio.")
            return np.nan
        energy_low = np.sum(Pxx[idx_band1])
        energy_high = np.sum(Pxx[idx_band2])
        alpha_ratio = self.get_ratio_db(energy_low, energy_high, "power") 
        # Replace infinite values with NaN
        alpha_ratio = np.nan if np.isinf(alpha_ratio) else alpha_ratio 
        return alpha_ratio

    def get_cpp(self, 
                samples: np.ndarray = None,
                sampling_rate: int = None) -> float:
        """
        Compute Cepstral Peak Prominence (CPP) for a given signal (single-window).
        If no samples provided, uses the full audio signal from the instance.
        Returns the CPP in dB (aligned with Praat-style CPPS calculation: peak above linear baseline).
        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        frame = samples * sg.windows.hamming(len(samples), sym=False) # hamming window
        n_fft = 2 ** int(np.ceil(np.log2(len(frame)))) # zero-pad to next power-of-two for FFT efficiency
        SpecMat = np.abs(np.fft.fft(frame, n=n_fft)) # compute the magnitude of the FFT
        SpecdB = 20*np.log10(SpecMat) # convert to dB scale
        cep = 20*np.log10(np.abs(np.fft.fft(SpecdB, n=n_fft))) # compute the cepstrum (convolution of the spectrum with itself)
        # preferred: compute quefrency bounds in seconds then convert to sample indices
        qmin = int(np.round(sampling_rate / float(self.f0_max)))
        qmax = int(np.round(sampling_rate / float(self.f0_min)))
        #extract cepstral segment of interest and find peak
        cep_seg = cep[qmin:qmax + 1] 
        relative_peak_idx = int(np.argmax(cep_seg))
        peak_idx = qmin + relative_peak_idx # absolute index in cep
        c_peak = cep[peak_idx]
        # estimate baseline (linear regression) across the same cepstral window
        # fit line to cep[qmin:qmax] vs sample-index (or vs quefrency in seconds — baseline subtraction is invariant to scale)
        x = np.arange(qmin, qmax + 1)
        y = cep[qmin:qmax + 1]
        # linear fit: y = a*x + b
        a, b = np.polyfit(x, y, 1)
        c_base = np.polyval([a, b], peak_idx)  # baseline at the peak index
        # difference between cepstral peak and baseline
        cpp_db = float(c_peak - c_base)
        return cpp_db

    def get_hammarberg_index(self, 
                             samples: np.ndarray = None,
                             sampling_rate: int = None,
                             low_band: tuple = (0, 2000),
                             high_band: tuple = (2000,  5000),
                             energy_threshold: float = 1e-8):
        """
        Calculate the ratio of the peak energy in the low frequency range
        to the peak energy in the high frequency range.
        :param samples:  Audio samples to analyze (if None, will use the whole audio signal)
        :param sampling_rate: The sampling rate of the audio (if None, will use the sampling rate of the sound)
        :return: The Hammarberg index in dB
        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        f, Pxx, _ = self.get_spectral_features(samples=samples, sampling_rate=sampling_rate)
        # sanity check
        if len(f) == 0 or len(Pxx) == 0 or np.max(Pxx) < energy_threshold:
            if self.verbose:
                print("No frequencies or power spectral density values found (or too low), " \
                      "returning NaN for Hammarberg index.")
            return np.nan
        # find the indices for the low and high frequency bands
        idx_low = np.where((f >= low_band[0]) & (f <= low_band[1]))
        idx_high = np.where((f >= high_band[0]) & (f <= high_band[1]))
        # peak detection with threshold
        peak_low = np.max(Pxx[idx_low])
        peak_high = np.max(Pxx[idx_high])
        if peak_low < energy_threshold:
            if self.verbose:
                print(f"No energy in the {low_band[0]}-{low_band[1]} Hz range, returning NaN for Hammarberg index.")
            return np.nan
        if peak_high < energy_threshold:
            if self.verbose:
                print(f"No energy in the {high_band[0]}-{high_band[1]} Hz range, returning NaN for Hammarberg index.")
            return np.nan
        hammarberg_index = self.get_ratio_db(peak_low + 1e-10, peak_high + 1e-10, "power") 
        return hammarberg_index    
    
    def get_mfccs(self, mfcc_numbers: list = [1], n_mfcc: int = 13):
        """
        Extract selected MFCC coefficients over the full signal.
        :param mfcc_numbers: List of MFCC indices to extract (1-based, e.g. [1,2,3])
        :param n_mfcc: Total number of MFCCs to compute
        :return: Dict of MFCC values {"mfcc_1": value, ...}
        """
        if not all(1 <= n <= n_mfcc for n in mfcc_numbers):
            raise ValueError(f"Invalid MFCC numbers: {mfcc_numbers}. " \
                             f"Expected numbers in the range 1 to {n_mfcc}.")
        hop_length = int(self.time_step * self.sampling_rate)
        win_length = int(self.window_size * self.sampling_rate)
        n_fft = 2 ** int(np.ceil(np.log2(win_length)))
        mfccs = librosa.feature.mfcc(y=self.samples.astype(np.float32), sr=self.sampling_rate, 
                                    n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length,
                                    n_fft=n_fft, window='hamming', center=False)
        
        times_mfccs = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=self.sampling_rate, 
                                            hop_length=hop_length, n_fft=n_fft)
        return {f"mfcc_{n}": self.shift_series(serie=mfccs[n-1], timestamp=times_mfccs) for n in mfcc_numbers}
    
    def get_spectral_features(self, 
                              samples: np.ndarray = None, 
                              sampling_rate: int = None):
        """
        Computes the spectral features of the sound using the Welch method.
        :param samples:  Audio samples to analyze (if None, will use the whole audio signal)
        :param sampling_rate: The sampling rate of the audio (if None, will use the sampling rate of the sound)
        :return: Frequencies, Power spectral density (Pxx), and Pxx in dB.
        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        nperseg = min(len(samples), 512) 
        n_fft = 2 ** int(np.ceil(np.log2(len(samples)))) # zero-pad to next power-of-two for FFT efficiency
        f, Pxx = sg.welch(samples, fs=self.sampling_rate, nperseg=nperseg, nfft=n_fft, window='hamming')
        Pxx_dB = 10 * np.log10(Pxx + 1e-10)
        return f, Pxx, Pxx_dB

    def get_spectral_slopes(self,
                            samples: np.ndarray = None,
                            sampling_rate: int = None):
        """
        Computes the spectral slopes in the frequency ranges 0-500 Hz and 500-1500 Hz.
        :param samples:  Audio samples to analyze (if None, will use the whole audio signal)
        :param sampling_rate: The sampling rate of the audio (if None, will use the sampling rate of the sound)
        :return: The slopes in the two frequency ranges.
        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        f, __, Pxx_dB = self.get_spectral_features(samples, sampling_rate)
        if len(f) == 0 or len(Pxx_dB) == 0:
            if self.verbose:
                print("No frequencies or power spectral density values found, returning NaN for spectral slopes.")
            return np.nan, np.nan
        idx_0_500 = np.where((f >= 0) & (f <= 500))
        slope_0_500, _, _, _, _ = scipy.stats.linregress(f[idx_0_500], Pxx_dB[idx_0_500])
        idx_500_1500 = np.where((f >= 500) & (f <= 1500))
        slope_500_1500, _, _, _, _ = scipy.stats.linregress(f[idx_500_1500], Pxx_dB[idx_500_1500])
        return slope_0_500, slope_500_1500