import numpy as np
import scipy.signal as sg
from features_extractor import VoiceFeaturesExtractor, call

class FrequencyExtractor(VoiceFeaturesExtractor):
    """
    A class to extract frequency-related features from audio samples.
    Inherits from VoiceFeaturesExtractor.
    This class provides methods to extract formants, fundamental frequency (F0),
    harmonic differences.
    """
    def __init__(self, audio_path: str, gender: str, time_step: float = 0.01, verbose: bool = True, **kwargs):
        """
        Initializes the FrequencyExtractor by inheriting from VoiceFeaturesExtractor.
        """
        super().__init__(audio_path, gender, time_step=time_step, verbose=verbose, **kwargs)
        self.formant_times = None
        self.pitch_times = None
        
    def get_formant(self, 
                    formant_number: int,
                    max_number_of_formants: int = 5,
                    maximum_formant: float = 5500.0, 
                    interpolate: bool = True,
                    shift_timestamp: bool = True) -> np.ndarray:   
        """
        Extracts formant values from the sound.
        :param formant_number: The formant number to extract (1, 2, 3 or 4)
        :param max_number_of_formants: Maximum number of formants to extract (default is 5)
        :param maximum_formant: Maximum frequency for the formants (default is 5500 Hz)
        :param interpolate: If True, NaN values will be interpolated using linear interpolation.
        :param shift_timestamp: If True, the formant values will be shifted to match the
                               global timestamp of the frames.

        :return: A numpy array of formant values at each time step.
        """
        if formant_number not in [1, 2, 3, 4]:
            raise ValueError("Formant number must be between 1 and 4.")     
        formant = self.sound.to_formant_burg(self.time_step, max_number_of_formants, maximum_formant)
        self.formant_times = formant.xs()
        values = [formant.get_value_at_time(formant_number, t) for t in self.formant_times]
        values = np.array(values)
        values[values == 0] = np.nan
        if interpolate:
            # Interpolate NaN values using linear interpolation
            values = np.interp(self.formant_times, self.formant_times[~np.isnan(values)], values[~np.isnan(values)])
        if shift_timestamp:
            if interpolate is False:
                raise ValueError("Interpolation must be enabled to shift the timestamp.")
            # Shift the formant values to the same global timestamp as the frames
            values = self.shift_series(values, self.formant_times)
        return values
    
    def get_fundamental_frequency(self, 
                                  interpolate: bool = True, 
                                  shift_timestamp: bool = True) -> np.ndarray:
        """
        Extracts the fundamental frequency (F0) from the sound, also known as pitch.
        :param interpolate: If True, NaN values will be interpolated using linear interpolation.
        :param shift_timestamp: If True, the F0 values will be shifted to match the
                               global timestamp of the frames.
        :return: A numpy array of F0 values at each time step.
        """
        pitch = self.sound.to_pitch(self.time_step, self.f0_min, self.f0_max)
        self.pitch_times = pitch.xs()
        f0_values = [pitch.get_value_at_time(t) for t in self.pitch_times]
        f0_values = np.array(f0_values)
        f0_values[f0_values == 0] = np.nan
        if interpolate:
            f0_values = np.interp(self.pitch_times, self.pitch_times[~np.isnan(f0_values)], f0_values[~np.isnan(f0_values)])
        if shift_timestamp:
            if interpolate is False:
                raise ValueError("Interpolation must be enabled to shift the timestamp.")
            f0_values = self.shift_series(f0_values, self.pitch_times)
        return f0_values
    
    def get_harmonic_differences(self, 
                                    f0: float, 
                                    f1: float,
                                    f2: float,
                                    f3: float,
                                    samples: np.ndarray = None,
                                    sampling_rate: int = None):
        """
        Calculate the harmonic differences
        :param f0: Fundamental frequency (F0) in Hz
        :param f1: First formant frequency (F1) in Hz
        :param f2: Second formant frequency (F2) in Hz
        :param f3: Third formant frequency (F3) in Hz
        :param samples: Audio samples to analyze (if None, will use the whole audio signal
                                                given by the VoiceFeaturesExtractor)
        :param sampling_rate: Sampling rate of the audio (if None, will use the sampling rate of the sound)
        :return:    - h1_h2, h1_h3 : ratio of energy of the first harmonic to the second and third harmonics in dB
                    - h1_a1, h1_a2, h1_a3 : difference between the first harmonic and picks of the second and third formants in dB

        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        frame = samples * sg.windows.hamming(len(samples))  # Hamming window  
        n_fft = 2 ** int(np.ceil(np.log2(len(frame))))
        fft_spectrum = np.fft.rfft(frame, n=n_fft)  # Compute the FFT of the frame
        freqs = np.fft.rfftfreq(len(fft_spectrum), d=1/sampling_rate)
        magnitude = np.abs(fft_spectrum)  # Get the magnitude of the FFT
        def idx_closest(target): return np.argmin(np.abs(freqs - target)) # Find the index of the closest frequency to the target
        def harmonic_near_formant(formant_freq, f0): # Find the harmonic index closest to the formant frequency
            k = int(round(formant_freq / f0))
            return idx_closest(k * f0)
        # Get the indices of the harmonics and formants
        h1, h2, h3 = idx_closest(f0), idx_closest(2 * f0), idx_closest(3 * f0)
        a1, a2, a3 = harmonic_near_formant(f1, f0), harmonic_near_formant(f2, f0), harmonic_near_formant(f3, f0)
        # Calculate the harmonic differences, replacing infinite values with NaN
        pairs = [(h1, h2), (h1, h3), (h1, a1), (h1, a2), (h1, a3)]
        ratios = []
        for num, den in pairs:
            val = self.get_ratio_db(magnitude[num], magnitude[den], "amplitude")
            ratios.append(np.nan if np.isinf(val) else val)
        h1_h2, h1_h3, h1_a1, h1_a2, h1_a3 = ratios
        
        return h1_h2, h1_h3, h1_a1, h1_a2, h1_a3
    
    def get_jitter(self,
                   min_time: float = 0.0,
                   max_time: float = 0.0,
                   period_floor: float = 0.0001,
                   period_ceiling: float = 0.02,
                   max_period_factor: float = 1.3,
                   method: str = "local") -> float:
        """
        Calculate jitter (i.e variation of the fundamental frequency (F0)) from the sound.
        Please refer to the Praat documentation for more details on the parameters: 
        https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html
        :param min_time: Minimum time in seconds to consider for jitter calculation.
        :param max_time: Maximum time in seconds to consider for jitter calculation.
                        If 0.0, the entire sound will be considered.
        :param period_floor: Shortest period to consider (default is 0.0001 seconds).
        :param period_ceiling: Longest period to consider (default is 0.02 seconds).
        :param max_period_factor: Maximum factor by which the period can vary (default is 1.3).
        :param method: Type of jitter to calculate. Options are "local", "ppq5", or "ddp".
                     - "local": Local jitter calculation.
                     - "ppq5": Period perturbation quotient (5-point).
                     - "ddp": Average absolute difference between consecutive differences between consecutive period
        :return: Jitter value based on the specified methodd.
        """
        if method not in ["local", "ppq5", "ddp"] :
            raise ValueError("Invalid jitter type. Choose from 'local', 'ppq5', or 'ddp'.")
        jitter = call(self.point_process, f"Get jitter ({method})",
                        min_time, max_time, period_floor, period_ceiling, max_period_factor)
        return jitter
