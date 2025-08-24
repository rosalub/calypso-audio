import numpy as np
from features_extractor import VoiceFeaturesExtractor, call

class AmplitudeExtractor(VoiceFeaturesExtractor):
    """
    A class to extract amplitude-related features from audio samples.
    Inherits from VoiceFeaturesExtractor.
    """    
    def __init__(self, audio_path: str, gender: str, time_step: float = 0.01, verbose: bool = True, **kwargs):
        """
        Initializes the AmplitudeExtractor by inheriting from VoiceFeaturesExtractor.
        """
        super().__init__(audio_path, gender, time_step=time_step, verbose=verbose, **kwargs)
        self.hnr_times = None  # To store the times of the harmonicity values for interpolation
    
    def get_hnr(self,
                interpolate: bool = True,
                shift_timestamp: bool = True) -> np.ndarray:
        """
        Calculate the harmonic-to-noise ratio (HNR) from the sound.
        :param interpolate: If True, NaN values will be interpolated using linear interpolation.
        :param shift_timestamp: If True, the formant values will be shifted to match the
        :return: The HNR value.
        """
        harmonicity = self.sound.to_harmonicity_cc(self.time_step, minimum_pitch=self.f0_min)
        self.hnr_times = harmonicity.xs()  # keep the times of the harmonicity values for interpolation
        harmonicity = harmonicity.values[0] 
        #replace values under -200 with NaN (as they are not valid HNR values, noise dominated)
        harmonicity[harmonicity <= -200] = np.nan
        if interpolate:
            # Interpolate NaN values using linear interpolation
            harmonicity = np.interp(self.hnr_times, self.hnr_times[~np.isnan(harmonicity)], 
                            harmonicity[~np.isnan(harmonicity)])
        if shift_timestamp:
            if interpolate is False:
                raise ValueError("Interpolation must be enabled to shift the timestamp.")
            # Shift the formant values to the same global timestamp as the frames
            harmonicity = self.shift_series(harmonicity, self.hnr_times)
        return harmonicity
    
    def get_loudness(self,
                     samples: np.ndarray = None,
                     sampling_rate: int = None) -> float:
        """
        Estimate global loudness in dB (dBFS) of the full signal using RMS.
        :param samples: Audio samples to analyze (if None, will use the whole audio signal
                                                given by the VoiceFeaturesExtractor)
        :return: Scalar loudness value (RMS of the signal in dB).
        """
        if samples is None:
            samples = self.samples
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        rms = np.sqrt(np.mean(samples ** 2))
        loudness = 20 * np.log10(rms + 1e-10) 
        return loudness
    
    def get_shimmer(self, 
                   min_time: float = 0.0, 
                   max_time: float = 0.0, 
                   period_floor: float = 0.0001, 
                   period_ceiling: float = 0.02, 
                   max_period_factor: float = 1.3,
                   max_amplitude_factor: float = 1.6, 
                   method: str = "local") -> float:
        """
        Calculate shimmer (i.e variation of amplitude) features from the sound, based on point process
        Please refer to the Praat documentation for more details on the parameters: 
        https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_shimmer__local____.html
        :param min_time: Minimum time to consider for the calculation (default is 0.0)
        :param max_time: Maximum time to consider for the calculation (default is 0.0)
        :param period_floor: Minimum period to consider (default is 0.0001)
        :param period_ceiling: Maximum period to consider (default is 0.02)
        :param max_period_factor: Maximum factor for the period (default is 1.3)
        :param max_amplitude_factor: Maximum factor for the amplitude (default is 1.6)
        :param method: Type of shimmer to calculate ("local", "apq5" or "dda").
                      - "local": Local shimmer
                      - "apq5": Amplitude perturbation quotient based on 5 points
                      - "dda": Average absolute difference between consecutive differences 
                            between the amplitudes of consecutive periods  
        :return: The shimmer value based on the specified type.
        """
        if method not in ["local", "apq5", "dda"]:
            raise ValueError("Type must be 'local', 'apq5' or 'dda'.")
        shimmer = call([self.sound, self.point_process], f"Get shimmer ({method})",
                    min_time, max_time, period_floor, period_ceiling,
                    max_period_factor, max_amplitude_factor)
        return shimmer