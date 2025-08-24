import parselmouth
from parselmouth.praat import call
import numpy as np
import scipy.signal as sg

class VoiceFeaturesExtractor:
    """
    A class to extract various voice features from audio files using the Parselmouth library.
    Features belong to few categories:
    - Energy (hnr, shimmer, loudness)
    - Frequency (f0, formants, jitter, pitch)
    - Spectral (alpha ratio, mfccs, hammarberg index, spectral slopes)
    """
    def __init__(self, 
                 audio_path: str, 
                 gender: str,
                 verbose: bool = False,
                 resample: bool = True,
                 time_step: float = 0.01,
                 window_size: float = 0.03, 
                 padding_mode: str = "reflect"):
        """
        Initializes the VoiceFeatureExtractor with the audio file
        :param audio_path: Path to the audio file
        :param gender: Gender of the speaker (M, F, NB, C) (respectively Male, Female, Non-Binary, Child)
                        or a tuple (f0_min, f0_max), default is (75, 300)
        :param verbose: If True, prints additional information during processing
        :param resample: If True, resamples the audio to 16000 Hz if the original sampling rate is lower
        :param time_step: Time step for pitch extraction in seconds (default is 0.01)
        :param window_size: To set the size of the window for each frame (default is 0.03 seconds)
        :param padding_mode: Padding mode for out-of-bounds indices (default is "reflect")
        """
        self.verbose = verbose
        self.sound = parselmouth.Sound(audio_path)
        self.frequencies = {"M": (75, 150), "F": (160, 280), "NB": (100, 240), "C": (250, 350)}
        if isinstance(gender, tuple): 
            self.f0_min, self.f0_max = gender
        else:
            self.f0_min, self.f0_max = self.frequencies.get(gender, (75, 300))
        self.point_process = call(self.sound, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)
        self.sampling_rate = self.sound.sampling_frequency
        if self.sampling_rate != 16000 and resample:
            if self.verbose:
                print(f"Resampling audio from {self.sampling_rate} Hz to 16000 Hz.")
            self.sound = self.sound.resample(16000)
            self.sampling_rate = 16000

        self.samples = self.sound.values[0]
        self.time_step = time_step
        self.window_size = window_size  
        self.times = np.arange(0, self.sound.get_total_duration(), self.time_step)
        self.frames = self.get_aligned_frames(times=self.times,
                                        window_size=window_size,
                                        padding_mode=padding_mode)
        
    def get_aligned_frames(self, 
                        times, 
                        window_size: float = 0.03,
                        padding_mode: str = "reflect"):
        """
        Cut the audio samples into frames aligned with the specified timestamps.
        Each frame is centered around the timestamp, with a specified window size.
        
        :param times: timestamps (in seconds) where the frames should be centered
        :param window_size: size of the window for each frame (in seconds)
        :param padding_mode: Padding mode (e.g. 'reflect', 'constant', 'edge')
        :return: frames: list of numpy arrays, each representing a frame of audio samples
        """
        frames = []
        half_win = int((window_size / 2) * self.sampling_rate)

        for t in times:
            center = int(t * self.sampling_rate)
            start = center - half_win
            end = center + half_win

            # Handle out-of-bounds indices
            if start < 0 or end > len(self.samples):
                frame = np.pad(
                    self.samples[max(start, 0):min(end, len(self.samples))],
                    (max(0, -start), max(0, end - len(self.samples))),
                    mode=padding_mode)
            else:
                frame = self.samples[start:end]
            frames.append(frame)
        frames = np.array(frames)
        return np.array(frames)
        
    def shift_series(self, serie, timestamp):
        """
        Shift the serie to the same global timestamp as the frames (by interpolation)
        :param serie: The series to shift (e.g. f0, hnr, formants)  
        :param timestamp: The timestamp to shift
        """
        return np.interp(self.times, timestamp, serie)
    
    def get_ratio_db(self, 
                     value1: float, 
                     value2: float, 
                     mode: str = "power"):
        """
        Computes the ratio in dB between two values.
        :param value1: The first value (numerator)
        :param value2: The second value (denominator)
        :param mode: The mode of the ratio :"power" (e.g. energy) or "amplitude"
        :return: The ratio in dB
        """
        if mode == "power":
            return 10 * np.log10(value1 / (value2 + 1e-10))
        elif mode == "amplitude":
            return 20 * np.log10(value1 / (value2 + 1e-10))
        else:
            raise ValueError("Mode must be 'power' or 'amplitude'.")
