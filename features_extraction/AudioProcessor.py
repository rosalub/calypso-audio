import os
import tempfile
import torch, torchaudio
from pydub import AudioSegment, effects
from scipy.signal import butter, filtfilt
import numpy as np

class AudioProcessor:
    def __init__(self, 
                 lowcut=20, 
                 highcut=8000, 
                 order=2,
                 device="cuda", 
                 verbose=False):
        """
        Initialize the AudioProcessor with device and verbosity settings.
        """
        self.device = device  
        self.verbose = verbose  
        self.temp_dir = tempfile.gettempdir()  # Temporary directory for intermediate files
        self.normalized_audio_path = None  # Path to the last normalized audio
        self.lowcut = lowcut  # Low cut frequency for bandpass filter
        self.highcut = highcut  # High cut frequency for bandpass filter
        self.order = order  # Order of the bandpass filter

    def bandpass_filter(self, y, sr):
        def butter_bandpass(lowcut, highcut, sr, order):
            nyq = 0.5 * sr  # Nyquist frequency
            low = lowcut / nyq  # Normalized low frequency
            if highcut >= nyq:
                highcut = nyq - 1e-3 # 0.001 Hz below Nyquist
            high = highcut / nyq  # Normalized high frequency
            b, a = butter(order, [low, high], btype='band')  # Bandpass filter coefficients
            return b, a
        b, a = butter_bandpass(self.lowcut, self.highcut, sr, self.order) 
        if y.ndim == 1:
            return filtfilt(b, a, y)
        elif y.ndim == 2:
            y_filtered = np.zeros_like(y)  
            for i in range(y.shape[0]):
                y_filtered[i, :] = filtfilt(b, a, y[i, :])  # Apply filter to each channel
            return y_filtered 
        else:
            raise ValueError("Input signal must be 1D (Mono) or 2D (Stéréo) array.")  

    def delete_audio_file(self):
        """
        Deletes the temporary audio file if it exists.
        """
        if os.path.exists(self.normalized_audio_path):  # Check if file exists
            os.remove(self.normalized_audio_path)  # Remove file

    def normalize_audio(self, audio_path, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"{os.path.basename(audio_path)}_cleaned.wav")  # default output path in temp dir
        self.normalized_audio_path = output_path  # store path for later deletion
        audio = AudioSegment.from_file(audio_path)  # Load audio file
        normalized_audio = effects.normalize(audio)  # Normalize audio
        normalized_audio.export(output_path, format="wav")  # Export normalized audio
        if self.verbose:
            print(f"Normalized audio saved to {output_path}")  # Print info if verbose
        return output_path  # Return path to normalized audio

    def filter_and_normalize(self, input_path, keep_cleaned=False):
        """
        Apply bandpass filter then normalization to an audio file.
        If keep_cleaned=True, save the cleaned file in a 'cleaned_audios' folder
        next to the original folder. Returns the path to the normalized file.
        """
        y, sr = torchaudio.load(input_path) # Load audio (mono or stereo)
        y_np = y.numpy()  # Convert to numpy array
        if y_np.shape[0] == 1:
            y_np = y_np[0]  # Remove channel dimension if mono
        y_filtered = self.bandpass_filter(y_np, sr)  #Bandpass filtering
        temp_filtered_path = os.path.join(self.temp_dir, f"{os.path.basename(input_path)}_filtered.wav")  # 2. Temp file for filtered audio
        if y_filtered.ndim == 1:
            y_filtered = y_filtered.copy()  
            y_filtered_torch = torch.from_numpy(y_filtered).unsqueeze(0)  # Convert mono to torch tensor with channel
        else:
            y_filtered_torch = torch.from_numpy(y_filtered)  # Stereo to torch tensor
        torchaudio.save(temp_filtered_path, y_filtered_torch, sr)  # Save filtered audio to temp file
        if keep_cleaned:
            input_dir = os.path.dirname(os.path.abspath(input_path))  # Get input file directory
            cleaned_dir = os.path.join(input_dir, "cleaned_audios")  # Create 'cleaned_audios' folder
            os.makedirs(cleaned_dir, exist_ok=True)  # Ensure folder exists
            cleaned_filename = os.path.splitext(os.path.basename(input_path))[0] + "_cleaned.wav"  # Cleaned file name
            cleaned_path = os.path.join(cleaned_dir, cleaned_filename)  # Full path for cleaned file
        else:
            cleaned_path = None  # Use temp dir if not keeping cleaned file
        normalized_path = self.normalize_audio(temp_filtered_path, output_path=cleaned_path)  # 4. Normalize filtered audio
        if os.path.exists(temp_filtered_path):
            os.remove(temp_filtered_path)  # 5. Remove temp filtered file
        return normalized_path  # Return path to normalized (cleaned) audio