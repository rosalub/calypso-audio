import whisperx

class WhisperAligner:
    def __init__(self, device="cuda", verbose=False):
        """
        Initialize the WhisperAligner with the specified device and verbosity.
        Parameters:
        -----------
        device : str
            The device to use for computation.
        verbose : bool
            Whether to display the text being decoded to the console. If True, displays all the details,
            If False does not display anything
        """
        self.device = device
        self.verbose = verbose
        self.model_align = None
        self.metadata = None

    def load_align_model(self, language_code):
        """
        Load the alignment model for the specified language code.
        Parameters:
        -----------
        language_code : str
            The language code for the model. This should be a valid ISO 639-1 code, e.g., "fr" for French.
        """
        self.model_align, self.metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device)

    def align_transcription(self, segments, audio_path):
        """
        Align phoneme recognition predictions to known transcription.
        Parameters:
        -----------
        segments : list
            A list of segments containing the transcription and their timestamps.
        audio_path : str
            The path to the audio file to be aligned.
        """
        # Preprocess segments to remove leading and trailing spaces
        for segment in segments:
            segment['text'] = segment['text'].strip()
        
        aligned_result = whisperx.align(
            segments, self.model_align, self.metadata, 
            audio_path, device=self.device, combined_progress=True
        )
        return aligned_result
    
    def perform_speaker_diarization(self, audio_path, num_speakers = 2):
        """
        Perform speaker diarization on the audio file.
        Parameters:
        -----------
        audio_path : str
            The path to the audio file to be diarized.
        num_speakers : int
            The number of speakers to identify in the audio. 
        """
        if self.verbose:
            print(f"Performing speaker diarization on {audio_path}")
        diarize_model = whisperx.DiarizationPipeline(device=self.device)
        diarized_segments = diarize_model(audio_path, num_speakers=num_speakers)
        if self.verbose:
            print("Speaker diarization completed")
        return diarized_segments