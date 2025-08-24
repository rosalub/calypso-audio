import torch
import whisper

class WhisperTranscriber:
    def __init__(self, device="cuda", model="large-v3-turbo", initial_prompt="Question-reponse",verbose=True):
        """
        Initialize the WhisperTranscriber with the specified device and initial prompt.
        Parameters:
        -----------
        device : str
            The device to use for computation. 
        model : str
            The Whisper model to use. 
        initial_prompt : str
            The initial prompt for the transcription.  This can be used to provide a context for transcription, 
            e.g. custom vocabularies or proper nouns to make it more likely to predict those word correctly.
        verbose : bool
            Whether to display the text being decoded to the console. If True, displays all the details,
            If False, displays minimal details. If None, does not display anything
        """
        self.device = device
        self.initial_prompt = initial_prompt
        self.verbose = verbose
        self.model = whisper.load_model(model, device=self.device)
        self.transcription_result = None

    def transcribe_audio(self, audio_path):
        """
        Returns a dictionary containing the resulting text ("text") and segment-level details ("segments"), and
        the spoken language ("language"), which is detected when `decode_options["language"]` is None.
        Example:
        {
            "text": "Bonjour, comment ça va ?",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Bonjour"},
                {"start": 2.5, "end": 5.0, "text": "comment ça va ?"}
            ],
            "language": "fr"
        }
        """
        self.transcription_result = self.model.transcribe(
            audio_path,
            initial_prompt=self.initial_prompt,
            word_timestamps=True,
            verbose=self.verbose,
            language="fr" ,
        )
        return self.transcription_result
    
    def remove_model(self):
        """
        Remove the model from memory and clear the cache.
        """
        del self.model
        torch.cuda.empty_cache()
        if self.verbose:
            print("Model removed from memory")