import torch, torchaudio
import pandas as pd
import numpy as np
from AudioToTextPipline import AudioToTextPipeline
from Utils.AudioProcessor import AudioProcessor

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ExtractPatientRecordPipeline(AudioToTextPipeline):
    """
    Pipeline to extract patient records from audio files.
    It performs speaker diarization, transcription, and extracts patient records based on predefined questions.
    """
    def __init__(self,
                 questions,
                 keywords,
                 device="cuda",
                 verbose=False,
                 normalize=True,
                 threshold=0.5):

        super().__init__(questions=questions,
                        keywords=keywords,
                        device=device,
                        verbose=verbose,
                        normalize=normalize,
                        threshold=threshold)
        
        self.audio_processor = AudioProcessor(device=device, verbose=verbose)

    def process_identification(self,
                              audio_path: str,
                              threshold_score: float = 0.75,
                              keep_cleaned:bool = False) -> tuple:
        """
        Process the audio file to identify the speakers and extract the time codes of the segments.

        :param audio_path: The path to the audio file to process.
        :param threshold_score: The cosine similarity score threshold to consider a segment as a question match.

        :returns: A tuple containing:
        - duration: The duration of the audio file in seconds.
        - time_code: A DataFrame containing the time code of the segments with the speaker information.
        - y: The audio signal of the audio file.
        - sr: The sample rate of the audio file.
        """
        # Perform speaker diarization and transcription
        matched_questions, transcription_result, segments_df, new_audio_path = self.process_transcription(audio_path)
        y, sr = torchaudio.load(new_audio_path) 
        if not keep_cleaned:
            self.audio_processor.delete_audio_file()
        duration = y.shape[1] / sr
        # Identify the interviewer speaker: get the speaker who ask the most questions 
        # (with a cosine similarity score high enough)
        clinician = matched_questions[matched_questions['similarity'] > threshold_score]['speaker'].value_counts().idxmax()
        # drop nan values in the segments_df
        segments_df = segments_df.dropna(subset=['speaker'])
        for speaker in segments_df['speaker'].unique(): 
            if speaker != clinician:
                patient = speaker
        time_code = segments_df[['speaker', 'start_time', 'end_time', 'segment']]
        time_code = time_code.drop_duplicates() # remove duplicates due to the splitting of the segments in sentences (EmbeddingProcessor.segment_and_embed_texts)
        time_code = time_code.reset_index(drop=True) 
        time_code['speaker'] = time_code['speaker'].replace({clinician: 'Clinician', patient: 'Patient'})
        return duration, time_code, y, sr
    
    def process_concatenation(self, 
                time_code: pd.DataFrame = None,
                y: torch.Tensor = None,
                sr: int = None,
                required_latency: float = 1,
                margin: float = 0.5) -> tuple: 
        """
        Process the audio file to concatenate the segments of the patient speech with a margin around them.
        :param time_code: A DataFrame containing the time code of the segments with the speaker information.
        :param y: The audio signal of the audio file.
        :param sr: The sample rate of the audio file.
        :param required_latency: The required latency in seconds to consider a segment as a patient speech.
        :param margin: The margin in seconds to add before and after the patient speech segments.
        :returns: A tuple containing:
        - patient_audio: The concatenated audio signal of the patient speech segments with the margin around them.
        """               
        if required_latency < margin:
            raise ValueError("The required latency should be greater than the margin.")
        
        patient_audio = []
        patient_segments = time_code[time_code['speaker'] == 'Patient'].copy()
        patient_segments = patient_segments.drop_duplicates(subset=['start_time', 'end_time'])
        patient_segments = patient_segments.reset_index(drop=True)
        # Get the diarized segments
        for i, segment in patient_segments.iterrows():
            latency_margin_before = 0
            latency_margin_after = 0
            # Calculate the start and end time of the segment with the margin
            current_start = segment['start_time']
            current_end = segment['end_time']

            # Check for previous and next segments to adjust the start and end time
            prev_segments = time_code[time_code['end_time'] <= current_start]
            if not prev_segments.empty:
                last_end = prev_segments['end_time'].max()
                latency_margin_before = current_start - last_end
            next_segments = time_code[time_code['start_time'] >= current_end]
            if not next_segments.empty:
                next_start = next_segments['start_time'].min()
                latency_margin_after = next_start - current_end
            
            # Adjust the start and end time with the latency margin
            latency_margin_before = margin if latency_margin_before > required_latency else 0 
            latency_margin_after = margin if latency_margin_after > required_latency else 0
            
            start = current_start - latency_margin_before
            end = current_end + latency_margin_after
            
            patient_audio.append(y[:, int(start * sr):int(end * sr)])
        
        patient_audio = torch.cat(patient_audio, dim=1) 
        return patient_audio