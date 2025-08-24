import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import necessary modules
from Utils.AudioProcessor import AudioProcessor
from Utils.WhisperTranscriber import WhisperTranscriber
from Utils.WhisperAligner import WhisperAligner
from Utils.EmbeddingProcessor import EmbeddingProcessor
from Utils.QuestionMatcher import QuestionMatcher
from Utils.DataFrameProcessor import DataFrameProcessor

import whisperx
    
class AudioToTextPipeline:
    def __init__(self, questions,keywords ,device="cuda",verbose=False, normalize=True, 
                keep_cleaned=False, threshold=0.5):
        self.verbose = verbose
        self.audio_processor = AudioProcessor(device=device, verbose=verbose)
        self.transcriber = WhisperTranscriber(device,verbose=verbose)
        self.aligner = WhisperAligner(device,verbose=verbose)
        self.embedding_processor = EmbeddingProcessor()
        self.questions = questions
        self.keywords = keywords
        self.normalize = normalize
        self.threshold = threshold
        self.keep_cleaned = keep_cleaned  
        self.matched_questions_df = None

    def process_transcription(self, audio_path):
        # Audio normalization
        if self.normalize:
            audio_path = self.audio_processor.filter_and_normalize(audio_path, 
                                                                   keep_cleaned=self.keep_cleaned)
      # Transcription
        transcription_result = self.transcriber.transcribe_audio(audio_path)
        self.transcriber.remove_model()
        # Alignment
        self.aligner.load_align_model(language_code=transcription_result["language"])
        aligned_result = self.aligner.align_transcription(transcription_result['segments'], audio_path)
        # Speaker Diarization
        diarized_segments = self.aligner.perform_speaker_diarization(audio_path)
        transcription_result = whisperx.assign_word_speakers(diarized_segments, aligned_result)
        # Dataframe creation and merging
        segments_df = DataFrameProcessor.create_dataframe_from_transcription(transcription_result)
        #save the segments_df to a csv file
        # Embedding texts
        segments_df = self.embedding_processor.segment_and_embed_texts(segments_df)
        question_embeddings = self.embedding_processor.compute_question_embeddings(self.questions)
        
        #save question_embeddings to a txt file
        with open("question_embeddings.txt", "w") as f:
            for question, embedding in question_embeddings.items():
                f.write(f"{question}: {embedding}\n")
        # Question Matching
        self.matched_questions_df = QuestionMatcher.find_top_matches_with_threshold(
            segments_df, 
            self.questions, 
            question_embeddings, 
            self.keywords,
            threshold=self.threshold , # all the segments with a cosine similarity score  close to the maximum score within a given threshold
        )
        #clean everything and remove the model
        return self.matched_questions_df, transcription_result, segments_df, audio_path