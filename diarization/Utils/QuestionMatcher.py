import torch
import pandas as pd
import numpy as np
from sentence_transformers import util
from typing import List, Dict
from torch.nn.functional import normalize

class QuestionMatcher:
    @staticmethod
    def calculate_mean_cosine_similarity(
        question_embeddings: torch.Tensor, 
        segment_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean cosine similarity between multiple question reformulation embeddings and segment embeddings.
        """
        cosine_scores = util.cos_sim(question_embeddings, segment_embeddings)  
        mean_cosine_score = cosine_scores.mean(dim=0) 
        return mean_cosine_score

    @staticmethod
    def extract_embeddings(segments_df: pd.DataFrame, question_key: str, question_embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor | torch.Tensor:
        """
        Extract segment and question embeddings for calculation.
        """
        segment_embeddings = torch.stack(segments_df['embedding'].tolist()).cpu()  # Shape: (num_segments, embedding_dim)
        question_embeddings = question_embeddings_dict.get(question_key)
        if question_embeddings is None:
            raise ValueError(f"Question '{question_key}' not found in the question embeddings dictionary.")
        if isinstance(question_embeddings, list):
            question_embeddings = torch.stack(question_embeddings)
        return segment_embeddings, question_embeddings.cpu()

    @staticmethod
    def find_top_matches_with_threshold(
        segments_df: pd.DataFrame, 
        questions: List[List[str]], 
        question_embeddings_dict: Dict[str, torch.Tensor], 
        question_keywords: Dict[str, List[str]],  # External keywords
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Find all matching segments for each question with scores close to the maximum score within a given threshold.
        Returns a DataFrame with matches for each question.
        """
        top_matches = []
        total_duration = segments_df['end_time'].max()  # Calculate total interview duration

        for i, question in enumerate(questions):
            keywords = question_keywords.get(question)  # Fetch keywords for the current question   
            try:
                # Extract embeddings for the question and segments
                segment_embeddings, question_embeddings = QuestionMatcher.extract_embeddings(
                    segments_df, question, question_embeddings_dict
                )
            except ValueError as e:
                print(e)
                continue
            # Calculate mean cosine similarities between the question and all segments
            mean_cosine_scores = QuestionMatcher.calculate_mean_cosine_similarity(
                question_embeddings, segment_embeddings
            ).detach().numpy()
            # Additional weighting for keyword relevance
            keyword_weights = np.array([
                1.2 if any(word in segments_df.iloc[j]['segment'] for word in keywords) else 1.0
                for j in range(len(mean_cosine_scores))
            ])
            final_scores = mean_cosine_scores * keyword_weights

            # Find all segments with scores within the threshold of the maximum score
            max_score = final_scores.max()
            selected_indices = np.where(final_scores >= max_score - threshold)[0]
            selected_segments = segments_df.iloc[selected_indices].copy()
            selected_segments['similarity'] = final_scores[selected_indices]
            selected_segments['question'] = question  # Add question text for each selected match

            # Append the selected matches to the list
            top_matches.append(selected_segments[['question', 'segment', 'start_time', 'end_time', 'similarity', 'speaker']])

        # Concatenate all selected matches into a single DataFrame and return it
        return pd.concat(top_matches, ignore_index=True)
