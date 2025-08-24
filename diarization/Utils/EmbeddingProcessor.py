import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize
from tqdm import tqdm
import spacy, os, re
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class EmbeddingProcessor:
    """
    Segmenting texts, embedding them using a pre-trained model, computing embeddings for questions.
    It uses the SentenceTransformer model for text embeddings and Spacy for text segmentation.
    
    Parameters:
    ----------
    model_name : str
        The name of the pre-trained SentenceTransformer model to use for embeddings.
        Can be : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" or 
        "sentence-transformers/all-mpnet-base-v2" or any other model from the SentenceTransformers library.
    spacy_model : str
        The name of the Spacy model to use for text segmentation.
    """
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2',spacy_model='fr_core_news_sm'):
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load(spacy_model)

    def segment_and_embed_texts(self, df):
        segments_list = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing texts'):
            text = row['text']
            doc = self.nlp(text) # Process the text with Spacy (tokenization and segmentation)
            segments = [sent.text.strip() for sent in doc.sents if sent.text.strip() != ''] # Segment the text into sentences
            cleaned_segments = []  
            for i, seg in enumerate(segments):
                if re.fullmatch(r'[^\w\s]+', seg): # Check if the segment contains only punctuation
                    if len(cleaned_segments) > 1:
                        cleaned_segments[-1] += seg # Merge with the previous segment
                    else:
                        continue
                else:
                    cleaned_segments.append(seg)
            segments = cleaned_segments
            segment_embeddings = self.model.encode(segments, convert_to_tensor=True)
            for s_idx, (segment, embedding) in enumerate(zip(segments, segment_embeddings)):
                segments_list.append({
                    'segment': segment,
                    'embedding': embedding,
                    'segment_idx': s_idx,
                    'start_time': row['start'],
                    'end_time': row['end'],
                    'speaker': row['speaker'],
                })
        return pd.DataFrame(segments_list)

    def compute_question_embeddings(self, questions):
        question_embeddings = {}
        assert isinstance(questions, dict), "Questions should be a dictionary."
        assert all(isinstance(q, list) for q in questions.values()), "Each question's value should be a list of reformulations."
        for i, question in enumerate(questions):
            embedding = self.model.encode(question, convert_to_tensor=True)
            embeddings = [embedding]
            for reformulation in questions[question]:
                embedding = self.model.encode(reformulation, convert_to_tensor=True)
                embeddings.append(embedding)
            question_embeddings[question] = embeddings
        return question_embeddings
