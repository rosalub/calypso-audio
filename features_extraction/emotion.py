import os
import librosa
import torch
import numpy as np
import pandas as pd
import random
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from AudioProcessor import AudioProcessor 
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SlidingEmotionInference:
    """
    Class to perform sliding window inference on audio files for emotion recognition.
    """
    def __init__(self, audio_path, processor, sampling_rate=16000, window_size=3, window_step=1.5, keep_cleaned=False):
        self.audio_path = audio_path
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.audio_processor = AudioProcessor(verbose=False)
        self.window_size = int(window_size * sampling_rate)
        self.window_step = int(window_step * sampling_rate)
        cleaned_path = self.audio_processor.filter_and_normalize(audio_path, keep_cleaned)
        waveform, _ = librosa.load(cleaned_path, sr=sampling_rate)
        if not keep_cleaned:
            self.audio_processor.delete_audio_file()
        self.windows = self.get_sliding_windows(waveform)

    def get_sliding_windows(self, waveform):
        num_samples = len(waveform)
        windows = []
        start = 0
        while start < num_samples:
            end = start + self.window_size
            if end <= num_samples:
                windows.append(waveform[start:end])
            else:
                last_window = waveform[start:]
                if len(last_window) < self.window_size:
                    pad_width = self.window_size - len(last_window)
                    last_window = np.pad(last_window, (0, pad_width))
                windows.append(last_window)
            start += self.window_step
        return np.array(windows)

    def infer(self, model, device):
        preds = []
        model.eval()
        with torch.no_grad():
            for window in self.windows:
                inputs = self.processor(window, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
                out = model(inputs)
                preds.append(out.cpu().numpy()[0])
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)

# --- Model ---
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            pooled = hidden_states.mean(dim=1)
            logits = self.classifier(pooled)
            return logits

def main():
    parser = argparse.ArgumentParser(description="Sliding window emotion inference on a directory of wav files.")
    parser.add_argument('--audio_dir', type=str, default='/media/rosalie/donnees/calypso_audio/data/processed/audios', 
                        help='Directory containing wav files')
    parser.add_argument('--output_path', type=str, default='/media/rosalie/donnees/calypso_audio/data/features/patients_emotion.csv', 
                        help='Path to save the CSV results')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate of audio files (default: 16000)')
    parser.add_argument('--model_name', type=str, default='/home/rosalie/Documents/DIVA/wav2vec_finetuned/model', 
                        help='Path to the fine-tuned model')
    parser.add_argument('--processor_name', type=str, default='/home/rosalie/Documents/DIVA/wav2vec_finetuned/processor',
                         help='Path to the processor')
    parser.add_argument('--window_size', type=float, default=3.0, help='Sliding window size in seconds (default: 3.0)')
    parser.add_argument('--window_step', type=float, default=1.5, help='Sliding window step in seconds (default: 1.5)')
    parser.add_argument('--keep_cleaned', action='store_true', help='Keep cleaned audio files')
    args = parser.parse_args()
    # Load model and processor, set device
    processor = Wav2Vec2Processor.from_pretrained(args.processor_name)
    model = EmotionModel.from_pretrained(args.model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for filename in tqdm(os.listdir(args.audio_dir), desc="Processing files"):
        if not filename.endswith(".wav"):
            continue
        audio_path = os.path.join(args.audio_dir, filename)
        inf = SlidingEmotionInference(audio_path, processor, window_size=args.window_size, window_step=args.window_step, keep_cleaned=args.keep_cleaned)
        vad_mean, vad_std = inf.infer(model, device)
        results.append({
            "filename": filename,
            "arousal_mean": vad_mean[0],
            "dominance_mean": vad_mean[1],
            "valence_mean": vad_mean[2],
            "arousal_std": vad_std[0],
            "dominance_std": vad_std[1],
            "valence_std": vad_std[2]
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()