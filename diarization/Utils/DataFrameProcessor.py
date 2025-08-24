import pandas as pd

class DataFrameProcessor:
    @staticmethod
    def create_dataframe_from_transcription(transcription_result):
        df = pd.DataFrame(transcription_result['segments'])
        return DataFrameProcessor.merge_consecutive_speaker_rows(df)

    @staticmethod
    def merge_consecutive_speaker_rows(df):
        merged_rows = []
        previous_row = df.iloc[0]
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            if (previous_row['speaker'] == current_row['speaker'] and 
                current_row['start'] - previous_row['end'] < 1):
                previous_row['text'] += " " + current_row['text']
                previous_row['end'] = current_row['end']
            else:
                merged_rows.append(previous_row)
                previous_row = current_row
                
        merged_rows.append(previous_row)
        return pd.DataFrame(merged_rows)