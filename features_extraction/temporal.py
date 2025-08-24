import re
import pandas as pd
import numpy as np

def remove_outliers(feature):
    """
    Remove outliers from a feature using the IQR method.
    
    :param feature: The feature to process (as a pandas Series).
    :return: The feature with outliers removed.
    """
    Q1 = feature.quantile(0.25)
    Q3 = feature.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return feature[(feature >= lower_bound) & (feature <= upper_bound)]

@staticmethod
def count_syllables_word(word):
    """
    Count the number of syllable groups in a word (French vowels).
    :param word: The word to count syllables in.
    :return: The number of syllable groups in the word.
    """
    word = word.lower()
    vowels = "aeiouyàâäéèêëîïôöùûüÿ"
    # Count groups of consecutive vowels
    return len(re.findall(rf"[{vowels}]+", word))

def count_syllables_sentence(sentence):
    """
    Count the total number of syllables in a sentence.
    :param sentence: The sentence to count syllables in.
    :return: The total number of syllables in the sentence.
    """
    words = re.findall(r"\b\w+\b", sentence)
    return sum(count_syllables_word(word) for word in words)

def get_speech_rate(segments_df):
    """
    Calculate the speech rate for each segment (in syllables per minute).
    :param segments_df: DataFrame containing transcribed segments ('segment') 
    with 'speaker', 'start_time' and 'end_time'.
    :return:
    - segment_df: DataFrame with an additional 'speech_rate' column.
                This column contains the speech rate in syllables 
                per minute for each segment.
    - mean_speech_rate: The mean speech rate across all patient's segments.

    """
    segments_df['speech_rate'] = segments_df.apply(
        lambda row: count_syllables_sentence(row['segment']) * 60 / (row['end_time'] - row['start_time']),
        axis=1
    )
    # Filter for patient's segments
    patient_segments = segments_df[segments_df['speaker'] == 'Patient']
    # Calculate the mean speech rate for the patient
    speech_rates = remove_outliers(patient_segments['speech_rate'])
    mean_speech_rate = np.mean(speech_rates)
    std_speach_rate = np.std(speech_rates)
    return segments_df, mean_speech_rate, std_speach_rate

def get_latency(segments_df):
    """
    Calculate the latency of the patient (time to answer to the clinician's question) 
    in seconds, and the number of interruptions (when the patient starts speaking before 
    the clinician finishes asking a question).
    :param segments_df: DataFrame containing transcribed segments 
                        with 'speaker', 'start_time', and 'end_time'.
    :return: Tuple containing:
    - mean_latency: The mean latency of the patient in seconds.
    - std_latency: The standard deviation of the latency.
    - min_latency: The minimum latency of the patient in seconds.
    - max_latency: The maximum latency of the patient in seconds.
    - interruptions: The number of interruptions from the patient.
    """
    old_speaker = None
    end_speak = None
    interruptions = 0
    latency = [] # store the latency of the patient
    for segment in segments_df.itertuples(index=False):
        speaker = segment.speaker
        if speaker == 'Patient':
            start_speak = segment.start_time
            if old_speaker == 'Clinician' and end_speak is not None:
                # compute the time to answer the question (time between the end of the question and the beginning of the answer)
                time_to_answer = start_speak - end_speak
                if time_to_answer < 0:
                    time_to_answer = 0
                    interruptions += 1
                latency.append(time_to_answer)            
        end_speak = segment.end_time
        old_speaker = speaker
    latency = remove_outliers(pd.Series(latency))
    mean_latency = np.mean(latency)
    std_latency = np.std(latency)
    min_latency = np.min(latency)   
    max_latency = np.max(latency)
    return mean_latency, std_latency, min_latency, max_latency, interruptions