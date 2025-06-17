import pandas as pd
import re
import matplotlib.pyplot as plt
from langdetect import detect

transcripts = pd.read_excel('BVCM_Automatic_Labeled_transcriptions.xlsx')

transcripts.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "Transcription", "DateTime", "Callduration"], inplace=True)
transcripts.rename(columns={"Labeled_transcription": "text"}, inplace=True)

transcripts_new = pd.DataFrame(columns=["transcript_id", "sentence_id", "person", "text", "language"])

for index, text in enumerate(transcripts["text"]):
    responses = pd.DataFrame(re.findall(r'\|([^|]+)\|\s*([^|]+)', text))
    responses.columns = ["person", "text"]
    responses["text"] = responses["text"].str.strip()
    responses["transcript_id"] = index
    responses["sentence_id"] = range(len(responses))
    transcripts_new = pd.concat([transcripts_new, responses], ignore_index=True)
transcripts_new.reset_index(drop=True, inplace=True)

transcripts = transcripts_new

# Language detection
full_conversations = transcripts.groupby('transcript_id')['text'].apply(lambda x : ' '.join(x))
IdToLanguage_dict = dict(full_conversations.apply(detect))
transcripts['language'] = transcripts['transcript_id'].map(IdToLanguage_dict)

transcripts.to_feather('BVCM_transcripts.feather')