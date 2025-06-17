import pandas as pd
import re
from langdetect import detect_langs

def detect_en_nl_only(text):
    try:
        langs = detect_langs(text)
        # Filter only 'en' and 'nl'
        filtered = [lang for lang in langs if lang.lang in ['en', 'nl']]
        if filtered:
            return max(filtered, key=lambda x: x.prob).lang
        else:
            return 'en'  # fallback default
    except:
        return 'en'  # fallback in case of error

# Function to parse the transcript file
def parse_transcript(file_path):
    data = []
    transcript_id = -1
    sentence_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Check for transcription ID line
            if line.startswith("==="):
                transcript_id += 1
                sentence_id = 0
            else:
                # Extract person and text
                match = re.match(r'\[(speaker_\d+)\] \[\d+\.\d+ - \d+\.\d+\] (.+)', line)
                if match:
                    person = match.group(1)
                    text = match.group(2)
                    data.append((transcript_id, sentence_id, person, text))
                    sentence_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["transcript_id", "sentence_id", "person", "text"])
        
    full_conversations = df.groupby('transcript_id')['text'].apply(lambda x: ' '.join(x))
    IdToLanguage_dict = dict(full_conversations.apply(detect_en_nl_only))
    df['language'] = df['transcript_id'].map(IdToLanguage_dict)

    
    return df

# Parse the transcript file
df = parse_transcript('bvcm_transcripties_verrijkt.txt')
#df.to_feather('bvcm_verrijkt_transcripts.feather')

print(df[["transcript_id", "sentence_id"]].head(50))