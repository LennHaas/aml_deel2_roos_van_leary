import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler

generate_gif = False # Set to True to generate GIFs, False to skip
generate_csv_from_data = True # Set to True to generate CSVs, False to skip

# Load Dutch emotion classification model
model_name = "antalvdb/robbert-v2-dutch-base-finetuned-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
emotion_classifier_dutch = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

# Load English emotion classification model]
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
emotion_classifier_english = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

# Emotion to Leary coordinate mapping
emotion_to_leary = {
    "anger": (0.9, 0.1),
    "disgust": (0.8, 0.2),
    "fear": (0.2, 0.2),
    "joy": (0.8, 0.9),
    "neutral": (0.5, 0.5),
    "sadness": (0.2, 0.3),
    "surprise": (0.5, 0.5),
    "trust": (0.6, 0.8)
}

dataset_sources = ["bvcm_automatic_transcripts", "bvcm_verrijkt_transcripts"]
#dataset_sources = ["bvcm_verrijkt_transcripts"]  # Only using enriched transcripts for this example
for dataset_source in dataset_sources:
    if dataset_source == "bvcm_automatic_transcripts":
        number_of_conversations = 26
        conversations = range(-1, number_of_conversations)  # -1 means all conversations
        conversations = range(-1, 0)
    elif dataset_source == "bvcm_verrijkt_transcripts":
        number_of_conversations = 269
        conversations = range(-1, number_of_conversations, 10)  # -1 means all conversations, step 10 for performance
        conversations = range(-1, 0)  # For testing purposes, only process the first conversation
    
    # Load conversation transcripts
    transcripts = pd.read_feather(f"{dataset_source}.feather")

    # Data clean
    transcripts["length"] = [len(transcripts["text"].iloc[i]) for i in range(len(transcripts["text"]))]
    transcripts.drop(transcripts[transcripts["length"] > 500].index, inplace=True)
    transcripts.drop(columns=["length"], inplace=True)
    
    for conversation in conversations:
        plt.close('all')   
        print(f"Processing conversation ID: {"all" if conversation == -1 else conversation} out of {number_of_conversations}, from dataset: {dataset_source}")
        
        # Load conversation transcripts
        transcripts_copy = transcripts.copy(deep=True)

        if conversation != -1: # Keep only the specified conversation
            transcripts.drop(transcripts[transcripts["transcript_id"] != conversation].index, inplace=True)
            transcripts.reset_index(drop=True, inplace=True)
        elif conversation == -1 and dataset_source == "bvcm_verrijkt_transcripts": # When the source is enriched transcripts, remove conversations that are not in the range
            #transcripts.drop(transcripts[~transcripts["transcript_id"].isin(conversations)].index, inplace=True)
            transcripts.reset_index(drop=True, inplace=True)

        # Analyze and map
        data = []
        for iteration, row in transcripts.iterrows():
            
            if row["language"] == "nl":
                emotion_classifier = emotion_classifier_dutch
            elif row["language"] == "en":
                emotion_classifier = emotion_classifier_english
            else:
                print(f"Unsupported language {row['language']} for text: {text}")
                continue
            
            transcript = row["transcript_id"]
            sentence_id = row["sentence_id"]
            person = row["person"]
            text = row["text"]
            
            emotions = emotion_classifier(text)[0]
            top_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]
            
            total_score = sum(e['score'] for e in top_emotions)
            dominance = sum(emotion_to_leary.get(e['label'], (0.5, 0.5))[0] * e['score'] for e in top_emotions) / total_score
            empathy = sum(emotion_to_leary.get(e['label'], (0.5, 0.5))[1] * e['score'] for e in top_emotions) / total_score
            
            data.append((transcript, sentence_id, person, text, [e['label'] for e in top_emotions], dominance, empathy))
            
            if iteration % np.ceil(len(transcripts) / 10.0) == 0:
                print(f"Processed {iteration} out of {len(transcripts)} total sentences.")
        print("Emotion analysis complete.")
        
        df = pd.DataFrame(data, columns=["transcript_id", "sentence_id", "person", "text", "top_emotions", "dominance", "empathy"])

        # Normalize scores
        scaler = MinMaxScaler()
        df[["dominance", "empathy"]] = scaler.fit_transform(df[["dominance", "empathy"]])
        
        if generate_csv_from_data == True and conversation == -1:
            # Save DataFrame to CSV
            folder_path = f"csvs"
            os.makedirs(folder_path, exist_ok=True)
            df.to_csv(f"{folder_path}/leary_circle_data_{dataset_source}_all.csv", index=False)
            print("DataFrame saved to CSV successfully.")
        
        if generate_gif == True:
            # Define colors
            unique_speakers = df["person"].unique()
            colors = {speaker: color for speaker, color in zip(unique_speakers, ['blue', 'red', 'green', 'orange'])}
            
            # Initialize plot
            fig, ax = plt.subplots()
            scat = ax.scatter([], [], s=100)

            # Add quadrant lines and labels
            ax.axhline(0.5, color='black', linestyle='--')
            ax.axvline(0.5, color='black', linestyle='--')
            ax.text(0.75, 0.95, 'Dominant-Together', ha='center', fontsize=12)
            ax.text(0.25, 0.95, 'Dominant-Against', ha='center', fontsize=12)
            ax.text(0.25, 0.05, 'Submissive-Against', ha='center', fontsize=12)
            ax.text(0.75, 0.05, 'Submissive-Together', ha='center', fontsize=12)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Empathy (0=Against, 1=Together)')
            ax.set_ylabel('Dominance (0=Submissive, 1=Dominant)')
            ax.grid(True)
            
            # Animation functions
            def init():
                scat.set_offsets(np.empty((0, 2)))
                return scat,

            def update(frame):
                current_data = df.iloc[max(0, frame-4):frame+1]
                offsets = current_data[["empathy", "dominance"]].values
                color_list = [colors[p] for p in current_data["person"]]
                alpha_list = np.linspace(0.1, 1, len(current_data))
                rgba_colors = [(*plt.cm.colors.to_rgba(c)[:3], a) for c, a in zip(color_list, alpha_list)]
                scat.set_offsets(offsets)
                scat.set_color(rgba_colors)
                
                current_transcript_id = df.iloc[frame]["transcript_id"]
                current_sentence_id = df.iloc[frame]["sentence_id"]
                ax.set_title("Leary Circle - Flow of Dominance and Empathy Over Time" +
                            f"\nConversation: {current_transcript_id} & Sentence: {current_sentence_id}", fontsize=10)
                return scat,
            
            print("Starting animation...")
            ani = animation.FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, repeat=False)
            print("Animation created.")
            
            # Save and show
            folder_path = f"gifs/{dataset_source}"
            os.makedirs(folder_path, exist_ok=True)
            
            print("Saving animation...")
            if conversation != -1:
                ani.save(f"{folder_path}/leary_circle_animation_{dataset_source}_{conversation}.gif", writer="pillow", fps=max((len(df)//25), 1))
            else:
                ani.save(f"{folder_path}/leary_circle_animation_{dataset_source}_all.gif", writer="pillow", fps=max((len(df)//25), 1))
            print("Animation saved successfully.")
            #plt.show()