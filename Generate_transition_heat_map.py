import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("csvs/leary_circle_data_bvcm_verrijkt_transcripts_all.csv")

# Define function to classify quadrant
def classify_quadrant(row):
    if row['dominance'] >= 0.5 and row['empathy'] >= 0.5:
        return 'Dominant-Samen'
    elif row['dominance'] >= 0.5 and row['empathy'] < 0.5:
        return 'Dominant-Tegen'
    elif row['dominance'] < 0.5 and row['empathy'] >= 0.5:
        return 'Submissief-Samen'
    else:
        return 'Submissief-Tegen'

# Apply quadrant classification
df['quadrant'] = df.apply(classify_quadrant, axis=1)

# Sort by transcript and sentence order
df = df.sort_values(by=['transcript_id', 'sentence_id'])

# Create transition pairs
df['next_quadrant'] = df.groupby('transcript_id')['quadrant'].shift(-1)

# Drop rows where next_quadrant is NaN
transitions = df.dropna(subset=['next_quadrant'])

# Create transition matrix
transition_counts = pd.crosstab(transitions['quadrant'], transitions['next_quadrant'])

# Normalize to percentages
transition_percentages = transition_counts.div(transition_counts.sum(axis=1), axis=0) * 100

# Create annotation labels with percentages and counts
annot = transition_percentages.round(1).astype(str) + '%\n(' + transition_counts.astype(str) + ')'

# Reorder rows and columns to match desired quadrant order and flip y-axis
order = ['Dominant-Samen', 'Dominant-Tegen', 'Submissief-Samen', 'Submissief-Tegen']
transition_percentages = transition_percentages.loc[order, order]
annot = annot.loc[order, order]

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(transition_percentages, annot=annot, fmt='', cmap='Blues', cbar_kws={'label': 'Percentage'})
plt.title('Transition Matrix van Roos van Leary-kwadranten')
plt.ylabel('Van kwadrant')
plt.xlabel('Naar kwadrant')
plt.tight_layout()
plt.savefig("transition_matrix_heatmap.png")
plt.show()

