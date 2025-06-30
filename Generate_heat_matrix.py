import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csvs\leary_circle_data_bvcm_verrijkt_transcripts_all.csv")

# Bepaal het kwadrant op basis van dominantie en empathie
def bepaal_kwadrant(d, e):
    if d >= 0.5 and e >= 0.5:
        return "Dominant-Samen"
    elif d >= 0.5 and e < 0.5:
        return "Dominant-Tegen"
    elif d < 0.5 and e >= 0.5:
        return "Submissief-Samen"
    else:
        return "Submissief-Tegen"

df["kwadrant"] = df.apply(lambda row: bepaal_kwadrant(row["dominance"], row["empathy"]), axis=1)

# Tel het aantal zinnen per kwadrant
kwadrant_counts = df["kwadrant"].value_counts().reindex([
    "Dominant-Tegen", "Dominant-Samen", "Submissief-Tegen", "Submissief-Samen"
])

# Zet de tellingen in een 2x2 matrix voor heatmap
matrix = pd.DataFrame([
    [kwadrant_counts["Dominant-Tegen"], kwadrant_counts["Dominant-Samen"]],
    [kwadrant_counts["Submissief-Tegen"], kwadrant_counts["Submissief-Samen"]]
], index=["Dominant", "Submissief"], columns=["Tegen", "Samen"])

# Plot de heatmap met wit-blauw kleurenschema
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5, linecolor='gray')
plt.title("Spreiding van zinnen over Roos van Leary-kwadranten")
plt.ylabel("Dominantie-as")
plt.xlabel("Samenwerkings-as")
plt.tight_layout()
plt.show()