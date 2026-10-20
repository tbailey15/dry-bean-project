"""
Look for natural groups in the beans WITHOUT using the bean type labels.

Run it with:  python cluster_beans.py
Everything it makes is saved in the "results" folder.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

SEED = 42
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ---------- Step 1: Load the data ----------
dataset = fetch_ucirepo(id=602)
X = dataset.data.features.copy()
hidden_labels = dataset.data.targets.squeeze()  # locked away until the end

# K-Means measures distance, so every measurement must be on the same scale.
# Otherwise "Area" (big numbers) would drown out "Roundness" (small numbers).
X_scaled = StandardScaler().fit_transform(X)


# ---------- Step 2: Try different numbers of groups ----------
# Inertia: how tight the groups are (lower is tighter).
# Silhouette: how well each bean fits its own group vs. the next closest
# group (from -1 to 1, higher is better).
scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=5000, random_state=SEED)
    scores.append({"k (number of groups)": k, "inertia": km.inertia_, "silhouette": sil})
    print(f"k={k:2d}  inertia={km.inertia_:9.0f}  silhouette={sil:.3f}")

scores = pd.DataFrame(scores)
scores.to_csv(RESULTS / "cluster_k_scores.csv", index=False)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(scores["k (number of groups)"], scores["inertia"], marker="o")
a1.set_title("Elbow chart (look for the bend)")
a1.set_xlabel("Number of groups (k)")
a1.set_ylabel("Inertia")
a2.plot(scores["k (number of groups)"], scores["silhouette"], marker="o")
a2.set_title("Silhouette score (higher is better)")
a2.set_xlabel("Number of groups (k)")
fig.tight_layout()
fig.savefig(RESULTS / "cluster_choose_k.png", dpi=150)


# ---------- Step 3: Make the groups ----------
# We look at two choices: the k with the best silhouette, and k=7 (because we
# happen to know there are 7 bean types, and we want to see if they line up).
best_k = int(scores.loc[scores["silhouette"].idxmax(), "k (number of groups)"])
print(f"\nBest silhouette was at k={best_k}.")

summary_lines = [f"Best silhouette: k={best_k}"]
for k in sorted({best_k, 7}):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    groups = km.labels_

    # Only NOW do we peek at the real bean types, to see how the groups compare.
    # Group numbers are just names like "Group 0". They do not mean a bean type.
    table = pd.crosstab(pd.Series(groups, name="Group"), hidden_labels)
    table.to_csv(RESULTS / f"clusters_k{k}_vs_bean_type.csv")
    ari = adjusted_rand_score(hidden_labels, groups)
    print(f"\nk={k}: Adjusted Rand Index vs. real types = {ari:.3f}")
    print(table)
    summary_lines.append(f"k={k}: adjusted Rand index = {ari:.3f}")

    # Typical (average) size and shape of each group, in real units.
    profile = X.groupby(groups)[["Area", "Perimeter", "AspectRatio",
                                 "Roundness", "Compactness"]].mean().round(3)
    profile.insert(0, "Beans in group", pd.Series(groups).value_counts().sort_index())
    profile.index.name = "Group"
    profile.to_csv(RESULTS / f"clusters_k{k}_profile.csv")
    print(profile)

summary_lines.append("")
(RESULTS / "cluster_summary.txt").write_text("\n".join(summary_lines))


# ---------- Step 4: Draw the beans on a flat map ----------
# PCA squashes 16 measurements down to 2 so we can draw them. Left side colored
# by the K-Means group, right side colored by the real type (for comparison).
points = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)
km7 = KMeans(n_clusters=7, n_init=10, random_state=SEED).fit(X_scaled)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
a1.scatter(points[:, 0], points[:, 1], c=km7.labels_, cmap="Greys", vmin=-3, s=3)
a1.set_title("Groups K-Means found (k=7, no labels used)")
for name in sorted(hidden_labels.unique()):
    mask = (hidden_labels == name).to_numpy()
    a2.scatter(points[mask, 0], points[mask, 1], s=3, label=name)
a2.set_title("Real bean types (hidden from K-Means)")
a2.legend(markerscale=4, fontsize=8)
for a in (a1, a2):
    a.set_xlabel("PCA direction 1")
a1.set_ylabel("PCA direction 2")
fig.tight_layout()
fig.savefig(RESULTS / "clusters_pca_map.png", dpi=150)

print("\nDone. Look in the results folder.")
