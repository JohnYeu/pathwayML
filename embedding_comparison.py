#!/usr/bin/env python3
"""
PathwayML-Ath: Non-linear Embedding Comparison Pipeline
========================================================
Plug-in module for the main PathwayML-Ath project.
Compares 4 embedding methods: SVD (linear), UMAP, Node2Vec on GO DAG, Autoencoder.
All results are evaluated with hard negatives (4 types) to avoid inflated AUROC.

Usage:
    cd PathwayML-Ath/
    python embedding_comparison.py

Output:
    - tables/embedding_comparison.csv
    - figures/fig_embedding_comparison.png
    - Console: full ablation table
    
Requirements:
    pip install numpy pandas scikit-learn xgboost umap-learn node2vec gensim matplotlib
    Optional: pip install torch (for proper autoencoder; falls back to sklearn MLP if absent)
"""

import os
import sys
import numpy as np
import pandas as pd
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import run_no_embedding_reproducible as core

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = 'data'
FIG_DIR = 'figures'
TABLE_DIR = 'tables'
SEED = 42
N_EMB = 20          # embedding dimensions for SVD and UMAP
N_EMB_N2V = 20      # node2vec dimensions
N_EMB_AE = 20       # autoencoder latent dimensions
CV_SPLITS = 5
CV_REPEATS = 5      # Repeated CV for stable estimates

np.random.seed(SEED)
for d in [FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 70)
print("Step 1: Loading data")
print("=" * 70)

# gene_go: gene -> set of GO terms; go_genes: GO term -> set of genes
gene_go = defaultdict(set)
go_genes = defaultdict(set)
# Parse TAIR GAF (Gene Association Format), keeping only valid Arabidopsis loci
with open(os.path.join(DATA_DIR, 'tair.gaf')) as f:
    for line in f:
        if line.startswith('!'):  # skip header/comment lines
            continue
        p = line.strip().split('\t')
        if len(p) < 15:
            continue
        g = p[1].upper()
        go = p[4]
        # Accept only canonical AGI locus identifiers (e.g. AT1G01010)
        if re.match(r'AT[0-9]G[0-9]{5}', g):
            gene_go[g].add(go)
            go_genes[go].add(g)

# Load KEGG pathway-gene memberships, stripping namespace prefixes
kegg = defaultdict(set)
with open(os.path.join(DATA_DIR, 'kegg_pathway_genes.txt')) as f:
    for l in f:
        p = l.strip().split('\t')
        if len(p) == 2:
            kegg[p[0].replace('path:', '')].add(p[1].replace('ath:', ''))

# Load AraCyc pathway-gene memberships; skip header and placeholder genes
aracyc = defaultdict(set)
aracyc_file = os.path.join(DATA_DIR, 'aracyc_pathways.20251021')
if os.path.exists(aracyc_file):
    with open(aracyc_file) as f:
        f.readline()  # skip header row
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                pw, gene = parts[0], parts[6].upper()
                if gene != 'NIL' and gene.startswith('AT'):
                    aracyc[pw].add(gene)

# Merge KEGG + AraCyc into a single positive pathway set (min 5 genes each)
combined = {}
for k, v in kegg.items():
    if len(v) >= 5:
        combined[k] = v
for k, v in aracyc.items():
    if len(v) >= 5:
        combined[f"AC_{k}"] = v  # prefix avoids ID collisions with KEGG

bg = list(gene_go.keys())
total_genes = len(gene_go)
# Filter GO terms: too-small terms lack signal; too-broad terms lack specificity
upper_thresh = int(0.30 * total_genes)
go_filtered = sorted([t for t, g in go_genes.items() if 20 <= len(g) <= upper_thresh])
go_gene_sets = {t: set(g for g in gene_go if t in gene_go[g]) for t in go_filtered}
pw_sizes = [len(v) for v in combined.values()]

print(f"  Pathways: {len(combined)} (KEGG: {sum(1 for k in combined if not k.startswith('AC_'))}, "
      f"AraCyc: {sum(1 for k in combined if k.startswith('AC_'))})")
print(f"  Genes with GO: {total_genes}")
print(f"  Filtered GO terms: {len(go_filtered)}")


# ============================================================
# 2. BUILD GO ANNOTATION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("Step 2: Building GO annotation matrix")
print("=" * 70)

# Build a sparse binary gene x GO-term matrix, then apply TF-IDF weighting
# to down-weight ubiquitous GO terms and up-weight specific ones
gene_list = sorted(gene_go.keys())
gene_idx = {g: i for i, g in enumerate(gene_list)}
go_idx = {t: i for i, t in enumerate(go_filtered)}

rows, cols = [], []
for g, terms in gene_go.items():
    for t in terms:
        if t in go_idx:
            rows.append(gene_idx[g])
            cols.append(go_idx[t])

go_matrix = csr_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(len(gene_list), len(go_filtered))
)
tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
go_tfidf = tfidf.fit_transform(go_matrix)

print(f"  GO matrix: {go_matrix.shape}")
print(f"  TF-IDF matrix: {go_tfidf.shape}")


# ============================================================
# 3. COMPUTE ALL EMBEDDINGS
# ============================================================
print("\n" + "=" * 70)
print("Step 3: Computing embeddings")
print("=" * 70)

embeddings = {}

# --- 3a. SVD (linear baseline) --- Truncated SVD on TF-IDF weighted GO matrix
print("\n  [SVD] Linear embedding (d=20)...")
svd = TruncatedSVD(n_components=N_EMB, random_state=SEED)
emb_svd = svd.fit_transform(go_tfidf)
embeddings['SVD'] = emb_svd
print(f"    Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
print(f"    Shape: {emb_svd.shape}")

# --- 3b. UMAP (non-linear manifold learning) --- Pre-reduced with SVD-50 for speed
print("\n  [UMAP] Non-linear embedding (d=20)...")
try:
    import umap
    # Pre-reduce with SVD for speed
    svd_pre = TruncatedSVD(n_components=50, random_state=SEED)
    X_pre = svd_pre.fit_transform(go_tfidf)
    reducer = umap.UMAP(
        n_components=N_EMB, n_neighbors=15, min_dist=0.1,
        metric='euclidean', random_state=SEED, verbose=False,
        n_jobs=1, n_epochs=100  # reduced epochs for speed
    )
    emb_umap = reducer.fit_transform(X_pre)
    embeddings['UMAP'] = emb_umap
    print(f"    Shape: {emb_umap.shape}")
except ImportError:
    print("    SKIPPED: pip install umap-learn")
except Exception as e:
    print(f"    ERROR: {e}")

# --- 3c. Node2Vec on GO co-annotation graph --- Edge = genes sharing >= 3 GO terms
print("\n  [Node2Vec] GO co-annotation graph embedding (d=20)...")
try:
    import networkx as nx
    from node2vec import Node2Vec as N2V

    # Build gene co-annotation graph: edge if two genes share >=3 GO terms
    print("    Building co-annotation graph...")
    G = nx.Graph()
    G.add_nodes_from(gene_list)

    # For speed: sample edges from GO term co-occurrence
    # For each GO term, all genes annotated with it are connected
    edge_count = defaultdict(int)
    for term in go_filtered[:200]:  # top 200 most specific terms
        genes_with_term = list(go_gene_sets.get(term, set()))
        if len(genes_with_term) > 100:
            continue  # skip very broad terms
        for i in range(len(genes_with_term)):
            for j in range(i + 1, len(genes_with_term)):
                pair = tuple(sorted([genes_with_term[i], genes_with_term[j]]))
                edge_count[pair] += 1

    # Keep edges with >=3 shared GO terms
    for (g1, g2), count in edge_count.items():
        if count >= 3:
            G.add_edge(g1, g2, weight=count)

    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() > 100:
        n2v = N2V(G, dimensions=N_EMB_N2V, walk_length=30, num_walks=10,
                  workers=1, seed=SEED, quiet=True)
        model = n2v.fit(window=5, min_count=1, batch_words=4)

        # Build embedding matrix aligned with gene_list
        emb_n2v = np.zeros((len(gene_list), N_EMB_N2V))
        for g in gene_list:
            if g in model.wv:
                emb_n2v[gene_idx[g]] = model.wv[g]
        embeddings['Node2Vec'] = emb_n2v
        n_covered = sum(1 for g in gene_list if g in model.wv)
        print(f"    Shape: {emb_n2v.shape}, coverage: {n_covered}/{len(gene_list)}")
    else:
        print("    SKIPPED: too few edges in co-annotation graph")
except ImportError:
    print("    SKIPPED: pip install node2vec networkx gensim")
except Exception as e:
    print(f"    ERROR: {e}")

# --- 3d. Autoencoder (non-linear) --- Learns a compressed representation via reconstruction
print("\n  [Autoencoder] Non-linear embedding (d=20)...")
try:
    # Prefer PyTorch for proper gradient-based autoencoder; falls back to sklearn below
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class GOAutoencoder(nn.Module):
        """Bottleneck autoencoder: input -> 256 -> 64 -> latent -> 64 -> 256 -> input."""
        def __init__(self, input_dim, latent_dim=20):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, input_dim), nn.Sigmoid()
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    X_dense = go_tfidf.toarray().astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(X_dense))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    ae = GOAutoencoder(X_dense.shape[1], N_EMB_AE)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    ae.train()
    for epoch in range(30):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    ae.eval()
    with torch.no_grad():
        _, emb_ae = ae(torch.from_numpy(X_dense))
        emb_ae = emb_ae.numpy()
    embeddings['Autoencoder'] = emb_ae
    print(f"    Shape: {emb_ae.shape}, final loss: {total_loss:.4f}")

except ImportError:
    print("    PyTorch not available, using sklearn MLP autoencoder fallback...")
    try:
        # Fallback: sklearn MLPRegressor as bottleneck autoencoder
        X_dense = go_tfidf.toarray()
        # Reduce input first with SVD-100 for tractability
        svd100 = TruncatedSVD(n_components=100, random_state=SEED)
        X_100 = svd100.fit_transform(go_tfidf)

        ae = MLPRegressor(
            hidden_layer_sizes=(64, N_EMB_AE, 64),
            activation='relu', max_iter=200, random_state=SEED,
            early_stopping=True, validation_fraction=0.1, verbose=False
        )
        ae.fit(X_100, X_100)  # autoencoder: reconstruct input

        # Extract bottleneck activations
        from sklearn.neural_network._multilayer_perceptron import ACTIVATIONS
        X_curr = X_100
        for i, (W, b) in enumerate(zip(ae.coefs_[:2], ae.intercepts_[:2])):
            X_curr = X_curr @ W + b
            X_curr = np.maximum(X_curr, 0)  # ReLU
        emb_ae = X_curr  # bottleneck layer output
        embeddings['Autoencoder'] = emb_ae
        print(f"    Shape: {emb_ae.shape} (sklearn fallback)")
    except Exception as e:
        print(f"    ERROR: {e}")

print(f"\n  Available embeddings: {list(embeddings.keys())}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("Step 4: Building features")
print("=" * 70)


def jaccard_stats(gs, max_n=15):
    """Summarise pairwise GO-annotation Jaccard within a gene set.

    Subsamples to max_n genes to keep O(n^2) tractable for large sets.
    Returns [mean, std, min, max] -- four features capturing functional
    cohesion of the gene set.
    """
    genes = sorted(gs)
    if len(genes) > max_n:
        genes = list(np.random.choice(genes, max_n, replace=False))
    jacs = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            a, b = gene_go.get(genes[i], set()), gene_go.get(genes[j], set())
            u = len(a | b)
            if u > 0:
                jacs.append(len(a & b) / u)
    if not jacs:
        return [0, 0, 0, 0]
    return [np.mean(jacs), np.std(jacs), np.min(jacs), np.max(jacs)]


def embedding_features(gs, emb_matrix):
    """Aggregate gene-level embeddings into a single set-level vector via mean pooling."""
    idx = [gene_idx[g] for g in gs if g in gene_idx]
    if not idx:
        return np.zeros(emb_matrix.shape[1])
    return emb_matrix[idx].mean(axis=0)


def size_features(gs):
    """Raw and log-transformed gene-set size (controls for trivial size effects)."""
    return [float(len(gs)), float(np.log1p(len(gs)))]


# Reuse the current split-aware sampler from the canonical no-embedding
# pipeline. This keeps optional embedding diagnostics aligned with the active
# four-decoy design and prevents pure partial pathways from being treated as
# ordinary negatives.
print("  Generating current four-decoy benchmark samples...")
data_bundle = core.load_data()
records, sample_meta, split_info, tr_idx, te_idx = core.build_split_samples(data_bundle, seed=SEED)
all_sets = [set(record["genes"]) for record in records]
y_all = np.array([int(record["label"]) for record in records])
n_pos = int((y_all == 1).sum())
n_neg = int((y_all == 0).sum())
print(f"  Dataset: {len(all_sets)} samples ({n_pos} pos, {n_neg} neg)")
print(f"  Negative types: {sample_meta['negative_counts']}")

# Build base features (Jaccard + Size)
print("  Computing Jaccard + Size features...")
F_jac = np.array([jaccard_stats(s) for s in all_sets])
F_sz = np.array([size_features(s) for s in all_sets])
F_base = np.hstack([F_jac, F_sz])

# Build embedding features for each method
F_emb = {}
for emb_name, emb_matrix in embeddings.items():
    print(f"  Computing {emb_name} features (d={emb_matrix.shape[1]})...")
    F_emb[emb_name] = np.array([embedding_features(s, emb_matrix) for s in all_sets])

print(f"  Base features: d={F_base.shape[1]} (Jaccard=4, Size=2)")
for name, feat in F_emb.items():
    print(f"  {name} features: d={feat.shape[1]}")


# ============================================================
# 5. EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("Step 5: Evaluation (Repeated 5-fold x 5 CV + held-out test)")
print("=" * 70)

# Train/test indices come from core.build_split_samples(): positives are split
# first, then negatives are generated separately for train and test.

rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)


def evaluate(name, X):
    """Train XGBoost on a feature matrix and report CV + held-out AUROC.

    Two-stage evaluation to separate tuning variance from generalization:
      1. Repeated stratified CV on training split -> CV AUROC distribution
      2. Retrain on full training split, score held-out test set -> test AUROC
    SE is computed over the CV folds so confidence intervals stay conservative.
    """
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y_all[tr_idx], y_all[te_idx]

    # Stage 1: repeated CV to estimate expected performance and variance
    aucs = []
    for tr, va in rskf.split(Xtr, ytr):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            scale_pos_weight=2, min_child_weight=3,
            eval_metric='logloss', random_state=SEED, verbosity=0
        )
        m.fit(Xtr[tr], ytr[tr])
        aucs.append(roc_auc_score(ytr[va], m.predict_proba(Xtr[va])[:, 1]))

    # Stage 2: retrain on all training data, evaluate on unseen test split
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        scale_pos_weight=2, min_child_weight=3,
        eval_metric='logloss', random_state=SEED, verbosity=0
    )
    m.fit(Xtr, ytr)
    test_auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    # Standard error of the mean across CV folds
    se = np.std(aucs) / np.sqrt(len(aucs))

    return {
        'name': name, 'd': X.shape[1],
        'cv_mean': np.mean(aucs), 'cv_std': np.std(aucs), 'cv_se': se,
        'test': test_auc
    }


results = []

# Baseline: Jaccard + Size only
print("\n  Running evaluations...")
r = evaluate("Jaccard + Size (baseline)", F_base)
results.append(r)
baseline_test = r['test']
print(f"    {r['name']:50s}  d={r['d']:3d}  CV={r['cv_mean']:.3f}±{r['cv_se']:.3f}  Test={r['test']:.3f}")

# Each embedding alone
for emb_name, emb_feat in F_emb.items():
    r = evaluate(f"{emb_name} only", emb_feat)
    results.append(r)
    print(f"    {r['name']:50s}  d={r['d']:3d}  CV={r['cv_mean']:.3f}±{r['cv_se']:.3f}  Test={r['test']:.3f}")

# Base + each embedding
for emb_name, emb_feat in F_emb.items():
    X_combined = np.hstack([F_base, emb_feat])
    r = evaluate(f"Jaccard + Size + {emb_name}", X_combined)
    results.append(r)
    delta = r['test'] - baseline_test
    print(f"    {r['name']:50s}  d={r['d']:3d}  CV={r['cv_mean']:.3f}±{r['cv_se']:.3f}  "
          f"Test={r['test']:.3f}  Δ={delta:+.3f}")

# All embeddings combined
if len(F_emb) > 1:
    X_all_emb = np.hstack([F_base] + list(F_emb.values()))
    r = evaluate("Jaccard + Size + ALL embeddings", X_all_emb)
    results.append(r)
    delta = r['test'] - baseline_test
    print(f"    {r['name']:50s}  d={r['d']:3d}  CV={r['cv_mean']:.3f}±{r['cv_se']:.3f}  "
          f"Test={r['test']:.3f}  Δ={delta:+.3f}")


# ============================================================
# 6. RESULTS TABLE
# ============================================================
print("\n" + "=" * 70)
print("Step 6: Results Summary")
print("=" * 70)

df = pd.DataFrame(results)
df['delta_test'] = df['test'] - baseline_test
df['cv_report'] = df.apply(lambda r: f"{r['cv_mean']:.3f}±{r['cv_se']:.3f}", axis=1)

print("\n" + df[['name', 'd', 'cv_report', 'test', 'delta_test']].to_string(index=False))
df.to_csv(os.path.join(TABLE_DIR, 'embedding_comparison.csv'), index=False)
print(f"\n  Saved: {TABLE_DIR}/embedding_comparison.csv")


# ============================================================
# 7. FIGURE
# ============================================================
print("\n  Generating figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (A) Bar chart: Test AUROC comparison
ax = axes[0]
configs_plot = [r for r in results if 'only' not in r['name']]
names = [r['name'].replace('Jaccard + Size', 'J+S') for r in configs_plot]
aurocs = [r['test'] for r in configs_plot]
colors_map = {'baseline': '#2196F3', 'SVD': '#4CAF50', 'UMAP': '#FF9800',
              'Node2Vec': '#9C27B0', 'Autoencoder': '#F44336', 'ALL': '#795548'}

bar_colors = []
for n in names:
    matched = False
    for key, color in colors_map.items():
        if key.lower() in n.lower():
            bar_colors.append(color)
            matched = True
            break
    if not matched:
        bar_colors.append('#2196F3')

bars = ax.bar(range(len(names)), aurocs, color=bar_colors, alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Test AUROC', fontsize=11)
ax.set_title('(A) Embedding Method Comparison (Hard Negatives)', fontweight='bold')
ax.axhline(baseline_test, color='gray', linestyle='--', alpha=0.5, label=f'Baseline={baseline_test:.3f}')
ax.legend(fontsize=9)
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# (B) Embedding-only comparison
ax = axes[1]
emb_only = [r for r in results if 'only' in r['name']]
if emb_only:
    e_names = [r['name'].replace(' only', '') for r in emb_only]
    e_aurocs = [r['test'] for r in emb_only]
    e_colors = []
    for n in e_names:
        matched = False
        for key, color in colors_map.items():
            if key.lower() in n.lower():
                e_colors.append(color)
                matched = True
                break
        if not matched:
            e_colors.append('#999999')

    bars2 = ax.bar(range(len(e_names)), e_aurocs, color=e_colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(e_names)))
    ax.set_xticklabels(e_names, fontsize=10)
    ax.set_ylabel('Test AUROC', fontsize=11)
    ax.set_title('(B) Embedding Only (No Jaccard/Size)', fontweight='bold')
    for bar, val in zip(bars2, e_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'fig_embedding_comparison.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")


# ============================================================
# 8. CONCLUSIONS
# ============================================================
print("\n" + "=" * 70)
print("Conclusions")
print("=" * 70)

# Compare only "base + single embedding" configs (exclude ALL and embedding-only)
best_emb = max([r for r in results if '+' in r['name'] and 'ALL' not in r['name']],
               key=lambda r: r['test'])
worst_emb = min([r for r in results if '+' in r['name'] and 'ALL' not in r['name']],
                key=lambda r: r['test'])

print(f"""
  Baseline (Jaccard + Size, d=6):  Test = {baseline_test:.3f}
  Best embedding addition:         {best_emb['name']} (Test = {best_emb['test']:.3f}, Δ = {best_emb['test']-baseline_test:+.3f})
  Worst embedding addition:        {worst_emb['name']} (Test = {worst_emb['test']:.3f}, Δ = {worst_emb['test']-baseline_test:+.3f})

  Key finding:
""")

max_delta = max(r['test'] - baseline_test for r in results if '+' in r['name'])
if max_delta > 0.02:
    print("  Embedding provides meaningful improvement. Consider keeping the best method.")
elif max_delta > 0.005:
    print("  Embedding provides marginal improvement. May not justify added complexity.")
else:
    print("  No embedding method provides meaningful improvement over Jaccard + Size.")
    print("  This confirms that pathway coherence is primarily captured by pairwise")
    print("  Jaccard statistics, regardless of embedding method (linear or non-linear).")

print(f"\nDone. Results saved to {TABLE_DIR}/embedding_comparison.csv")
