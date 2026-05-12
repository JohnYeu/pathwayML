#!/usr/bin/env python3
"""Fast LOFO validation using the frozen seed-42 GO feature vocabulary.

This diagnostic tests whether an XGBoost model trained without a broad family of
positive pathways still gives held-out family pathways higher scores than mixed
constructed negatives. It uses the frozen seed-42 selected GO terms to isolate
family-holdout behaviour from feature-selection instability.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import family_table

TABLE_DIR=Path('tables'); FIG_DIR=Path('figures')

def main():
    TABLE_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    data=core.load_data()
    # Pre-build the full feature matrix using frozen seed-42 GO terms once,
    # then slice by index for each family (isolates holdout from feature-selection instability)
    records=json.loads((TABLE_DIR/'reproducibility'/'samples.json').read_text())
    selected=json.loads((TABLE_DIR/'reproducibility'/'selected_go_terms.json').read_text())
    X, feature_names, groups=core.build_feature_matrix(records, selected, data, seed=42)
    y=np.array([int(r['label']) for r in records])
    id_to_idx={r['id']:i for i,r in enumerate(records)}
    fmap=family_table(data)
    fmap.to_csv(TABLE_DIR/'pathway_family_assignment.csv', index=False)
    dist=fmap.groupby('family').agg(n_pathways=('pathway_id','count'), n_kegg=('source', lambda s:int((s=='KEGG').sum())), n_aracyc=('source', lambda s:int((s=='AraCyc').sum())), median_size=('n_genes','median'), median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways', ascending=False)
    dist.to_csv(TABLE_DIR/'table7_family_distribution.csv', index=False)
    neg_indices=[i for i,r in enumerate(records) if int(r['label'])==0]
    neg_by_type=defaultdict(list)
    for i in neg_indices:
        neg_by_type[str(records[i].get('type','unknown'))].append(i)
    rng=np.random.default_rng(42)
    rows=[]
    fams=dist[dist['n_pathways']>=10]['family'].tolist()
    for fam in fams:
        heldout_ids=set(fmap.loc[fmap['family']==fam,'pathway_id'])
        test_pos=[id_to_idx[pid] for pid in heldout_ids if pid in id_to_idx]
        train_pos=[id_to_idx[pid] for pid in data.pathways if pid not in heldout_ids and pid in id_to_idx]
        # Test negatives: balanced mix, 2x held-out positives.
        n_test_neg=2*len(test_pos)
        test_neg=[]
        per=max(1, n_test_neg//max(1,len(neg_by_type)))
        for t, inds in sorted(neg_by_type.items()):
            k=min(per, len(inds), n_test_neg-len(test_neg))
            if k>0:
                test_neg += [inds[j] for j in rng.choice(len(inds), size=k, replace=False)]
            if len(test_neg)>=n_test_neg: break
        if len(test_neg)<n_test_neg:
            remaining=[i for i in neg_indices if i not in set(test_neg)]
            k=min(n_test_neg-len(test_neg), len(remaining))
            test_neg += [remaining[j] for j in rng.choice(len(remaining), size=k, replace=False)]
        train_neg=[i for i in neg_indices if i not in set(test_neg)]
        # To maintain class ratio, sample at most 2x train positives.
        if len(train_neg)>2*len(train_pos):
            train_neg=[train_neg[j] for j in rng.choice(len(train_neg), size=2*len(train_pos), replace=False)]
        train_idx=np.array(train_pos+train_neg, dtype=int)
        test_idx=np.array(test_pos+test_neg, dtype=int)
        model=xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, scale_pos_weight=2, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', random_state=42, n_jobs=1, verbosity=0)
        model.fit(X[train_idx], y[train_idx])
        p=model.predict_proba(X[test_idx])[:,1]
        yy=y[test_idx]
        labels=(p>=0.5).astype(int)
        rows.append(dict(family=fam, n_heldout_pathways=len(test_pos), n_train_pathways=len(train_pos), n_test_negatives=len(test_neg), n_selected_go=len(selected), D=len(feature_names), median_size=float(dist.loc[dist['family']==fam,'median_size'].iloc[0]), median_jaccard=float(dist.loc[dist['family']==fam,'median_jaccard'].iloc[0]), test_auroc=float(roc_auc_score(yy,p)), test_auprc=float(average_precision_score(yy,p)), f1=float(f1_score(yy, labels)), heldout_positive_score_mean=float(p[yy==1].mean()), test_negative_score_mean=float(p[yy==0].mean())))
        print(f"{fam}: AUROC={rows[-1]['test_auroc']:.3f} AUPRC={rows[-1]['test_auprc']:.3f} n={len(test_pos)}", flush=True)
    df=pd.DataFrame(rows).sort_values('test_auroc')
    df.to_csv(TABLE_DIR/'table8_lofo_generalization.csv', index=False)
    # plot LOFO
    plot=df.sort_values('test_auroc')
    plt.figure(figsize=(9.0,5.4))
    pos=np.arange(len(plot)); plt.barh(pos, plot['test_auroc']);
    plt.yticks(pos, [f"{f} (n={n})" for f,n in zip(plot['family'], plot['n_heldout_pathways'])], fontsize=8)
    plt.xlim(0.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out diagnostic')
    for yi,v in zip(pos, plot['test_auroc']): plt.text(min(v+0.01,0.97), yi, f"{v:.3f}", va='center', fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png', dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    # plot dist top families
    top=dist.sort_values('n_pathways')
    plt.figure(figsize=(9.0,5.6)); pos=np.arange(len(top)); plt.barh(pos, top['n_pathways']); plt.yticks(pos, top['family'], fontsize=8); plt.xlabel('Curated pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png', dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()

if __name__=='__main__': main()
