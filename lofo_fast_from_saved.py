#!/usr/bin/env python3
"""Fast LOFO validation using the saved seed-42 sample pool.

This avoids regenerating hard negatives and therefore runs quickly. Positive
pathways from one broad family are held out; the model is trained on all other
positive pathways plus mixed generated negatives from the saved sample pool.
GO feature selection is fitted on the training records only using the same
variance + MI principle as the main pipeline, with a fixed 70% cumulative-MI
cutoff for speed.
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
from generalization_and_negative_analysis import family_table, fast_select_go_terms

TABLE_DIR=Path('tables'); FIG_DIR=Path('figures')

def load_records():
    """Load saved seed-42 samples; falls back to regeneration if not found."""
    p=TABLE_DIR/'reproducibility'/'samples.json'
    if p.exists():
        return json.loads(p.read_text())
    records,_=core.build_samples(core.load_data(), seed=42)
    return records

def stratified_take(ids_by_type, n_needed, rng):
    """Select n_needed IDs balanced across negative types."""
    types=sorted(ids_by_type)
    selected=[]
    if not types or n_needed<=0: return selected
    per=max(1, n_needed//len(types))
    for t in types:
        ids=list(ids_by_type[t])
        if not ids: continue
        k=min(per, len(ids), n_needed-len(selected))
        if k>0:
            selected.extend([ids[i] for i in rng.choice(len(ids), size=k, replace=False)])
        if len(selected)>=n_needed: break
    if len(selected)<n_needed:
        remaining=[]
        already=set(selected)
        for ids in ids_by_type.values():
            remaining.extend([x for x in ids if x not in already])
        if remaining:
            k=min(n_needed-len(selected), len(remaining))
            selected.extend([remaining[i] for i in rng.choice(len(remaining), size=k, replace=False)])
    return selected

def main():
    TABLE_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    data=core.load_data()
    records=load_records()
    fmap=family_table(data)
    fmap.to_csv(TABLE_DIR/'pathway_family_assignment.csv', index=False)
    dist=fmap.groupby('family').agg(n_pathways=('pathway_id','count'), n_kegg=('source', lambda s:int((s=='KEGG').sum())), n_aracyc=('source', lambda s:int((s=='AraCyc').sum())), median_size=('n_genes','median'), median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways', ascending=False)
    dist.to_csv(TABLE_DIR/'table7_family_distribution.csv', index=False)
    id_to_rec={r['id']: r for r in records}
    neg_by_type=defaultdict(list)
    for r in records:
        if int(r['label'])==0:
            neg_by_type[str(r.get('type','unknown'))].append(r['id'])
    rng=np.random.default_rng(42)
    rows=[]
    fams=dist[dist['n_pathways']>=10]['family'].tolist()[:3]  # Diagnostic LOFO: three largest coarse families
    for fam in fams:
        heldout=set(fmap.loc[fmap['family']==fam,'pathway_id'])
        test_pos=[id_to_rec[pid] for pid in heldout if pid in id_to_rec]
        train_pos=[id_to_rec[pid] for pid in data.pathways if pid not in heldout and pid in id_to_rec]
        n_test_neg=2*len(test_pos)
        test_neg_ids=stratified_take(neg_by_type, n_test_neg, rng)
        train_neg_by_type={t:[x for x in ids if x not in set(test_neg_ids)] for t,ids in neg_by_type.items()}
        train_neg_ids=stratified_take(train_neg_by_type, min(2*len(train_pos), sum(len(v) for v in train_neg_by_type.values())), rng)
        train_records=train_pos+[id_to_rec[i] for i in train_neg_ids]
        test_records=test_pos+[id_to_rec[i] for i in test_neg_ids]
        selected=fast_select_go_terms(train_records, data, seed=42, mi_fraction=0.70)
        X_train, feature_names, _=core.build_feature_matrix(train_records, selected, data, seed=42)
        y_train=np.array([int(r['label']) for r in train_records])
        X_test, _, _=core.build_feature_matrix(test_records, selected, data, seed=42)
        y_test=np.array([int(r['label']) for r in test_records])
        model=xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, scale_pos_weight=2, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', random_state=42, n_jobs=1, verbosity=0)
        model.fit(X_train, y_train)
        p=model.predict_proba(X_test)[:,1]
        lab=(p>=0.5).astype(int)
        pos_scores=p[y_test==1]; neg_scores=p[y_test==0]
        rows.append(dict(family=fam, n_heldout_pathways=len(test_pos), n_train_pathways=len(train_pos), n_test_negatives=len(test_neg_ids), n_go_selected=len(selected), D=len(feature_names), median_size=float(dist.loc[dist['family']==fam,'median_size'].iloc[0]), median_jaccard=float(dist.loc[dist['family']==fam,'median_jaccard'].iloc[0]), test_auroc=float(roc_auc_score(y_test,p)), test_auprc=float(average_precision_score(y_test,p)), f1=float(f1_score(y_test, lab)), heldout_positive_score_mean=float(pos_scores.mean()), test_negative_score_mean=float(neg_scores.mean())))
        print(f"{fam}: AUROC={rows[-1]['test_auroc']:.3f} AUPRC={rows[-1]['test_auprc']:.3f} n={len(test_pos)}", flush=True)
    df=pd.DataFrame(rows).sort_values('test_auroc')
    df.to_csv(TABLE_DIR/'table8_lofo_generalization.csv', index=False)
    # plots
    plot=df.sort_values('test_auroc')
    plt.figure(figsize=(8.8,5.2))
    y=np.arange(len(plot))
    plt.barh(y, plot['test_auroc'])
    plt.yticks(y, [f"{f} (n={n})" for f,n in zip(plot['family'], plot['n_heldout_pathways'])], fontsize=8)
    plt.xlim(0.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out generalisation')
    for yi,v in zip(y, plot['test_auroc']): plt.text(min(v+0.01,0.97), yi, f"{v:.3f}", va='center', fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png', dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    top=dist.sort_values('n_pathways').copy()
    plt.figure(figsize=(8.8,5.5)); y=np.arange(len(top)); plt.barh(y, top['n_pathways']); plt.yticks(y, top['family'], fontsize=8); plt.xlabel('Number of pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png', dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()
if __name__=='__main__': main()
