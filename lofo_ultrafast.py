#!/usr/bin/env python3
"""Ultra-fast LOFO diagnostic with precomputed full-GO feature matrix.

Uses ALL frequency-filtered GO terms (no per-family selection) and a very
lightweight XGBoost (25 trees) for maximum speed. Includes a sensitivity
analysis that removes the dominant family entirely.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import family_table, fast_select_go_terms

TABLE_DIR=Path('tables'); FIG_DIR=Path('figures')

def main():
    t0=time.time()
    data=core.load_data(); print('data',time.time()-t0,flush=True)
    fmap=family_table(data); fmap.to_csv(TABLE_DIR/'pathway_family_assignment.csv',index=False)
    dist=fmap.groupby('family').agg(n_pathways=('pathway_id','count'), n_kegg=('source',lambda s:int((s=='KEGG').sum())), n_aracyc=('source',lambda s:int((s=='AraCyc').sum())), median_size=('n_genes','median'), median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways',ascending=False)
    dist.to_csv(TABLE_DIR/'table7_family_distribution.csv',index=False)
    # use saved canonical records
    records=json.loads((TABLE_DIR/'reproducibility/samples.json').read_text())
    id_to_idx={r['id']:i for i,r in enumerate(records)}
    y_all=np.array([int(r['label']) for r in records])
    # Precompute full filtered-GO features once; LOFO uses all 790 filtered GO terms to avoid label-based leakage and make diagnostic deterministic.
    selected_go=data.go_terms
    print('build X...',flush=True)
    X_all, feature_names, _=core.build_feature_matrix(records, selected_go, data, seed=42)
    print('X built', X_all.shape, time.time()-t0,flush=True)
    # split negative pool per family deterministically
    neg_indices=[i for i,r in enumerate(records) if int(r['label'])==0]
    neg_by_type=defaultdict(list)
    for i in neg_indices: neg_by_type[records[i].get('type','unknown')].append(i)
    rows=[]; rng=np.random.default_rng(42)
    for fam in dist.loc[dist['n_pathways']>=10,'family']:
        heldout=set(fmap.loc[fmap['family']==fam,'pathway_id'])
        pos_test=[id_to_idx[pid] for pid in heldout if pid in id_to_idx]
        pos_train=[id_to_idx[pid] for pid in data.pathways if pid not in heldout and pid in id_to_idx]
        # deterministic negative split: one test block per family
        test_neg=[]
        for typ,inds in sorted(neg_by_type.items()):
            k=max(1, int(round((2*len(pos_test))/4)))
            choose=rng.choice(inds, size=min(k,len(inds)), replace=False)
            test_neg.extend(list(map(int,choose)))
        test_neg=test_neg[:2*len(pos_test)]
        test_neg_set=set(test_neg)
        train_neg=[i for i in neg_indices if i not in test_neg_set]
        train_neg=train_neg[:2*len(pos_train)]
        train_idx=np.array(pos_train+train_neg,dtype=int)
        test_idx=np.array(pos_test+test_neg,dtype=int)
        model=XGBClassifier(n_estimators=25,max_depth=3,learning_rate=0.08,subsample=0.8,colsample_bytree=0.7,scale_pos_weight=2,min_child_weight=2,reg_lambda=1.0,eval_metric='logloss',random_state=42,n_jobs=1,verbosity=0)
        model.fit(X_all[train_idx], y_all[train_idx])
        p=model.predict_proba(X_all[test_idx])[:,1]
        yt=y_all[test_idx]; lab=(p>=0.5).astype(int)
        rows.append(dict(family=fam,n_heldout_pathways=len(pos_test),n_train_pathways=len(pos_train),n_test_negatives=len(test_neg),D=len(feature_names),n_go_terms=len(selected_go),median_size=float(dist.loc[dist['family']==fam,'median_size'].iloc[0]),median_jaccard=float(dist.loc[dist['family']==fam,'median_jaccard'].iloc[0]),test_auroc=float(roc_auc_score(yt,p)),test_auprc=float(average_precision_score(yt,p)),f1=float(f1_score(yt,lab)),positive_score_mean=float(p[yt==1].mean()),negative_score_mean=float(p[yt==0].mean())))
        print(f"{fam}: {rows[-1]['test_auroc']:.3f}", flush=True)
    df=pd.DataFrame(rows).sort_values('test_auroc',ascending=False)
    df.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
    # sensitivity: exclude largest family and train/test on canonical split using same full-GO feature set
    excl=str(dist.iloc[0]['family'])
    keep_pos=[id_to_idx[pid] for pid in data.pathways if pid in id_to_idx and fmap.set_index('pathway_id').loc[pid,'family'] != excl]
    neg=neg_indices[:2*len(keep_pos)]
    indices=np.array(keep_pos+neg,dtype=int); labels=y_all[indices]
    tr,te=train_test_split(np.arange(len(indices)),test_size=0.2,stratify=labels,random_state=42)
    model=XGBClassifier(n_estimators=50,max_depth=3,learning_rate=0.08,subsample=0.8,colsample_bytree=0.7,scale_pos_weight=2,min_child_weight=2,reg_lambda=1.0,eval_metric='logloss',random_state=42,n_jobs=1,verbosity=0)
    model.fit(X_all[indices[tr]],labels[tr]); pp=model.predict_proba(X_all[indices[te]])[:,1]
    sens=pd.DataFrame([dict(excluded_family=excl,n_positive_remaining=int((labels==1).sum()),n_negative_remaining=int((labels==0).sum()),D=len(feature_names),test_auroc=float(roc_auc_score(labels[te],pp)),test_auprc=float(average_precision_score(labels[te],pp)))])
    sens.to_csv(TABLE_DIR/'table9_sensitivity_without_dominant_family.csv',index=False)
    # plots
    plot=df.sort_values('test_auroc')
    plt.figure(figsize=(8.8,5.2)); y=np.arange(len(plot)); plt.barh(y,plot['test_auroc']); plt.yticks(y,[f"{f} (n={n})" for f,n in zip(plot['family'],plot['n_heldout_pathways'])],fontsize=8); plt.xlim(0,1); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out diagnostic');
    for yi,v in zip(y,plot['test_auroc']): plt.text(min(v+0.015,0.95),yi,f"{v:.3f}",va='center',fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    fs=dist.sort_values('n_pathways'); plt.figure(figsize=(8.8,5.5)); y=np.arange(len(fs)); plt.barh(y,fs['n_pathways']); plt.yticks(y,fs['family'],fontsize=8); plt.xlabel('Number of curated pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png',dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()
    summary={'lofo':df.to_dict(orient='records'),'family_distribution':dist.to_dict(orient='records'),'sensitivity':sens.to_dict(orient='records'),'notes':['LOFO uses the complete frequency-filtered GO term set (790 terms) plus Jaccard and size features to avoid label-based feature-selection leakage from held-out families.','The classifier is a lightweight no-embedding XGBoost diagnostic (25 trees), so LOFO should be interpreted as a generalisation stress test rather than the exact seed-42 reference model.']}
    (TABLE_DIR/'lofo_diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('done',time.time()-t0,flush=True)
if __name__=='__main__': main()
