#!/usr/bin/env python3
"""Optimized generalization analyses (per-negative-type + LOFO).

Key optimization: precomputes a binary GO annotation matrix once, then uses
matrix slicing for fast GO-frequency computation instead of per-record loops.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import assign_family

TABLE_DIR=Path('tables'); FIG_DIR=Path('figures')

def build_go_binary(data: core.DataBundle):
    """Precompute a binary gene x GO-term matrix for vectorized frequency calculation."""
    terms=list(data.go_terms)
    term_idx={t:i for i,t in enumerate(terms)}
    genes=list(data.background_genes)
    gene_idx={g:i for i,g in enumerate(genes)}
    M=np.zeros((len(genes), len(terms)), dtype=np.float32)
    for g, gos in data.gene_go.items():
        gi=gene_idx.get(g)
        if gi is None: continue
        for t in gos:
            j=term_idx.get(t)
            if j is not None:
                M[gi,j]=1.0
    return genes, gene_idx, terms, M

def go_freq_matrix(records: Sequence[Dict[str,Any]], gene_idx, M):
    """Compute GO-frequency features for all records via matrix mean (vectorized)."""
    out=np.zeros((len(records), M.shape[1]), dtype=np.float32)
    for i,r in enumerate(records):
        idx=[gene_idx[g] for g in r['genes'] if g in gene_idx]
        if idx:
            out[i,:]=M[idx,:].mean(axis=0)
    return out

def jac_size_matrix(records, data, seed):
    """Batch-compute Jaccard statistics and size features for all records."""
    jac=np.vstack([core.jaccard_stats(r['genes'], data.gene_go, salt=f"fast:{r['id']}", seed=seed) for r in records]).astype(np.float32)
    size=np.vstack([core.size_features(r['genes']) for r in records]).astype(np.float32)
    return jac,size

def select_indices(X_go_train, y_train, seed, frac=0.70):
    """Select GO-term column indices via variance filter + MI ranking (70% cumulative MI)."""
    vt=VarianceThreshold(threshold=0.001)
    X_vt=vt.fit_transform(X_go_train)
    support=np.where(vt.get_support())[0]
    mi=mutual_info_classif(X_vt, y_train, random_state=seed, n_neighbors=5)
    order=np.argsort(mi)[::-1]
    total=float(mi[order].sum())
    if total<=0:
        k=min(20,len(order))
    else:
        k=int(np.argmax(np.cumsum(mi[order])/total>=frac)+1)
    k=max(3,min(k,len(order)))
    return support[order[:k]]

def make_fast_xgb(seed):
    """Lightweight XGBoost for fast LOFO diagnostics (120 trees)."""
    return xgb.XGBClassifier(n_estimators=120,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.7,scale_pos_weight=2,min_child_weight=3,reg_alpha=0.1,reg_lambda=1.0,eval_metric='logloss',random_state=seed,n_jobs=1,verbosity=0)

def per_negative_type_fast(data, gene_idx, M):
    """Decompose the canonical seed-42 XGBoost test set by negative type.

    This intentionally reads the saved seed-42 artifacts from the main
    reproducible pipeline rather than rebuilding a fast approximation. The
    "All mixed negatives" row should therefore exactly match the seed-42
    XGBoost result in tables/results_no_embedding.json.
    """
    seed=42
    repro=TABLE_DIR/'reproducibility'
    records=json.loads((repro/'samples.json').read_text(encoding='utf-8'))
    splits=json.loads((repro/'splits.json').read_text(encoding='utf-8'))
    selected=json.loads((repro/'selected_go_terms.json').read_text(encoding='utf-8'))

    X, _feature_names, _groups=core.build_feature_matrix(records, selected, data, seed=seed)
    y=np.array([int(r['label']) for r in records], dtype=int)
    id_to_idx={str(r['id']):i for i,r in enumerate(records)}
    train_idx=np.array([id_to_idx[str(x)] for x in splits['train_ids']], dtype=int)
    test_idx=np.array([id_to_idx[str(x)] for x in splits['test_ids']], dtype=int)

    model=core.make_models(seed)['XGBoost']
    model.fit(X[train_idx], y[train_idx])
    scores=model.predict_proba(X[test_idx])[:,1]
    yt=y[test_idx]
    types=np.array([records[int(i)]['type'] for i in test_idx])
    rows=[]
    def add(name,mask):
        labels=yt[mask]; sc=scores[mask]
        rows.append(dict(comparison=name,n_positive=int((labels==1).sum()),n_negative=int((labels==0).sum()),
                         test_auroc=float(roc_auc_score(labels,sc)), test_auprc=float(average_precision_score(labels,sc)),
                         negative_score_median=float(np.median(sc[labels==0])),
                         negative_score_iqr=f"{np.percentile(sc[labels==0],25):.3f}-{np.percentile(sc[labels==0],75):.3f}"))
    add('All mixed negatives', np.ones(len(yt), dtype=bool))
    pos=yt==1
    for nt in ['jaccard_matched','co_annotation','chimera','shuffled']:
        add(nt.replace('_',' '), pos | ((yt==0)&(types==nt)))
    df=pd.DataFrame(rows)
    df.to_csv(TABLE_DIR/'table6_negative_type_performance.csv',index=False)
    plot_df=df[df.comparison!='All mixed negatives']
    plt.figure(figsize=(8,4.5)); plt.bar(plot_df.comparison, plot_df.test_auroc); plt.ylim(0.5,1.0); plt.ylabel('Held-out AUROC vs positives'); plt.xticks(rotation=25,ha='right'); plt.title('Performance by negative-set type (canonical seed 42)')
    for i,v in enumerate(plot_df.test_auroc): plt.text(i,v+0.01,f'{v:.3f}',ha='center',fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig11_negative_type_performance.png',dpi=300); plt.savefig(FIG_DIR/'fig11_negative_type_performance.pdf'); plt.close()
    return df

def family_table(data):
    """Map each pathway to its coarse family with source, size, and coherence."""
    rows=[]
    for pid, genes in data.pathways.items():
        name=data.pathway_names.get(pid,pid); fam=assign_family(pid,name); src='AraCyc' if pid.startswith('AC_') else 'KEGG'
        rows.append(dict(pathway_id=pid,pathway_name=name,family=fam,source=src,n_genes=len(genes),jaccard_mean=core.pathway_jaccard_mean(genes,data.gene_go,salt=f'fam:{pid}',seed=42)))
    return pd.DataFrame(rows)

def lofo_fast(data, gene_idx, M, min_family_size=10, use_cached=False, save_canonical=True):
    """Optimized LOFO: precomputes all feature matrices once, then slices per family."""
    canonical=TABLE_DIR/'table8_lofo_generalization_training_only.csv'
    if use_cached and canonical.exists():
        df=pd.read_csv(canonical)
        df.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
        (TABLE_DIR/'lofo_generalization_summary.json').write_text(json.dumps(df.to_dict(orient='records'),indent=2,sort_keys=True),encoding='utf-8')
        plot_df=df.sort_values('test_auroc')
        plt.figure(figsize=(8.5,5.2)); plt.barh(plot_df.family,plot_df.test_auroc); plt.xlim(0.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out generalisation')
        for i,v in enumerate(plot_df.test_auroc): plt.text(v+0.006,i,f'{v:.3f}',va='center',fontsize=9)
        plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
        return df

    seed=42
    fmap=family_table(data); fmap.to_csv(TABLE_DIR/'pathway_family_assignment.csv',index=False)
    famsum=(fmap.groupby('family').agg(n_pathways=('pathway_id','count'),n_kegg=('source',lambda s:int((s=='KEGG').sum())),n_aracyc=('source',lambda s:int((s=='AraCyc').sum())),median_size=('n_genes','median'),median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways',ascending=False))
    famsum.to_csv(TABLE_DIR/'table7_family_distribution.csv',index=False)
    train_pool,_=core.build_samples(data,seed=seed+3000)
    test_pool,_=core.build_samples(data,seed=seed+4000)
    neg_train=[r for r in train_pool if r['label']==0]
    neg_test=[r for r in test_pool if r['label']==0]
    # Precompute negative pool matrices once
    neg_train_go=go_freq_matrix(neg_train,gene_idx,M); neg_train_j,neg_train_s=jac_size_matrix(neg_train,data,seed)
    neg_test_go=go_freq_matrix(neg_test,gene_idx,M); neg_test_j,neg_test_s=jac_size_matrix(neg_test,data,seed)
    # Positive records matrix once
    pos_records=[dict(id=pid,label=1,type='curated_pathway',name=data.pathway_names.get(pid,pid),genes=sorted(genes)) for pid,genes in data.pathways.items()]
    pos_ids=[r['id'] for r in pos_records]
    pos_go=go_freq_matrix(pos_records,gene_idx,M); pos_j,pos_s=jac_size_matrix(pos_records,data,seed)
    fam_by_pid=dict(zip(fmap.pathway_id,fmap.family))
    rng=np.random.default_rng(seed+5000)
    rows=[]
    for _,fr in famsum.iterrows():
        fam=str(fr.family); nheld=int(fr.n_pathways)
        if nheld<min_family_size: continue
        held=np.array([fam_by_pid[pid]==fam for pid in pos_ids])
        train_pos_idx=np.where(~held)[0]; test_pos_idx=np.where(held)[0]
        ntrneg=min(len(neg_train),2*len(train_pos_idx)); nteneg=min(len(neg_test),2*len(test_pos_idx))
        trnidx=rng.choice(len(neg_train),size=ntrneg,replace=False); tenidx=rng.choice(len(neg_test),size=nteneg,replace=False)
        Xgo_train=np.vstack([pos_go[train_pos_idx],neg_train_go[trnidx]])
        y_train=np.r_[np.ones(len(train_pos_idx),dtype=int),np.zeros(ntrneg,dtype=int)]
        sel=select_indices(Xgo_train,y_train,seed+sum(map(ord,fam)))
        X_train=np.hstack([Xgo_train[:,sel],np.vstack([pos_j[train_pos_idx],neg_train_j[trnidx]]),np.vstack([pos_s[train_pos_idx],neg_train_s[trnidx]])])
        Xgo_test=np.vstack([pos_go[test_pos_idx],neg_test_go[tenidx]])
        X_test=np.hstack([Xgo_test[:,sel],np.vstack([pos_j[test_pos_idx],neg_test_j[tenidx]]),np.vstack([pos_s[test_pos_idx],neg_test_s[tenidx]])])
        y_test=np.r_[np.ones(len(test_pos_idx),dtype=int),np.zeros(nteneg,dtype=int)]
        model=make_fast_xgb(seed); model.fit(X_train,y_train); scores=model.predict_proba(X_test)[:,1]; pred=(scores>=0.5).astype(int)
        rows.append(dict(family=fam,n_heldout_pathways=int(len(test_pos_idx)),n_test_negatives=int(nteneg),n_train_pathways=int(len(train_pos_idx)),n_go_selected=int(len(sel)),D=int(len(sel)+6),median_size=float(fr.median_size),median_jaccard=float(fr.median_jaccard),test_auroc=float(roc_auc_score(y_test,scores)),test_auprc=float(average_precision_score(y_test,scores)),f1=float(f1_score(y_test,pred)),precision=float(precision_score(y_test,pred,zero_division=0)),recall=float(recall_score(y_test,pred,zero_division=0)),positive_score_median=float(np.median(scores[y_test==1])),negative_score_median=float(np.median(scores[y_test==0]))))
    df=pd.DataFrame(rows).sort_values('test_auroc',ascending=False); df.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
    if save_canonical:
        df.to_csv(canonical,index=False)
    (TABLE_DIR/'lofo_generalization_summary.json').write_text(json.dumps(df.to_dict(orient='records'),indent=2,sort_keys=True),encoding='utf-8')
    plot_df=df.sort_values('test_auroc')
    plt.figure(figsize=(8.5,5.2)); plt.barh(plot_df.family,plot_df.test_auroc); plt.xlim(0.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out generalisation')
    for i,v in enumerate(plot_df.test_auroc): plt.text(v+0.006,i,f'{v:.3f}',va='center',fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    return df

def main():
    parser=argparse.ArgumentParser(description='No-embedding generalization analyses.')
    parser.add_argument('--use-cached-lofo', action='store_true', help='Read tables/table8_lofo_generalization_training_only.csv instead of recomputing LOFO.')
    parser.add_argument('--no-save-lofo-canonical', action='store_true', help='Do not update tables/table8_lofo_generalization_training_only.csv after recomputing LOFO.')
    args=parser.parse_args()
    TABLE_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    data=core.load_data(); genes,gene_idx,terms,M=build_go_binary(data)
    neg=per_negative_type_fast(data,gene_idx,M)
    lofo=lofo_fast(data,gene_idx,M,use_cached=args.use_cached_lofo,save_canonical=not args.no_save_lofo_canonical)
    summary={'negative_type_performance':neg.to_dict(orient='records'),'lofo_generalization':lofo.to_dict(orient='records'), 'note':'Additional analyses; final manuscript uses these as sensitivity/generalization checks.'}
    (TABLE_DIR/'generalization_analysis_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    print(neg.to_string(index=False)); print('\n'); print(lofo.to_string(index=False))
if __name__=='__main__': main()
