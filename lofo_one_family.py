#!/usr/bin/env python3
"""Single-family LOFO evaluation.

Usage: python lofo_one_family.py "Family Name"

Evaluates one held-out family with training-only GO selection and full-strength
XGBoost (500 trees). Appends results to tables/lofo_500_partial.csv across
multiple runs.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import xgboost as xgb
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import assign_family

FAM=sys.argv[1]
seed=42
TABLE_DIR=Path('tables')

def build_go_binary(data):
    """Precompute binary gene x GO-term matrix for vectorized frequency calculation."""
    terms=list(data.go_terms); term_idx={t:i for i,t in enumerate(terms)}; genes=list(data.background_genes); gene_idx={g:i for i,g in enumerate(genes)}
    M=np.zeros((len(genes),len(terms)),dtype=np.float32)
    for g,gos in data.gene_go.items():
        gi=gene_idx.get(g)
        if gi is None: continue
        for t in gos:
            j=term_idx.get(t)
            if j is not None: M[gi,j]=1.0
    return gene_idx,M

def go_freq(records,gene_idx,M):
    """Vectorized GO frequency via matrix mean over member gene rows."""
    out=np.zeros((len(records),M.shape[1]),dtype=np.float32)
    for i,r in enumerate(records):
        idx=[gene_idx[g] for g in r['genes'] if g in gene_idx]
        if idx: out[i]=M[idx].mean(axis=0)
    return out

def jac_size(records,data):
    jac=np.vstack([core.jaccard_stats(r['genes'],data.gene_go,salt=f'lofo:{r["id"]}',seed=seed) for r in records]).astype(np.float32)
    sz=np.vstack([core.size_features(r['genes']) for r in records]).astype(np.float32)
    return jac,sz

def select(X,y):
    """Select GO-term column indices via variance filter + 70% cumulative MI."""
    vt=VarianceThreshold(0.001); Xv=vt.fit_transform(X); support=np.where(vt.get_support())[0]
    mi=mutual_info_classif(Xv,y,random_state=seed+sum(map(ord,FAM)),n_neighbors=5); order=np.argsort(mi)[::-1]
    total=float(mi[order].sum()); k=int(np.argmax(np.cumsum(mi[order])/total>=0.70)+1) if total>0 else min(20,len(order))
    k=max(3,min(k,len(order))); return support[order[:k]]

data=core.load_data(); gene_idx,M=build_go_binary(data)
# family assignment
pos_records=[]; fams=[]; jmeans=[]
for pid,genes in data.pathways.items():
    name=data.pathway_names.get(pid,pid); fam=assign_family(pid,name)
    rec=dict(id=pid,label=1,type='curated_pathway',name=name,genes=sorted(genes))
    pos_records.append(rec); fams.append(fam); jmeans.append(core.pathway_jaccard_mean(genes,data.gene_go,salt=f'fam:{pid}',seed=seed))
if FAM not in set(fams):
    print('unknown',FAM); sys.exit(2)
train_pool,_=core.build_samples(data,seed=seed+3000); test_pool,_=core.build_samples(data,seed=seed+4000)
neg_train=[r for r in train_pool if r['label']==0]; neg_test=[r for r in test_pool if r['label']==0]
rng=np.random.default_rng(seed+5000+sum(map(ord,FAM)))
held=np.array([f==FAM for f in fams]); train_pos_idx=np.where(~held)[0]; test_pos_idx=np.where(held)[0]
ntrneg=min(len(neg_train),2*len(train_pos_idx)); nteneg=min(len(neg_test),2*len(test_pos_idx))
trnidx=rng.choice(len(neg_train),size=ntrneg,replace=False); tenidx=rng.choice(len(neg_test),size=nteneg,replace=False)
train_records=[pos_records[i] for i in train_pos_idx]+[neg_train[int(i)] for i in trnidx]
test_records=[pos_records[i] for i in test_pos_idx]+[neg_test[int(i)] for i in tenidx]
Xgo_train=go_freq(train_records,gene_idx,M); y_train=np.array([r['label'] for r in train_records])
sel=select(Xgo_train,y_train)
trj,trs=jac_size(train_records,data); tej,tes=jac_size(test_records,data); Xgo_test=go_freq(test_records,gene_idx,M)
X_train=np.hstack([Xgo_train[:,sel],trj,trs]); X_test=np.hstack([Xgo_test[:,sel],tej,tes]); y_test=np.array([r['label'] for r in test_records])
model=xgb.XGBClassifier(n_estimators=500,max_depth=5,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,scale_pos_weight=2,min_child_weight=3,reg_alpha=0.1,reg_lambda=1.0,eval_metric='logloss',random_state=seed,n_jobs=1,verbosity=0)
model.fit(X_train,y_train); scores=model.predict_proba(X_test)[:,1]; pred=(scores>=0.5).astype(int)
row=dict(family=FAM,n_heldout_pathways=int(len(test_pos_idx)),n_test_negatives=int(nteneg),n_train_pathways=int(len(train_pos_idx)),n_go_selected=int(len(sel)),D=int(len(sel)+6),median_size=float(np.median([len(pos_records[i]['genes']) for i in test_pos_idx])),median_jaccard=float(np.median(np.array(jmeans)[test_pos_idx])),test_auroc=float(roc_auc_score(y_test,scores)),test_auprc=float(average_precision_score(y_test,scores)),f1=float(f1_score(y_test,pred)),precision=float(precision_score(y_test,pred,zero_division=0)),recall=float(recall_score(y_test,pred,zero_division=0)),positive_score_median=float(np.median(scores[y_test==1])),negative_score_median=float(np.median(scores[y_test==0])))
out=TABLE_DIR/'lofo_500_partial.csv'
if out.exists():
    df=pd.read_csv(out); df=df[df.family!=FAM]; df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
else:
    df=pd.DataFrame([row])
df.to_csv(out,index=False)
print(json.dumps(row,indent=2))
