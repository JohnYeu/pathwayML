#!/usr/bin/env python3
"""LOFO with per-family training-only GO selection and XGBoost.

For each held-out family, GO terms are re-selected on the training subset
using 70% cumulative MI. Uses saved seed-42 samples with separate train/test
negative splits.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import assign_family
TABLE_DIR=Path('tables'); FIG_DIR=Path('figures'); TABLE_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

def select_terms(train_records,data,seed,frac=.70):
    """Select GO terms via variance filter + MI ranking with cumulative MI fraction cutoff."""
    y=np.array([r['label'] for r in train_records])
    F=np.vstack([core.go_frequency(r['genes'],data.go_terms,data.gene_go) for r in train_records])
    vt=VarianceThreshold(threshold=.001); F2=vt.fit_transform(F)
    terms=[t for t,keep in zip(data.go_terms,vt.get_support()) if keep]
    mi=mutual_info_classif(F2,y,random_state=seed,n_neighbors=5)
    order=np.argsort(mi)[::-1]; s=mi[order]; total=float(s.sum())
    k=int(np.argmax(np.cumsum(s)/total>=frac)+1) if total>0 else min(20,len(terms))
    k=max(3,min(k,len(terms)))
    return [terms[int(i)] for i in order[:k]]

def metric(y,s):
    """Compute classification metrics from true labels and predicted scores."""
    p=(s>=.5).astype(int)
    return dict(test_auroc=float(roc_auc_score(y,s)), test_auprc=float(average_precision_score(y,s)), f1=float(f1_score(y,p)), precision=float(precision_score(y,p,zero_division=0)), recall=float(recall_score(y,p,zero_division=0)), positive_score_median=float(np.median(s[y==1])), negative_score_median=float(np.median(s[y==0])))

def main():
    data=core.load_data(); records=json.load(open(TABLE_DIR/'reproducibility'/'samples.json'))
    pos=[]; neg=[]; fam_rows=[]
    for rec in records:
        if rec['label']==1:
            pid=rec['id']; name=data.pathway_names.get(pid,rec.get('name',pid)); fam=assign_family(pid,name); rec['family']=fam; pos.append(rec)
            fam_rows.append({'pathway_id':pid,'pathway_name':name,'family':fam,'source':'AraCyc' if pid.startswith('AC_') else 'KEGG','n_genes':len(rec['genes']),'jaccard_mean':core.pathway_jaccard_mean(rec['genes'],data.gene_go,salt=f'fam:{pid}',seed=42)})
        else: neg.append(rec)
    fam_df=pd.DataFrame(fam_rows); fam_df.to_csv(TABLE_DIR/'pathway_family_assignment.csv',index=False)
    dist=fam_df.groupby('family').agg(n_pathways=('pathway_id','count'),n_kegg=('source',lambda s:int((s=='KEGG').sum())),n_aracyc=('source',lambda s:int((s=='AraCyc').sum())),median_size=('n_genes','median'),median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways',ascending=False)
    dist.to_csv(TABLE_DIR/'table7_family_distribution.csv',index=False)
    rng=np.random.default_rng(42); out=[]
    for j,fam in enumerate(dist.loc[dist.n_pathways>=10,'family'],1):
        train_pos=[r for r in pos if r['family']!=fam]; test_pos=[r for r in pos if r['family']==fam]
        train_neg=[neg[int(i)] for i in rng.choice(len(neg),size=min(len(neg),2*len(train_pos)),replace=False)]
        train_ids={r['id'] for r in train_neg}; rem=[r for r in neg if r['id'] not in train_ids]
        if len(rem)<2*len(test_pos): rem=neg
        test_neg=[rem[int(i)] for i in rng.choice(len(rem),size=min(len(rem),2*len(test_pos)),replace=False)]
        train=train_pos+train_neg; test=test_pos+test_neg
        selected=select_terms(train,data,seed=42+j,frac=.70)
        Xtr,names,_=core.build_feature_matrix(train,selected,data,seed=42); Xte,_,_=core.build_feature_matrix(test,selected,data,seed=42)
        ytr=np.array([r['label'] for r in train]); yte=np.array([r['label'] for r in test])
        model=xgb.XGBClassifier(n_estimators=80,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.7,scale_pos_weight=2,min_child_weight=3,reg_alpha=.1,reg_lambda=1.0,eval_metric='logloss',random_state=42+j,n_jobs=4,verbosity=0,tree_method='hist')
        model.fit(Xtr,ytr); score=model.predict_proba(Xte)[:,1]
        d=metric(yte,score); row={'family':fam,'n_heldout_pathways':len(test_pos),'n_test_negatives':len(test_neg),'n_train_pathways':len(train_pos),'n_go_selected':len(selected),'D':len(names),'median_size':float(dist.loc[dist.family==fam,'median_size'].iloc[0]),'median_jaccard':float(dist.loc[dist.family==fam,'median_jaccard'].iloc[0]),**d}
        out.append(row); print(j,fam,d['test_auroc'],len(selected),flush=True)
    out=pd.DataFrame(out).sort_values('test_auroc',ascending=False); out.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
    with open(TABLE_DIR/'lofo_generalization_summary.json','w') as f: json.dump(out.to_dict(orient='records'),f,indent=2)
    plot=out.sort_values('test_auroc')
    plt.figure(figsize=(8.5,5.2)); plt.barh(plot['family'],plot['test_auroc']); plt.xlim(.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out validation')
    for i,v in enumerate(plot['test_auroc']): plt.text(min(v+.006,.985),i,f'{v:.3f}',va='center',fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    top=dist.sort_values('n_pathways',ascending=True)
    plt.figure(figsize=(8.5,5.2)); plt.barh(top['family'],top['n_pathways']); plt.xlabel('Number of curated pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png',dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()
if __name__=='__main__': main()
