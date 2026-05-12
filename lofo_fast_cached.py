#!/usr/bin/env python3
"""Fast LOFO using all filtered GO terms and fresh negative pools.

Uses data.go_terms (all frequency-filtered GO terms) to avoid supervised
feature-selection leakage. Generates separate negative pools from seeds
3042 (train) and 4042 (test). Lightweight XGBoost (60 trees, hist method).
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import assign_family
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
TABLE_DIR=Path('tables'); FIG_DIR=Path('figures'); TABLE_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

def metric(y,score):
    """Compute classification metrics from true labels and predicted scores."""
    p=(score>=.5).astype(int)
    return dict(test_auroc=float(roc_auc_score(y,score)), test_auprc=float(average_precision_score(y,score)), f1=float(f1_score(y,p)), precision=float(precision_score(y,p,zero_division=0)), recall=float(recall_score(y,p,zero_division=0)), positive_score_median=float(np.median(score[y==1])), negative_score_median=float(np.median(score[y==0])))

def main():
    data=core.load_data()
    pos=[]; fam_rows=[]
    for pid,genes in data.pathways.items():
        name=data.pathway_names.get(pid,pid); fam=assign_family(pid,name); src='AraCyc' if pid.startswith('AC_') else 'KEGG'
        rec={'id':pid,'label':1,'type':'curated_pathway','name':name,'genes':sorted(genes),'family':fam}
        pos.append(rec); fam_rows.append({'pathway_id':pid,'pathway_name':name,'family':fam,'source':src,'n_genes':len(genes),'jaccard_mean':core.pathway_jaccard_mean(genes,data.gene_go,salt=f'fam:{pid}',seed=42)})
    fam_df=pd.DataFrame(fam_rows); fam_df.to_csv(TABLE_DIR/'pathway_family_assignment.csv',index=False)
    dist=fam_df.groupby('family').agg(n_pathways=('pathway_id','count'),n_kegg=('source',lambda s:int((s=='KEGG').sum())),n_aracyc=('source',lambda s:int((s=='AraCyc').sum())),median_size=('n_genes','median'),median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways',ascending=False)
    dist.to_csv(TABLE_DIR/'table7_family_distribution.csv',index=False)
    # negative pools
    rec_train,_=core.build_samples(data,seed=3042); rec_test,_=core.build_samples(data,seed=4042)
    neg_train=[r for r in rec_train if r['label']==0]
    neg_test=[r for r in rec_test if r['label']==0]
    rng=np.random.default_rng(5042)
    # Use fixed filtered GO vocabulary, no supervised family-specific GO selection.
    selected=list(data.go_terms)
    rows=[]
    for idx,fam in enumerate(dist.loc[dist.n_pathways>=10,'family'],1):
        t0=time.time()
        train_pos=[r for r in pos if r['family']!=fam]
        test_pos=[r for r in pos if r['family']==fam]
        tr_neg=[neg_train[int(i)] for i in rng.choice(len(neg_train),size=min(len(neg_train),2*len(train_pos)),replace=False)]
        te_neg=[neg_test[int(i)] for i in rng.choice(len(neg_test),size=min(len(neg_test),2*len(test_pos)),replace=False)]
        train=train_pos+tr_neg; test=test_pos+te_neg
        Xtr,names,_=core.build_feature_matrix(train,selected,data,seed=42); Xte,_,_=core.build_feature_matrix(test,selected,data,seed=42)
        ytr=np.array([r['label'] for r in train]); yte=np.array([r['label'] for r in test])
        model=xgb.XGBClassifier(n_estimators=60,max_depth=4,learning_rate=.07,subsample=.8,colsample_bytree=.7,scale_pos_weight=2,min_child_weight=3,reg_alpha=.1,reg_lambda=1.0,eval_metric='logloss',random_state=42+idx,n_jobs=4,verbosity=0,tree_method='hist')
        model.fit(Xtr,ytr)
        score=model.predict_proba(Xte)[:,1]
        d=metric(yte,score)
        row={'family':fam,'n_heldout_pathways':len(test_pos),'n_test_negatives':len(te_neg),'n_train_pathways':len(train_pos),'n_go_features':len(selected),'D':len(names),'median_size':float(dist.loc[dist.family==fam,'median_size'].iloc[0]),'median_jaccard':float(dist.loc[dist.family==fam,'median_jaccard'].iloc[0]),**d}
        rows.append(row); print(idx,fam,d['test_auroc'], 'time', round(time.time()-t0,1), flush=True)
    out=pd.DataFrame(rows).sort_values('test_auroc',ascending=False); out.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
    with open(TABLE_DIR/'lofo_generalization_summary.json','w') as f: json.dump(out.to_dict(orient='records'),f,indent=2)
    plot=out.sort_values('test_auroc')
    plt.figure(figsize=(8.5,5.2)); plt.barh(plot['family'],plot['test_auroc']); plt.xlim(.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out generalisation')
    for i,v in enumerate(plot['test_auroc']): plt.text(min(v+.006,.985),i,f'{v:.3f}',va='center',fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    top=dist.sort_values('n_pathways',ascending=True)
    plt.figure(figsize=(8.5,5.2)); plt.barh(top['family'],top['n_pathways']); plt.xlabel('Number of curated pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png',dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()
if __name__=='__main__': main()
