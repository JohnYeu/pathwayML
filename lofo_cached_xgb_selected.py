#!/usr/bin/env python3
"""LOFO validation using the canonical seed-42 MI-selected GO terms.

Unlike lofo_cached_seed42.py (all filtered GO terms), this uses the subset
selected by the main pipeline's MI procedure. This tests whether the selected
features generalize across families.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import assign_family
TABLE_DIR=Path('tables'); FIG_DIR=Path('figures')

def metric(y,s):
    """Compute classification metrics from true labels and predicted scores."""
    p=(s>=.5).astype(int)
    return dict(test_auroc=float(roc_auc_score(y,s)), test_auprc=float(average_precision_score(y,s)), f1=float(f1_score(y,p)), precision=float(precision_score(y,p,zero_division=0)), recall=float(recall_score(y,p,zero_division=0)), positive_score_median=float(np.median(s[y==1])), negative_score_median=float(np.median(s[y==0])))

def main():
    data=core.load_data(); records=json.load(open(TABLE_DIR/'reproducibility'/'samples.json'))
    # Use the MI-selected GO terms from the canonical seed-42 run
    selected=json.load(open(TABLE_DIR/'reproducibility'/'selected_go_terms.json'))
    pos=[]; neg=[]; fam_rows=[]
    for rec in records:
        if rec['label']==1:
            pid=rec['id']; name=data.pathway_names.get(pid,rec.get('name',pid)); fam=assign_family(pid,name); rec['family']=fam; pos.append(rec)
            fam_rows.append({'pathway_id':pid,'pathway_name':name,'family':fam,'source':'AraCyc' if pid.startswith('AC_') else 'KEGG','n_genes':len(rec['genes']),'jaccard_mean':core.pathway_jaccard_mean(rec['genes'],data.gene_go,salt=f'fam:{pid}',seed=42)})
        else: neg.append(rec)
    fam_df=pd.DataFrame(fam_rows)
    dist=fam_df.groupby('family').agg(n_pathways=('pathway_id','count'),n_kegg=('source',lambda s:int((s=='KEGG').sum())),n_aracyc=('source',lambda s:int((s=='AraCyc').sum())),median_size=('n_genes','median'),median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways',ascending=False)
    all_records=pos+neg
    X,names,_=core.build_feature_matrix(all_records,selected,data,seed=42)
    y=np.array([r['label'] for r in all_records]); fams=[r.get('family','NEG') for r in all_records]
    pos_idx=np.array([i for i,r in enumerate(all_records) if r['label']==1]); neg_idx=np.array([i for i,r in enumerate(all_records) if r['label']==0])
    rng=np.random.default_rng(42); out=[]
    for j,fam in enumerate(dist.loc[dist.n_pathways>=10,'family'],1):
        test_pos_idx=np.array([i for i in pos_idx if fams[i]==fam]); train_pos_idx=np.array([i for i in pos_idx if fams[i]!=fam])
        train_neg_idx=rng.choice(neg_idx,size=min(len(neg_idx),2*len(train_pos_idx)),replace=False)
        rem=np.setdiff1d(neg_idx,train_neg_idx,assume_unique=False)
        if len(rem)<2*len(test_pos_idx): rem=neg_idx
        test_neg_idx=rng.choice(rem,size=min(len(rem),2*len(test_pos_idx)),replace=False)
        tr=np.concatenate([train_pos_idx,train_neg_idx]); te=np.concatenate([test_pos_idx,test_neg_idx])
        model=xgb.XGBClassifier(n_estimators=80,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.7,scale_pos_weight=2,min_child_weight=3,reg_alpha=.1,reg_lambda=1.0,eval_metric='logloss',random_state=42+j,n_jobs=4,verbosity=0,tree_method='hist')
        model.fit(X[tr],y[tr]); score=model.predict_proba(X[te])[:,1]
        d=metric(y[te],score)
        row={'family':fam,'n_heldout_pathways':len(test_pos_idx),'n_test_negatives':len(test_neg_idx),'n_train_pathways':len(train_pos_idx),'probe_model':'XGBoost, fixed seed-42 selected features','n_go_features':len(selected),'D':len(names),'median_size':float(dist.loc[dist.family==fam,'median_size'].iloc[0]),'median_jaccard':float(dist.loc[dist.family==fam,'median_jaccard'].iloc[0]),**d}
        out.append(row); print(j,fam,d['test_auroc'],flush=True)
    out=pd.DataFrame(out).sort_values('test_auroc',ascending=False); out.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
    with open(TABLE_DIR/'lofo_generalization_summary.json','w') as f: json.dump(out.to_dict(orient='records'),f,indent=2)
    plot=out.sort_values('test_auroc')
    plt.figure(figsize=(8.5,5.2)); plt.barh(plot['family'],plot['test_auroc']); plt.xlim(.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out generalisation')
    for i,v in enumerate(plot['test_auroc']): plt.text(min(v+.006,.985),i,f'{v:.3f}',va='center',fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    top=dist.sort_values('n_pathways',ascending=True)
    plt.figure(figsize=(8.5,5.2)); plt.barh(top['family'],top['n_pathways']); plt.xlabel('Number of curated pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png',dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()
if __name__=='__main__': main()
