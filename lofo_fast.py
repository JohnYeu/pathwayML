#!/usr/bin/env python3
"""LOFO validation with per-family training-only GO selection.

For each held-out family, GO terms are selected on the training records using
a 70% cumulative MI cutoff, avoiding leakage from held-out pathway labels.
Uses separate negative pools (seeds 3042/4042) for train and test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import run_no_embedding_reproducible as core
from generalization_and_negative_analysis import assign_family

TABLE_DIR=Path('tables'); FIG_DIR=Path('figures')
TABLE_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)


def family_assignments(data):
    rows=[]
    for pid, genes in data.pathways.items():
        name=data.pathway_names.get(pid,pid)
        rows.append({'pathway_id':pid,'pathway_name':name,'family':assign_family(pid,name),'source':'AraCyc' if pid.startswith('AC_') else 'KEGG','n_genes':len(genes),'jaccard_mean':core.pathway_jaccard_mean(genes,data.gene_go,salt=f'fam:{pid}',seed=42)})
    df=pd.DataFrame(rows)
    df.to_csv(TABLE_DIR/'pathway_family_assignment.csv',index=False)
    dist=df.groupby('family').agg(n_pathways=('pathway_id','count'), n_kegg=('source',lambda s:int((s=='KEGG').sum())), n_aracyc=('source',lambda s:int((s=='AraCyc').sum())), median_size=('n_genes','median'), median_jaccard=('jaccard_mean','median')).reset_index().sort_values('n_pathways',ascending=False)
    dist.to_csv(TABLE_DIR/'table7_family_distribution.csv',index=False)
    return df, dist

def select_terms(train_records,data,seed,frac=.70):
    """Select GO terms via variance filter + MI ranking with cumulative MI fraction cutoff."""
    y=np.array([r['label'] for r in train_records])
    F=np.vstack([core.go_frequency(r['genes'], data.go_terms, data.gene_go) for r in train_records])
    vt=VarianceThreshold(threshold=.001)
    F2=vt.fit_transform(F)
    terms=[t for t,k in zip(data.go_terms, vt.get_support()) if k]
    mi=mutual_info_classif(F2,y,random_state=seed,n_neighbors=5)
    order=np.argsort(mi)[::-1]
    s=mi[order]; total=float(s.sum())
    k=int(np.argmax(np.cumsum(s)/total >= frac)+1) if total>0 else min(20,len(terms))
    k=max(3,min(k,len(terms)))
    return [terms[int(i)] for i in order[:k]]

def metric(y,score):
    """Compute classification metrics from true labels and predicted scores."""
    pred=(score>=.5).astype(int)
    return dict(test_auroc=float(roc_auc_score(y,score)), test_auprc=float(average_precision_score(y,score)), f1=float(f1_score(y,pred)), precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)), positive_score_median=float(np.median(score[y==1])), negative_score_median=float(np.median(score[y==0])))

def main():
    data=core.load_data(); fam_df, dist=family_assignments(data)
    # Separate negative pools for train (seed 3042) and test (seed 4042) to prevent overlap
    records_train,_=core.build_samples(data,seed=3042)
    records_test,_=core.build_samples(data,seed=4042)
    neg_train=[r for r in records_train if r['label']==0]
    neg_test=[r for r in records_test if r['label']==0]
    rng=np.random.default_rng(5042)
    fam_map=dict(zip(fam_df.pathway_id,fam_df.family))
    pos_all=[{'id':pid,'label':1,'type':'curated_pathway','name':data.pathway_names.get(pid,pid),'genes':sorted(genes)} for pid,genes in data.pathways.items()]
    rows=[]
    families=dist.loc[dist.n_pathways>=10,'family'].tolist()
    for idx,fam in enumerate(families,1):
        train_pos=[r for r in pos_all if fam_map[r['id']]!=fam]
        test_pos=[r for r in pos_all if fam_map[r['id']]==fam]
        train_neg=[neg_train[int(i)] for i in rng.choice(len(neg_train),size=min(len(neg_train),2*len(train_pos)),replace=False)]
        test_neg=[neg_test[int(i)] for i in rng.choice(len(neg_test),size=min(len(neg_test),2*len(test_pos)),replace=False)]
        train=train_pos+train_neg; test=test_pos+test_neg
        selected=select_terms(train,data,seed=42+idx)
        Xtr, names, groups=core.build_feature_matrix(train, selected, data, seed=42)
        Xte, _, _=core.build_feature_matrix(test, selected, data, seed=42)
        ytr=np.array([r['label'] for r in train]); yte=np.array([r['label'] for r in test])
        model=xgb.XGBClassifier(n_estimators=120,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.7,scale_pos_weight=2,min_child_weight=3,reg_alpha=.1,reg_lambda=1.0,eval_metric='logloss',random_state=42+idx,n_jobs=4,verbosity=0,tree_method='hist')
        model.fit(Xtr,ytr)
        score=model.predict_proba(Xte)[:,1]
        d=metric(yte,score)
        row={'family':fam,'n_heldout_pathways':len(test_pos),'n_test_negatives':len(test_neg),'n_train_pathways':len(train_pos),'n_go_selected':len(selected),'D':len(names),'median_size':float(dist.loc[dist.family==fam,'median_size'].iloc[0]),'median_jaccard':float(dist.loc[dist.family==fam,'median_jaccard'].iloc[0]), **d}
        rows.append(row)
        print(idx,fam,row['test_auroc'],flush=True)
    out=pd.DataFrame(rows).sort_values('test_auroc',ascending=False)
    out.to_csv(TABLE_DIR/'table8_lofo_generalization.csv',index=False)
    with open(TABLE_DIR/'lofo_generalization_summary.json','w') as f: json.dump(out.to_dict(orient='records'),f,indent=2)
    # figures
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot=out.sort_values('test_auroc')
    plt.figure(figsize=(8.5,5.2))
    plt.barh(plot['family'],plot['test_auroc'])
    plt.xlim(0.5,1.0); plt.xlabel('LOFO AUROC'); plt.title('Leave-one-family-out generalisation')
    for i,v in enumerate(plot['test_auroc']): plt.text(min(v+.006,.985),i,f'{v:.3f}',va='center',fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/'fig12_lofo_generalization.png',dpi=300); plt.savefig(FIG_DIR/'fig12_lofo_generalization.pdf'); plt.close()
    top=dist.sort_values('n_pathways',ascending=True)
    plt.figure(figsize=(8.5,5.2)); plt.barh(top['family'],top['n_pathways']); plt.xlabel('Number of curated pathways'); plt.title('Coarse pathway-family distribution'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig13_family_distribution.png',dpi=300); plt.savefig(FIG_DIR/'fig13_family_distribution.pdf'); plt.close()

if __name__=='__main__': main()
