import pandas as pd, numpy as np
from scipy import stats
from common import nadeau_bengio, RESULTS, paper3_all_path
i=pd.read_csv(RESULTS / 'islam_fair_all.csv'); a=pd.read_csv(paper3_all_path())
a=a[a.model.str.startswith('OURS')]; allr=pd.concat([i,a]); rows=[]
for kind in ['cv','holdout']:
    s=allr[allr.kind==kind]; print('==',kind)
    print(s.groupby('model')[['accuracy','macro_f1','f1_Enrolled','recall_Enrolled']].agg(['mean','std']).round(4))
    for me in ['OURS v1 (balanced weights)','OURS v2 (sqrt weights)']:
        for ref in ['Islam XGB (fair)','Islam RF (fair)','Islam GB (fair)','Islam DT (fair)']:
            for met in ['accuracy','macro_f1','f1_Enrolled','recall_Enrolled']:
                x=s[s.model==me].set_index('split')[met]; z=s[s.model==ref].set_index('split')[met]; d=(x-z)
                p=nadeau_bengio(d.values,s.n_train.iloc[0],s.n_test.iloc[0])[1] if kind=='cv' else stats.ttest_1samp(d,0)[1]
                rows.append(dict(kind=kind,ours=me,baseline=ref,metric=met,diff=d.mean(),p=p,wins=int((d>0).sum()),n=len(d)))
r=pd.DataFrame(rows); r.to_csv(RESULTS / 'islam_fair_tests.csv',index=False)
print(r[r.baseline.isin(['Islam XGB (fair)','Islam RF (fair)'])].round(4).to_string())
