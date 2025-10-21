import pandas as pd 
import os
import numpy as np 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from joblib import Parallel, delayed
import statsmodels.api as sm
from tqdm import tqdm
import re
import plotly.express as px


## make directory 
os.makedirs(snakemake.params.output_dir, exist_ok=True)

## load cleaned pg table 
data = pd.read_table(snakemake.params.data, sep='\t', index_col=0).reset_index(names='sample')
meta = pd.read_table(snakemake.params.meta, sep='\t', index_col=0).reset_index(names='sample')

## reshape and merge
data_long = data.melt(id_vars='sample', value_name='abundance', var_name='feature')
df = pd.merge(data_long, meta, on='sample', how='inner')

def test_feature_series(value_series, group_series):
    if len(group_series.unique()) != 2:
        raise ValueError('only two groups are supported')    
    model = ols(f'value ~ C(type)', data=pd.DataFrame({'value':value_series, 'type':group_series})).fit()
    anova_table = sm.stats.anova_lm(model)
    pval = anova_table['PR(>F)'].iloc[0].round(4)
    fold = np.log2(value_series[group_series == group_series.unique()[0]].mean()/value_series[group_series == group_series.unique()[1]].mean())
    fold = np.round(fold,3)
    level = group_series.unique()[0] + '/' + group_series.unique()[1]
    return {'log2foldratio': fold, 'pval': pval, 'level': level}

def test_feature_dataframe(dataframe, feature_col, value_col, group_col, n_jobs=1):
    if dataframe[group_col].nunique() != 2:
        raise ValueError('only two groups are supported')
    features = dataframe[feature_col].unique()
    ot =  Parallel(n_jobs) (
        delayed(test_feature_series)(value, group) for value, group in 
                [(dataframe[dataframe[feature_col]==feature][value_col], dataframe[dataframe[feature_col]==feature][group_col]) 
                 for feature in tqdm(features)]
    )
    ot = pd.DataFrame(ot)
    ot['pval_adj'] = sm.stats.multipletests(ot['pval'],method='fdr_bh')[1].round(4)
    ot['feature'] = features
    return ot

df_sig = test_feature_dataframe(df, feature_col='feature', value_col='abundance', group_col=snakemake.params.group_column, n_jobs=snakemake.params.threads)
df_sig.to_csv(f"{snakemake.params.output_dir}/df_differential_test.tsv", sep='\t', index=False)

## plot
df_sig['Significance'] = df_sig['pval_adj'].apply(lambda x: 'Significant' if x < 0.05 else 'Non-significant')

raw_offset = df_sig.loc[df_sig['pval_adj'] > 0, 'pval_adj'].min() / 10
offset = 10 ** np.floor(np.log10(raw_offset))
print(offset)
level = re.sub('>', '/', df_sig['level'].values[0])
df_sig['y'] = df_sig['pval_adj'].apply(lambda x: -np.log10(offset) if x == 0 else -np.log10(x))
fig = px.scatter(df_sig, x='log2foldratio', y='y', color='Significance')
fig.add_vline(
    x=0,
    line_color='gray',
    line_dash='dash'
)
fig.update_layout(
    yaxis_title="-log10(adjusted p-value)",
    xaxis_title=f"log2(fold change {level})",
)
fig.update_yaxes(
    range=[0,-np.log10(offset)+0.5], 
    tickvals=[i for i in range(0,int(-np.log10(offset)+1))],
    ticktext=[str(i) for i in range(0,int(-np.log10(offset)+1))],
    )

fig.write_html(f"{snakemake.params.output_dir}/volcano.html")

## rule finishes
with open(snakemake.output[0],'w') as f:
    f.write('')
