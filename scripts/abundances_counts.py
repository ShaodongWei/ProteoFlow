import pandas as pd 
import os

## make directory 
os.makedirs(snakemake.params.output_dir, exist_ok=True)

## load cleaned data
data = pd.read_table(snakemake.input.data, sep='\t', index_col=0)
# data = pd.read_csv('./data_input/cleaned_data.tsv', sep='\t', index_col=0)
data_long = pd.melt(data.reset_index(names='sample'), id_vars='sample', var_name='feature', value_name='abundance')

## calculate median abundance for each feature
df_abundance = data.median().reset_index()
df_abundance.columns = ['feature','abundance']
df_abundance = df_abundance.sort_values('abundance', ascending=False).reset_index(drop=True)
df_abundance['ranks'] = df_abundance.index + 1

## calculate sample feature counts 
df_raw = pd.read_csv('./data_input/raw_data.tsv', sep='\t', index_col=0)
df_raw = df_raw.count(axis=1).reset_index(name='count').sort_values('count', ascending=False).reset_index(drop=True)
df_raw['ranks'] = df_raw.index + 1
df_raw.head()

## scatter plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go 

fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=['Feature abundance', 'Number of detected features per sample'])

fig.add_trace(
    go.Scatter(
    x=df_abundance['ranks'],
    y=df_abundance['abundance'],
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
    x=df_raw['ranks'],
    y=df_raw['count'],
    marker=dict(color='gray')
    ),
    row=2,col=1
)
fig.update_layout(
    showlegend=False
)
fig.update_xaxes(title_text='Sample ranks', row=2, col=1)
fig.update_xaxes(title_text='Feature ranks', row=1, col=1)
fig.write_html(f"{snakemake.params.output_dir}/abundances_counts.html")

# rule finishes 
with open(snakemake.output[0],'w') as f:
    f.write('')
