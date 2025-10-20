import pandas as pd 
import os

## make directory 
os.makedirs(snakemake.params.output_dir, exist_ok=True)

## load cleaned pg table 
data = pd.read_table(snakemake.input.data, sep='\t', index_col=0)

## calculate median LFQ intensity for each protein
data = data.median().reset_index()
data.columns = ['protein','LFQ']
data = data.sort_values('LFQ', ascending=False)

# scatter plot
import plotly.express as px
fig = px.scatter(data, x='protein', y='LFQ')
fig.update_traces(
    marker=dict(size=3)
)
fig.update_layout(
    yaxis_title='Median LFQ intensity (log2)',
    xaxis_title='',
    xaxis=dict(showticklabels=False),
)
fig.write_html(f"{snakemake.params.output_dir}/LFQ.html")

# rule finishes 
with open(snakemake.output[0],'w') as f:
    f.write('')
