import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

import plotly.express as px
import plotly.graph_objects as go

# load data
data = pd.read_table(snakemake.params.data, sep='\t', index_col=0).reset_index(names='sample')
meta = pd.read_table(snakemake.params.meta, sep='\t', index_col=0).reset_index(names='sample')
data_long = data.melt(id_vars='sample', value_name='abundance', var_name='feature')
df = pd.merge(data_long, meta, on='sample', how='inner')

df_sig = pd.read_csv(f"{snakemake.params.output_dir}/df_differential_test.tsv", sep='\t')

## prepare data for ML, only keep significant proteins
df_ml = pd.merge(df, df_sig[['feature','pval_adj']], on='feature', how='left')
df_ml = df_ml[df_ml['pval_adj'] < 0.5]

X = df_ml.pivot(index=['sample',snakemake.params.group_column], columns='feature', values='abundance')
y = pd.Series(X.index.get_level_values(snakemake.params.group_column)).astype('category').cat.codes

## read in the best model name 
with open(f"{snakemake.params.output_dir}/.best_model.txt") as f:
    model_name = f.read().strip()

# use the best model 
models = {
	"Logistic Regression": LogisticRegression(max_iter=1000),
	"Random Forest": RandomForestClassifier(n_estimators=1000, random_state=42),
	"Support Vector Machine": SVC(),  # optionally use probability=True if needed
	"K-Nearest Neighbors": KNeighborsClassifier(),
	"Gradient Boosting": GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, random_state=42),
	"XGBoost": xgb.XGBClassifier(n_estimators=1000, eval_metric='logloss', random_state=42),
	"Naive Bayes": GaussianNB(),
	"Decision Tree": DecisionTreeClassifier(random_state=42)
}

model = models[model_name]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

mean_fpr = np.linspace(0, 1, 10)  # common FPR scale
ot_roc = pd.DataFrame()
ot_confusion = pd.DataFrame()
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)
    # Compute ROC curve and AUC
    from sklearn.metrics import roc_curve, auc
    y_prob = model.predict_proba(X_test)[:, 1]  # probability estimates
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    mean_fpr = np.insert(mean_fpr, 0, 0)
    interp_tpr = np.insert(interp_tpr, 0, 0)
    #interp_tpr[0] = 0.0  # ensure TPR starts at 0
    tmp = pd.DataFrame({
        'fpr': mean_fpr,
        'tpr': interp_tpr,
        'fold': i+1,
        'auc': roc_auc
    })
    ot_roc = pd.concat([ot_roc, tmp])
    tmp = pd.DataFrame({
        'true': y_test.values,
        'pred': y_pred,
        'fold': i+1
    })
    ot_confusion = pd.concat([ot_confusion, tmp])


# calculate mean roc
ot_roc_mean = ot_roc[ot_roc['tpr'] != 0].groupby('fpr')['tpr'].mean().reset_index() #should skip points when tpr is also 0, 
start_point = pd.DataFrame({'fpr':0, 'tpr':0}, index=[0])
ot_roc_mean = pd.concat([start_point, ot_roc_mean]).reset_index(drop=True)

# mean auc 
mask = ot_roc[['fold','auc']].duplicated()
mean_auc = ot_roc.loc[~mask, ['auc']].mean().values[0] # mean AUC for all folds

# plot 
fig = go.Figure()
for fold in ot_roc['fold'].unique():
    fold_data = ot_roc[ot_roc['fold'] == fold]
    fig.add_trace(go.Scatter(
        x=fold_data['fpr'],
        y=fold_data['tpr'],
        mode='lines',
        line=dict(color='gray', shape='hv'),  # "hv" = horizontal then vertical
        opacity=0.3,
        legendgroup='Fold',
        showlegend=bool(fold==1),
        name='Cross-validations'
    ))

fig.add_trace(go.Scatter(
    x=ot_roc_mean['fpr'],
    y=ot_roc_mean['tpr'],
    mode='lines',
    line=dict(color='blue', width=3, shape='hv'),
    name='Mean ROC'
))
fig.add_annotation(
    text=f"Mean AUC is {mean_auc}",
    x=0.8,
    y=0.5,
    showarrow=False
)
fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    showlegend=True,
    template='plotly_white'
)
fig.update_xaxes(range=[-0.01, 1.01])
fig.update_yaxes(range=[0, 1.01])
fig.write_html(f"{snakemake.params.output_dir}/ROC.html")

# # plot prediction matrix 
# ot_confusion.head()
# fig = px.density_heatmap(ot_confusion,x='true',y='pred',nbinsx=2,nbinsy=2,
#                          color_continuous_scale='Blues', text_auto=True)
# fig.update_traces(
#     textfont_size=18
# )
# fig.update_layout(
#     xaxis_title='True label',
#     yaxis_title='Predicted label',
#     coloraxis_showscale=False,
#     template='plotly_white'
# )
# fig.show()

## rule finishes
with open(snakemake.output[0], 'w') as f:
    f.write('')
