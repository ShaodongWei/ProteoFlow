from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np 

# data = pd.read_table('./data_input/cleaned_data.tsv', sep='\t', index_col=0).reset_index(names='sample')
# meta = pd.read_table('./data_input/metadata.tsv', sep='\t', index_col=0).reset_index(names='sample')
# df_sig = pd.read_table('./output/df_differential_test.tsv', sep='\t', index_col=0).reset_index(names='sample')

## load data
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

# make label to number matching table, used for confustion matrix
y_label = pd.DataFrame({
    'lable': X.index.get_level_values(snakemake.params.group_column), 
    'value': y
    })
y_label = y_label[~y_label.duplicated()]

# Define models
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

## Train and evaluate models
ot = []
for name, model in models.items():
    # name='Random Forest'
    # model = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    score = cross_val_score(model, cv=cv, X=X, y=y, scoring='accuracy', n_jobs=snakemake.params.threads)
    ot.append({
        'model': name, 
        'acc_mean': np.round(np.mean(score),2),
        'acc_median': np.round(np.median(score), 2)
    })

ot = pd.DataFrame(ot)
print(f"Accuracy for all models \n {ot}")
# choose the best model 
ot['score'] = ot[['acc_mean','acc_median']].mean(axis=1)
ot = ot.sort_values('score',ascending=False)
model_name = ot['model'].values[0]
print(f"The best model is {model_name}")
with open(f"{snakemake.params.output_dir}/.best_model.txt", 'w') as f:
    f.write(model_name)
model = models[model_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)
acc = np.round(accuracy_score(y_pred=y_pred, y_true=y_test),2)
print(f"The best model acccuracy is {acc} in a random train test splition")

# rule finsihes
with open(snakemake.output[0], 'w') as f:
    f.write('')

