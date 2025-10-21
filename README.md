# When to use
This script is to do proteomics downstream analysis, including statistics, volcano plot, machine learning modelling, ROC plot. The input abundance table has to be preprocessed first, please check [ProteoPrep](https://github.com/ShaodongWei/ProteoPrep) how to preprocessing proteomics data. 

# Getting Started
## Prerequisites
### Before using the pipeline, ensure you have the following installed:

Snakemake: [Installation guide](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) 

conda: [installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 

# How to Use the Pipeline
## 1. Clone the Repository
First, clone the repository containing the Snakemake pipeline to your local machine:

```bash
git clone https://github.com/ShaodongWei/ProteoFlow.git

```
## 2. Set up the configuration file 
```
cleaned_data: "data_input/cleaned_data.tsv" # the input abundance table (first column is sample id, other columns are features, rows are samples), can be protein or peptide level 
raw_data: "data_input/raw_data.tsv" # the abundance table that is not preprocessed. 
metadata: "data_input/metadata.tsv" # the input metadata table (first column is sample id). 
group_column: "sample_type" # the column names in metadata to group samples. 
output_directory: "output" # the output directory name 
threads: 10 # number of threads 
```

## 3. Run the pipeline using snakemake
### Run the entire pipeline 
```
snakemake --cores threads_number --use-conda --dryrun # A dry run to show what steps will be run. 
snakemake --cores threads_number --use-conda # Conda will install dependencies automatically. All steps will be executed sequentially. 
```
### Run a specific step 
```
snakemake --list # show all steps

snakemake step_name --cores threads_number --use-conda # Run a specific step 

```
## 4. Steps in the workflow 
### Step 1, Label-free quantification (LFQ) intensity plot and the number of detected proteins/peptides per sample 
This step is to show the global intensity distribution and the protein/peptide number per sample. 
```
snakemake counts_abundances --cores threads_number --use-conda
```
### Step 2, differential analysis
This step is to do differential test to identify which proteins/peptides show statistically significant differences in abundance between two biological conditions (so far only 2 levels is supported).
```
snakemake differential_test --cores threads_number --use-conda
```
### Step 3, machine learning
This step is to do thorough machine learning by iterating different machine learning models and perform cross-validations and eventually returns the best selected method. 
```
snakemake  machine_learning --cores threads_number --use-conda
```
### Step 4, plot ROC
This step is to plot ROC based on the selected machine learning method. 
```
snakemake ROC --cores threads_number --use-conda
```

## 5. Troubleshooting

### 5.1 Directory locked
This normally happens when you run snakemake multiple times, you can unlock it by deleting the lock files 
```
rm .snakemake/locks
```
