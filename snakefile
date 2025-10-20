configfile: "config.yaml" # path to the config.yaml or in the same directory as snakefile 

rule all:
    input:
        config["output_directory"] + "/.LFQ.done",
        config["output_directory"] + "/.differential_test.done",
        config["output_directory"] + "/.machine_learning.done",
        config["output_directory"] + "/.ROC_plot.done"
        

rule LFQ:
    input:
        data = config['lfq_table'],
        meta = config['metadata']
    output:
        config["output_directory"] + "/.LFQ.done"
    params:
        output_dir = config["output_directory"]
    conda:
        "./env/packages.yaml"
    script:
        './LFQ.py'

        
rule differential_test:
    input:
        config["output_directory"] + "/.LFQ.done"
    output:
        config["output_directory"] + "/.differential_test.done"
    params:
        data = config['lfq_table'],
        meta = config['metadata'],
        output_dir = config["output_directory"],
        group_column = config['group_column'],
        threads = config['threads']
    conda:
        "./env/packages.yaml"
    script:
        './differential_test.py'

rule machine_learning:
    input:
        config["output_directory"] + "/.differential_test.done"
    output:
        config["output_directory"] + "/.machine_learning.done"
    params:
        data = config['lfq_table'],
        meta = config['metadata'],
        # data_sig = config['data_differential_test']
        output_dir = config["output_directory"],
        group_column = config['group_column'],
        threads = config['threads']
    conda:
        "./env/packages.yaml"
    script:
        './machine_learning.py'

rule ROC:
    input:
        config["output_directory"] + "/.machine_learning.done"
    output:
        config["output_directory"] + "/.ROC_plot.done"
    params:
        data = config['lfq_table'],
        meta = config['metadata'],
        output_dir = config["output_directory"],
        group_column = config['group_column'],
        threads = config['threads']
    conda:
        "./env/packages.yaml"
    script:
        './ROC.py'

