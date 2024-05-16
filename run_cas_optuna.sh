datasets=("ogbn-arxiv" "cora" "pubmed")
results="results"

mkdir -p $results

for dataset in "${datasets[@]}"
do
    dataset_results="${results}/${dataset}"
    rm -r $dataset_results
    mkdir -p $dataset_results
    python -m core.runCaS_optuna dataset $dataset gnn.model.name MLP seed 0 cas.optuna.n_jobs 4 >> "${dataset_results}/${dataset}_mlp.txt" 2>> "${dataset_results}/${dataset}_mlp.err" &
    python -m core.runCaS_optuna dataset $dataset gnn.model.name GCN seed 0 cas.optuna.n_jobs 4 >> "${dataset_results}/${dataset}_gcn.txt" 2>> "${dataset_results}/${dataset}_gcn.err" &
    python -m core.runCaS_optuna dataset $dataset gnn.model.name SAGE seed 0 cas.optuna.n_jobs 4 >> "${dataset_results}/${dataset}_sage.txt" 2>> "${dataset_results}/${dataset}_sage.err" &
    python -m core.runCaS_optuna dataset $dataset gnn.model.name RevGAT seed 0 cas.optuna.n_jobs 4 >> "${dataset_results}/${dataset}_revgat.txt" 2>> "${dataset_results}/${dataset}_revgat.err" &
    wait
    echo "DONE: $dataset"
done
