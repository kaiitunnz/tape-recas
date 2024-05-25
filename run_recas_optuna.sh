datasets=("arxiv_2023" "ogbn-arxiv" "cora" "pubmed")
results="results"
seed=None
runs=4
emb="node2vec"
n_trials=500

mkdir -p $results

for dataset in "${datasets[@]}"
do
    dataset_results="${results}/${dataset}"
    rm -r $dataset_results
    mkdir -p $dataset_results
    python -m core.runReCaS_optuna dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials >> "${dataset_results}/${dataset}_mlp.txt" 2>> "${dataset_results}/${dataset}_mlp.err" &
    python -m core.runReCaS_optuna dataset $dataset gnn.model.name GCN seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials >> "${dataset_results}/${dataset}_gcn.txt" 2>> "${dataset_results}/${dataset}_gcn.err" &
    python -m core.runReCaS_optuna dataset $dataset gnn.model.name SAGE seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials >> "${dataset_results}/${dataset}_sage.txt" 2>> "${dataset_results}/${dataset}_sage.err" &
    python -m core.runReCaS_optuna dataset $dataset gnn.model.name RevGAT seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials >> "${dataset_results}/${dataset}_revgat.txt" 2>> "${dataset_results}/${dataset}_revgat.err" &
    python -m core.runReCaS_optuna dataset $dataset seed $seed runs $runs cas.optuna.n_jobs 4 cas.use_lm_pred True cas.optuna.n_trials $n_trials >> "${dataset_results}/${dataset}_none.txt" 2>> "${dataset_results}/${dataset}_none.err" &
    python -m core.runReCaS_optuna dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 gnn.train.use_emb $emb cas.optuna.n_trials $n_trials >> "${dataset_results}/${dataset}_mlp_$emb.txt" 2>> "${dataset_results}/${dataset}_mlp_$emb.err" &
    wait
    echo "DONE: $dataset"
done
