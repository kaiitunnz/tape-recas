datasets=("arxiv_2023")
results="results"

rm -r $results
mkdir $results

for dataset in "${datasets[@]}"
do
    python -m core.runCaS_optuna dataset $dataset gnn.model.name MLP seed 0 cas.optuna.n_jobs 4 >> "${results}/${dataset}_mlp.txt" 2>> "${results}/${dataset}_mlp.err" &
    python -m core.runCaS_optuna dataset $dataset gnn.model.name GCN seed 0 cas.optuna.n_jobs 4 >> "${results}/${dataset}_gcn.txt" 2>> "${results}/${dataset}_gcn.err" &
    python -m core.runCaS_optuna dataset $dataset gnn.model.name SAGE seed 0 cas.optuna.n_jobs 4 >> "${results}/${dataset}_sage.txt" 2>> "${results}/${dataset}_sage.err" &
    python -m core.runCaS_optuna dataset $dataset gnn.model.name RevGAT seed 0 cas.optuna.n_jobs 4 >> "${results}/${dataset}_revgat.txt" 2>> "${results}/${dataset}_revgat.err" &
done

wait