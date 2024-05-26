datasets=("arxiv_2023" "ogbn-arxiv" "cora" "pubmed")
seed=None
runs=4
emb="node2vec"
device=1

n_trials=200
results="results_recas_sc"
mkdir -p $results

for dataset in "${datasets[@]}"
do
    dataset_results="${results}/${dataset}"
    rm -r $dataset_results
    mkdir -p $dataset_results
    python -m core.runReCaS_optuna_sc dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_mlp.txt" 2>> "${dataset_results}/${dataset}_mlp.err" &
    python -m core.runReCaS_optuna_sc dataset $dataset gnn.model.name GCN seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_gcn.txt" 2>> "${dataset_results}/${dataset}_gcn.err" &
    python -m core.runReCaS_optuna_sc dataset $dataset gnn.model.name SAGE seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_sage.txt" 2>> "${dataset_results}/${dataset}_sage.err" &
    python -m core.runReCaS_optuna_sc dataset $dataset gnn.model.name RevGAT seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_revgat.txt" 2>> "${dataset_results}/${dataset}_revgat.err" &
    python -m core.runReCaS_optuna_sc dataset $dataset seed $seed runs $runs cas.optuna.n_jobs 4 cas.use_lm_pred True cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_none.txt" 2>> "${dataset_results}/${dataset}_none.err" &
    python -m core.runReCaS_optuna_sc dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 gnn.train.use_emb $emb cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_mlp_$emb.txt" 2>> "${dataset_results}/${dataset}_mlp_$emb.err" &
    wait
    echo "DONE: $dataset"
done

n_trials=300
results="results_recas_scs"
mkdir -p $results

for dataset in "${datasets[@]}"
do
    dataset_results="${results}/${dataset}"
    rm -r $dataset_results
    mkdir -p $dataset_results
    python -m core.runReCaS_optuna_scs dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_mlp.txt" 2>> "${dataset_results}/${dataset}_mlp.err" &
    python -m core.runReCaS_optuna_scs dataset $dataset gnn.model.name GCN seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_gcn.txt" 2>> "${dataset_results}/${dataset}_gcn.err" &
    python -m core.runReCaS_optuna_scs dataset $dataset gnn.model.name SAGE seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_sage.txt" 2>> "${dataset_results}/${dataset}_sage.err" &
    python -m core.runReCaS_optuna_scs dataset $dataset gnn.model.name RevGAT seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_revgat.txt" 2>> "${dataset_results}/${dataset}_revgat.err" &
    python -m core.runReCaS_optuna_scs dataset $dataset seed $seed runs $runs cas.optuna.n_jobs 4 cas.use_lm_pred True cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_none.txt" 2>> "${dataset_results}/${dataset}_none.err" &
    python -m core.runReCaS_optuna_scs dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 gnn.train.use_emb $emb cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_mlp_$emb.txt" 2>> "${dataset_results}/${dataset}_mlp_$emb.err" &
    wait
    echo "DONE: $dataset"
done

n_trials=400
results="results_recas_cscs"
mkdir -p $results

for dataset in "${datasets[@]}"
do
    dataset_results="${results}/${dataset}"
    rm -r $dataset_results
    mkdir -p $dataset_results
    python -m core.runReCaS_optuna_cscs dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_mlp.txt" 2>> "${dataset_results}/${dataset}_mlp.err" &
    python -m core.runReCaS_optuna_cscs dataset $dataset gnn.model.name GCN seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_gcn.txt" 2>> "${dataset_results}/${dataset}_gcn.err" &
    python -m core.runReCaS_optuna_cscs dataset $dataset gnn.model.name SAGE seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_sage.txt" 2>> "${dataset_results}/${dataset}_sage.err" &
    python -m core.runReCaS_optuna_cscs dataset $dataset gnn.model.name RevGAT seed $seed runs $runs cas.optuna.n_jobs 4 cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_revgat.txt" 2>> "${dataset_results}/${dataset}_revgat.err" &
    python -m core.runReCaS_optuna_cscs dataset $dataset seed $seed runs $runs cas.optuna.n_jobs 4 cas.use_lm_pred True cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_none.txt" 2>> "${dataset_results}/${dataset}_none.err" &
    python -m core.runReCaS_optuna_cscs dataset $dataset gnn.model.name MLP seed $seed runs $runs cas.optuna.n_jobs 4 gnn.train.use_emb $emb cas.optuna.n_trials $n_trials device $device >> "${dataset_results}/${dataset}_mlp_$emb.txt" 2>> "${dataset_results}/${dataset}_mlp_$emb.err" &
    wait
    echo "DONE: $dataset"
done