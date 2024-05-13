for dataset in "arxiv_2023"
do
    python -m core.runCaS_optuna dataset $dataset gnn.model.name MLP seed 0 >> ${dataset}_mlp.txt
    python -m core.runCaS_optuna dataset $dataset gnn.model.name GCN seed 0 >> ${dataset}_gcn.txt
    python -m core.runCaS_optuna dataset $dataset gnn.model.name SAGE seed 0 >> ${dataset}_sage.txt
    python -m core.runCaS_optuna dataset $dataset gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.5 seed 0 >> ${dataset}_revgat.txt
done