# Download original text attributes
cd dataset

# 1. ogbn-arxiv
wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
tar -zxvf titleabs.tsv.gz
rm titleabs.tsv.gz

# 2. arxiv_2023
gdown 1-s1Hf_2koa1DYp_TQvYetAaivK9YDerv
unzip arxiv_2023_orig.zip
rm arxiv_2023_orig.zip

# 3. cora
gdown 1hxE0OPR7VLEHesr48WisynuoNMhXJbpl
unzip cora_orig.zip
rm cora_orig.zip

# 4. pubmed
gdown 1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W
unzip PubMed_orig.zip
rm PubMed_orig.zip

cd -

# Download LLM responses
mkdir -p gpt_responses
cd gpt_responses

# 1. ogbn-arxiv
gdown 1A6mZSFzDIhJU795497R6mAAM2Y9qutI5
# 2. ogbn-products
gdown 1C769tlhd8pT0s7I3vXIEUI-PK7A4BB1p
# 3. arxiv_2023
wget -O arxiv_2023.zip https://www.dropbox.com/scl/fi/cpy9m3mu6jasxr18scsoc/arxiv_2023.zip?rlkey=4wwgw1pgtrl8fo308v7zpyk59&dl=0
unzip arxiv_2023.zip
rm arxiv_2023.zip
# 4. cora
gdown 1tSepgcztiNNth4kkSR-jyGkNnN7QDYax
# 5. PubMed
gdown 166waPAjUwu7EWEvMJ0heflfp0-4EvrZS

cd -

# Download LM predictions and features
function download_pred_emb() {
    data_dir=prt_lm/$1/microsoft
    mkdir -p $data_dir
    cd $data_dir
    gdown $2
    gdown $3
    cd -
}

mkdir prt_lm
# 1. ogbn-arxiv
download_pred_emb ogbn-arxiv 1APw6hnOtdMFa5rXhhC9520ZmoXhytDDr 137uBunrp0jsiG_AFvlcCME4kbldGDkLW
download_pred_emb ogbn-arxiv2 1WUsYI3MM-K9mtB9mPnqJlh9DalHB8a1l 1jnKdDYT_F9Fx9w29h1uD6aeDm_JK0kDi

# 2. cora
download_pred_emb cora 1lVF0aJFUCUBCTKOq8Jm5ju8fKhKW-coB 1LDXJHeJrqQa2e8u6ivEgJVJUpKo1DTmL
download_pred_emb cora2 1potvRUweicMGy_F8I9QRS4ySctp4opfh 1b4EILJ9RR6nBtzpDH-I2OYl4TTukH9u6

# 3. pubmed
download_pred_emb pubmed 1u8yDPzCrOoFdr5jxp_5Dyeu97NtAKrcN 1-ov-yRwodMmyGPWpA7wXZ02pT4oF5JFp
download_pred_emb pubmed2 1VoQhGXKILNq2CHuq1Jzlry7rehL6Oxx_ 1CvbVs3QDrS1ulwJ2-d8hKwVHNkwFRnEM