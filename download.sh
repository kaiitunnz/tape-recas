# Download original text attributes
cd dataset

# 1. ogbn-arxiv
wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
tar -zxvf titleabs.tsv.gz
rm titleabs.tsv.gz

# 2. arxiv_2023
gdown 1-s1Hf_2koa1DYp_TQvYetAaivK9YDerv
yes | unzip arxiv_2023_orig.zip
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

# 4. cora
gdown 1tSepgcztiNNth4kkSR-jyGkNnN7QDYax

# 5. PubMed
gdown 166waPAjUwu7EWEvMJ0heflfp0-4EvrZS

unzip ogbn-arxiv.zip
rm ogbn-arxiv.zip

unzip ogbn-products.zip
rm ogbn-products.zip

unzip arxiv_2023.zip
rm arxiv_2023.zip

unzip Cora.zip
rm Cora.zip

unzip PubMed.zip
rm PubMed.zip

cd -

# Download LM predictions and features
gdown 1vywgfqge2_O6kf3emtG7TiQD0oWdiz9v
unzip prt_lm.zip
rm prt_lm.zip

# arxiv_2023
# gdown https://drive.google.com/drive/folders/1QV0Ihwtfcgj6bjHktKzvDcDqoMuyZa7M --folder

# cd prt_lm
# # cora
# gdown https://drive.google.com/drive/folders/1mR_qtWhzH-LX34n4SaFnDFefcKovaVxy --folder
# gdown https://drive.google.com/drive/folders/1LrMVAaD5Lu2fvJ_SghfgsyiVeyhjDXv6 --folder
# # ogbn-arxiv
# gdown https://drive.google.com/drive/folders/1YebF5eo_Hm4A7CPrDDbq900UCE95iWOv --folder
# gdown https://drive.google.com/drive/folders/1cgdkU9pCNpZ3D7MA3lOx3x5s3OFDU6t5 --folder
# # pubmed
# gdown https://drive.google.com/drive/folders/1wys171LrFJA9Hx__1Fj_2K3SWwQfTsGg --folder
# gdown https://drive.google.com/drive/folders/1Kufal4btFKT7sYNVxoxIyhDiY3YBpsGn --folder

# cd -