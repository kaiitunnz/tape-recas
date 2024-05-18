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
    # gdown $2
    # gdown $3
    gdown.download(
    f"https://drive.google.com/uc?export=download&confirm=pbef&id={$2}",
    output
    )
    gdown.download(
    f"https://drive.google.com/uc?export=download&confirm=pbef&id={$3}",
    output
    )
    cd -
}

mkdir -p prt_lm
# 1. tape-arxiv-2023
# seed 0
download_pred_emb arxiv_2023 1tujyreEsPm67dZCmEvciMzp14ta_7Qvm 1Tz4E3l8QCvgCJnqPaFdmteRDQW2Tlr69
download_pred_emb arxiv_20232 1WEIAV7ZZmR7h00f2akBxS9-5wonkV2oB 1iPSyjAqI2eoBHRm7jjVjpqflYZKDuJ9W
# seed 1
download_pred_emb arxiv_2023 1ABu0ie_TzJo1S2gQjyNnwfOx64u4zf4h 1ddwqcN-2hnjxX72Ti7U8FQoTLKUr4kc7
download_pred_emb arxiv_20232 1hgxKbe5otmOWrSEIz4_j3cOJw42FI5dl 1f5DwEJmDNuaUkQzt8vOArZ4zwTOR_vs1
# seed 2
download_pred_emb arxiv_2023 1z-xfeZg9RJMN4L5BXxoU5Wxbwb6UzCU_ 1170k4vcqL2x-xOY9nSs7y_MBrdiF0xsu
download_pred_emb arxiv_20232 1oHHJlnVnq4IOfJnZkcV1qmSZpirp8SOr 1FMBonRykZEAyCx-yFOg6-Sj5eG38wfBr
# seed 3
download_pred_emb arxiv_2023 12EGPfL99fVI7HfVqySqUqls-Eoung_P6 1_zKy5eC1pxuYQeG-6R9nxcHMj5tebnb7
download_pred_emb arxiv_20232 13WWT3FDdgxt_Grro48ZL8P4v4LW0WgJZ 1OKD8er66F6hsvHdrWNpT11V_zY5yAXLV

# 2. ogbn-arxiv
# seed 0
download_pred_emb ogbn-arxiv 1APw6hnOtdMFa5rXhhC9520ZmoXhytDDr 137uBunrp0jsiG_AFvlcCME4kbldGDkLW
download_pred_emb ogbn-arxiv2 1WUsYI3MM-K9mtB9mPnqJlh9DalHB8a1l 1jnKdDYT_F9Fx9w29h1uD6aeDm_JK0kDi
# seed 1
download_pred_emb ogbn-arxiv 1AVjT-gvWoi-oG6gSouGFqyTYxbgFrVwY 1o56qGSXk__fqLECT3BRLjnR8SZEk-B3A
download_pred_emb ogbn-arxiv2 1jAImJfoM7EQubWZZFTXLDbZ2vsOK0TKR 16FIlpl2uOhwvf2cvSWJIWBLftAs-SVtw
# seed 2
download_pred_emb ogbn-arxiv 1V_NWBAGXI4i6u6kVSowuuilEmYS4mjUA 1cd4oC2PBaEw9s116hjnOOa7H9dZhdTrw
download_pred_emb ogbn-arxiv2 1uivNxXc80Q6TWHnJchBgBoN9lao4LSqw 1DFaBWGCCp99fEODvFx8XHv1mJKlAEtHl
# seed 3
download_pred_emb ogbn-arxiv 163qJKBKt-WtdYOScdBAGjFydtuzGvqtk 1b6QklL7bzmmLD_Ig7rz1syUeAj8ZJ28M
download_pred_emb ogbn-arxiv2 1yxtr2WAUf3Ss9jWwz2c5AfKnca2z9iz- 1yGRuXskVdVs-fdzrlSEvdAynEaVEF_2i

# 3. cora
# seed 0
download_pred_emb cora 1lVF0aJFUCUBCTKOq8Jm5ju8fKhKW-coB 1LDXJHeJrqQa2e8u6ivEgJVJUpKo1DTmL
download_pred_emb cora2 1potvRUweicMGy_F8I9QRS4ySctp4opfh 1b4EILJ9RR6nBtzpDH-I2OYl4TTukH9u6
# seed 1
download_pred_emb cora 1pl-2K10cRc7guT1QOK3vUjbYkD68XvRv 1xxPAFzBxlpHO464bQ0eXtX3Ky5Oc5S2u
download_pred_emb cora2 19BEyfy0rD33jMnJprzawfaIGlkwHScH1 19DsrAJGzSE8jyHgSLSagpL288rd7JxAq
# seed 2
download_pred_emb cora 1ryJVh0kXSkglkesOD8hl4XsWjxk8WJRq 1_51g3cx0uQmHoYvUEItz1RPvRsgQ3Efi
download_pred_emb cora2 1oBwoKX-zRU8r9BvmMh5VCOr4t463p1sZ 1_lzhP2zZTQWkVsCF-9HpwOoOvO94saJ5
# seed 3
download_pred_emb cora 1JTwi1ozpz7OOeX1Ls8Qz0AGsRNiaWLQi 1Ttn1zlI20PslTcjDZodYelRoHUUuxoRY
download_pred_emb cora2 1zsEoMjhKmjTrzttCPzQ5iH9HeiWwuLBw 1Wa2cLS3gBUv5OTL-hBzIPs-0zW1Bmuz4

# 4. pubmed
# seed 0
download_pred_emb pubmed 1u8yDPzCrOoFdr5jxp_5Dyeu97NtAKrcN 1-ov-yRwodMmyGPWpA7wXZ02pT4oF5JFp
download_pred_emb pubmed2 1VoQhGXKILNq2CHuq1Jzlry7rehL6Oxx_ 1CvbVs3QDrS1ulwJ2-d8hKwVHNkwFRnEM
# seed 1
download_pred_emb pubmed 1UKBEMOKcVqz25Vpj8dTg-J5bc1uzeGUN 1ynKc7znF6B08-l9shUkk7vL0pHu_xftz
download_pred_emb pubmed2 1I5538W6p1Nwy28E4_4N_RSYiT8V4pPT5 11V3H4c-T8QkSTLARi_xE1R1f4EtyD8iL
# seed 2
download_pred_emb pubmed 1qNInZnoXGmpHozfU5yyvbj8UtwP9Ghnp 1uFxpPtXVCkF2v9PZFmitAB8vQ38jz3Ax
download_pred_emb pubmed2 1JzsxTwxpw2AFe2HasSrIR9woaCdFb3qi 1oKkilbYv3EP6x_Dd7Z7qURllgm03jOCI
# seed 3
download_pred_emb pubmed 1TXRIyMLoc-scvUQgwwPQwqG8-piVVc0J 1J4lmW3VxYjKEjK-jAVU6T6OinykAgKsh
download_pred_emb pubmed2 1enpeI-VgkiN8hclw_EEgAstIaqxiwX74 1hyTL26U0LS1n-Bkc5xHTWZj8uriOwWDn