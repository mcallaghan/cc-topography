#!/bin/bash

# Args: run_id, perplexity

echo "assembling plots for run $1"

cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_wgs.png plots_pub/all_topic_words_wgs.png

cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_wgs.png plots_pub/all_topic_words_ars.png

cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_wgs.png plots_pub/all_topic_words_oecds.png

cp plots/ipcc_representation/ipcc_rep_new$1_all.pdf plots_pub/ipcc_represntation.pdf

cp plots/ipcc_representation/ipcc_rep_oecds_time.pdf plots_pub/ipcc_rep_oecds_time.pdf

cp plots/ipcc_representation/ipcc_rep_oecds_simplified.pdf plots_pub/ipcc_rep_oecds_simplified.pdf
