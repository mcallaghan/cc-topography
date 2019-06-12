#!/bin/bash

# Args: run_id, perplexity

echo "assembling plots for run $1"

cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_oecds.png plots_pub/all_topic_words_oecds.png
cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_oecds.pdf plots_pub/all_topic_words_oecds.pdf

cp tsne_results/plots/run_$1_s_0_p$2_evolution_4.png plots_pub/topic_evolution_4.png
cp tsne_results/plots/run_$1_s_0_p$2_evolution_4.pdf plots_pub/topic_evolution_4.pdf
