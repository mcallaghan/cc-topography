#!/bin/bash

# Args: run_id, perplexity

echo "assembling plots for run $1"

cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_oecds.png plots_pub/all_topic_words_oecds.png
cp tsne_results/plots/run_$1_s_0_p$2_all_topic_words_oecds.pdf plots_pub/all_topic_words_oecds.pdf

cp tsne_results/plots/run_$1_s_0_p$2_evolution_4.png plots_pub/topic_evolution_4.png
cp tsne_results/plots/run_$1_s_0_p$2_evolution_4.pdf plots_pub/topic_evolution_4.pdf


cp plots_pub/topic_oecd_entropy.pdf plots_pub/upload_figures/Figure_SI_1.pdf
cp plots_pub/single_doc_3_536594_1861.pdf plots_pub/upload_figures/Figure_SI_2.pdf
cp plots_pub/ipcc_rep_wcs_simplified.pdf plots_pub/upload_figures/Figure_SI_3.pdf
cp plots_pub/wgs_socsci.pdf plots_pub/upload_figures/Figure_SI_4.pdf
cp plots_pub/topic_rep_ks.pdf plots_pub/upload_figures/Figure_SI_5.pdf
cp plots_pub/pubs_time_wgb.pdf plots_pub/upload_figures/Figure_1.pdf
cp plots_pub/all_topic_words_oecds.png plots_pub/upload_figures/Figure_2.png
cp plots_pub/all_topic_words_oecds.pdf plots_pub/upload_figures/Figure_2_hq.pdf
cp plots_pub/topic_evolution_4.png plots_pub/upload_figures/Figure_3.png
cp plots_pub/topic_evolution_4.pdf plots_pub/upload_figures/Figure_3_hq.pdf
cp plots_pub/big_panel_representation.pdf plots_pub/upload_figures/Figure_4.pdf
