# Additional Analysis

In addition to the experiments presented in our article, we further examined the performance of the post-processing pipeline on the BELOPSEM language pairs HSB-DE and CHV-RU (train datasets).
We set up the pipeline using the best-performing configuration in terms of components and hyperparameters, with the exception of the final filtering threshold.
Please refer to `precision_recall_plot_hsb-de.pdf` and `precision_recall_plot_chv-ru.pdf` for the respective results.

In these plots, the x-axis shows the applied filtering threshold, and the y-axis indicates the precision/recall value. 
The dashed line represents the threshold used in our article to achieve the best results for each language pair.