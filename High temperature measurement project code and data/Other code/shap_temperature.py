# -------------------------------------------------回归的贡献程度----------------------------------------------------------------------
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
import pandas as pd
import os
shap.initjs()

normalized_data = pd.read_excel(r'D:\Molten salt temperature prediction\ALL_data_txt\Laboratory dataset.xlsx', engine='openpyxl')
X = normalized_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
y = normalized_data.iloc[:, 19]
size = len(y)

model = xgboost.XGBRegressor().fit(X, y)

explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap.Explanation(values=shap_values[1000], base_values=explainer.expected_value, data=X.iloc[1000], feature_names=X.columns), max_display=13, show=False)
plt.tight_layout()
plt.show()

# visualize the first prediction's explanation with a force plot
shap.summary_plot(shap_values, X)
plt.show()
shap.summary_plot(shap_values, X, plot_type="bar")

# shap.summary_plot(shap_values, X, plot_type="bar")
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.show()

# visualize the first prediction's explanation with a force plot and save it as HTML file
shap.plots.force(shap_values[1000], show=False, matplotlib=True, figsize=(26, 5))
plt.tight_layout()
plt.show()

explainer = shap.Explainer(model, X)
shap_values = explainer(X[:size])
shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1), max_display=20, show=False)
plt.tight_layout()
plt.show()

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:,"G_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"B_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"L_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"R_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"Gray_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"L/b"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"b_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"L/a"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"R/G"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"H/V"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"a/b"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"a_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"S_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"G/B"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"H_mean"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"H/S"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"S/V"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"R/B"], color=shap_values[:,"G_mean"])
shap.plots.scatter(shap_values[:,"V_mean"], color=shap_values[:,"G_mean"])
plt.tight_layout()
plt.show()

