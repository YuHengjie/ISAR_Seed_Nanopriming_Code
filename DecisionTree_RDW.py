# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
import graphviz 
from dtreeviz.trees import dtreeviz

import joblib
from sklearn.inspection import permutation_importance
import shap
from pdpbox import pdp
import lime
import lime.lime_tabular

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./dataset_RDW.xlsx",index_col = 0,)
data

# %%
X = data.loc[:,['Concentration (mg/L)','TEM size (nm)','Zeta potential (mV)']]
X

# %%
target_name = 'Root dry weight'
y = data.loc[:,target_name]
y

# %%
# run LightGBM models first to get train/test index
# same train/test set with LightGBM
X_cv_index = np.load('X_cv_index_RDW.npy',allow_pickle=True)
X_test_index = np.load('X_test_index_RDW.npy',allow_pickle=True)
X_cv = X.loc[X_cv_index]
X_test = X.loc[X_test_index]
y_cv = y.loc[X_cv_index]
y_test = y.loc[X_test_index]
X_cv.to_excel('./train.xlsx')
X_test.to_excel('./test.xlsx')
X_cv

# %% 使用gridsearch对所有特征进行决策树建模
parameter = {

    # 1.
    #"max_depth":np.arange(1,20,1),
    #"min_samples_leaf":  np.arange(1,20,1),
    #"min_samples_split":  np.arange(2,20,1),
    
    # 2.
    # "max_features":np.arange(1,8,1),
    # "max_leaf_nodes":np.arange(2,50,1),

}

model_gs = DecisionTreeClassifier(
                                 max_depth = 11, 
                                 min_samples_leaf = 1,  
                                 min_samples_split = 4,
                                 max_features = 3,
                                 max_leaf_nodes = 30,
                                random_state = 42,
                                )

grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='accuracy', cv=5, n_jobs=-1)

grid_search.fit(X_cv, y_cv)

print('best score: ', grid_search.best_score_)
print('best_params:', grid_search.best_params_)

DT_Gs_best = grid_search.best_estimator_


# %% 
model = DecisionTreeClassifier(
                                 max_depth = 11, 
                                 min_samples_leaf = 1,  
                                 min_samples_split = 4,
                                 max_features = 3,
                                 max_leaf_nodes = 30,
                                random_state = 42,
                                )
model.fit(X_cv, y_cv)
joblib.dump(model, 'dtmodel.pkl')

# %%
print('AUC_CV: %.2f ' %(sum(cross_val_score(model, X_cv, y_cv, cv=5, scoring='roc_auc'))/5))
print('F1_CV: %.2f ' %(sum(cross_val_score(model, X_cv, y_cv, cv=5, scoring='f1'))/5))
print('Accuracy_CV: %.2f ' %(sum(cross_val_score(model, X_cv, y_cv, cv=5, scoring='accuracy'))/5))

# %%
y_pred_cv = model.predict(X_cv)
y_proba_cv = model.predict_proba(X_cv)[:, 1]

print('Train AUC: %.2f'%metrics.roc_auc_score(y_cv,y_proba_cv))
print('Train F1: %.2f'%metrics.f1_score(y_cv,y_pred_cv))
print('Train Accuracy: %.2f'%metrics.accuracy_score(y_cv,y_pred_cv))
#print('Train MCC: %.2f'%metrics.matthews_corrcoef(y_cv,y_pred))

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred))
print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))
#print('Test MCC: %.2f'%metrics.matthews_corrcoef(y_test,y_pred))

# %%
fig, ax= plt.subplots(figsize = (3,3))
plt.style.use('classic')
plt.rcParams['font.size'] ='8'
plt.margins(0.02)

fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, 'dodgerblue',label = 'Test set')

fpr, tpr, threshold = metrics.roc_curve(y_cv, y_proba_cv)
plt.plot(fpr, tpr, 'hotpink',label = 'Train set')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],c='dimgray',linestyle='--')
margin = 0.03
plt.xlim([0-margin, 1+margin])
plt.ylim([0-margin, 1+margin])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

fig.savefig("./Image/Test_ROC_DT.jpg",dpi=600,bbox_inches='tight')

# %%
dot_data = tree.export_graphviz(model, out_file=None,filled=True
                        ) 
graph = graphviz.Source(dot_data, format="pdf") 
graph.render("decistion_tree_graphviz") 

# %%
viz = dtreeviz(model, X, y,
                target_name="Root dry weight",
                feature_names=X_cv.columns,
                class_names=['Low','High'])
viz

# %%
X_cv.columns


# %%
DT_importance = model.feature_importances_
figure = plt.figure(figsize=(8,3))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = DT_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         DT_importance[sorted_idx], align='center', color="#1E90FF")

plt.title('LightGBM feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=18)

figure.savefig("./Image/DT_importance_DT.jpg",dpi=600,bbox_inches='tight')

# %%
result = permutation_importance(model, X_cv, y_cv, scoring='accuracy', 
                                n_repeats=10, random_state=0, n_jobs=-1)
Permutation_importance = result.importances_mean

figure = plt.figure(figsize=(8,3))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = Permutation_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         Permutation_importance[sorted_idx], align='center', color="#1E90FF")

plt.title('Permutation feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)

figure.savefig("./Image/Permutation_importance_DT.jpg",dpi=600,bbox_inches='tight')

# %%
explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_cv)[1]
global_shap_values = np.abs(shap_values).mean(0)

figure = plt.figure(figsize=(8,3))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = global_shap_values.argsort()
sorted_features = X_cv.columns[sorted_idx]
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         global_shap_values[sorted_idx], align='center', color="#1E90FF")

plt.title('SHAP feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)
figure.savefig("./Image//Shap_importance_DT.jpg",dpi=600,bbox_inches='tight')

# %%
# Calculate the relative importance, the maximum is 1
DT_importance_relative = DT_importance/max(DT_importance)
Permutation_importance_relative = Permutation_importance/max(Permutation_importance)
shap_values__relative = global_shap_values/max(global_shap_values)

# Sort by the sum of relative importance
importance_sum = DT_importance_relative+Permutation_importance_relative+shap_values__relative
sorted_idx_sum = importance_sum.argsort()
sorted_features = X_cv.columns[sorted_idx_sum][::-1]

np.save('sorted_features.npy',sorted_features.tolist())

importance_df = pd.DataFrame({'Feature':X_cv.columns[sorted_idx_sum],
                    'Decision Tree':DT_importance_relative[sorted_idx_sum],
                    'Permutatio':Permutation_importance_relative[sorted_idx_sum],
                    'SHAP':shap_values__relative[sorted_idx_sum]},
                    )
importance_df

# %%
sorted_features

# %%
importance_df = pd.DataFrame(columns=('Feature','Method','Relative importance value'))
n_feature = len(X_cv.columns)

for i in range(0,n_feature):
    importance_df.loc[i,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i,'Method'] = 'Decision Tree'
    importance_df.loc[i,'Relative importance value'] = DT_importance_relative[sorted_idx_sum][-i-1]

for i in range(0,n_feature):
    importance_df.loc[i+n_feature,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature,'Method'] = 'Permutation'
    importance_df.loc[i+n_feature,'Relative importance value'] = Permutation_importance_relative[sorted_idx_sum][-i-1]
    
for i in range(0,n_feature):
    importance_df.loc[i+n_feature*2,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature*2,'Method'] = 'SHAP'
    importance_df.loc[i+n_feature*2,'Relative importance value'] = shap_values__relative[sorted_idx_sum][-i-1]

LightGBM_split_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][0:n_feature].values,reverse=True)
Permutation_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*1:n_feature*2].values,reverse=True)
SHAP_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values,reverse=True)

# %%
annotate_LightGBM_split = []
annotate_LightGBM_gain = []
annotate_Permutation = []
annotate_SHAP = []
n_feature = len(X_cv.columns)
for i in range(0,n_feature,1):
    annotate_LightGBM_split.append(LightGBM_split_sorted_value.index(importance_df.loc[:,'Relative importance value'][0:n_feature].values[i])+1)
    annotate_Permutation.append(Permutation_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature:n_feature*2].values[i])+1)
    annotate_SHAP.append(SHAP_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values[i])+1)
    
annotate_value = np.hstack((annotate_LightGBM_split, annotate_LightGBM_gain, annotate_Permutation,annotate_SHAP))
annotate_value

# %%
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

figure = plt.figure(figsize=(6,4))
plt.style.use('classic')
bar = sns.barplot(data = importance_df,y='Feature',x='Relative importance value',hue='Method',palette="rocket")
bar.set_ylabel('',fontsize=16)
bar.set_xlabel('Relative importance value',fontsize=16)
bar.set_yticklabels(feature_this_plot,fontsize=16)
plt.legend(loc='lower right')
i=0
plt.margins(0.02)
for p in bar.patches:
    bar.annotate("%d" %annotate_value[i], xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    i=i+1
figure.savefig("./Image/Importance_summary_DT.jpg",dpi=600,bbox_inches='tight')





# %%
shap_values = explainer.shap_values(X)[1] # for class 1

class ShapObject:
    
    def __init__(self, base_values, data, values, feature_names):
        self.base_values = base_values # Single value
        self.data = data # Raw feature values for 1 row of data
        self.values = values # SHAP values for the same row of data
        self.feature_names = feature_names # Column names
        

shap_object = ShapObject(base_values = explainer.expected_value[1],
                         values = shap_values[107],
                         feature_names = X.columns,
                         data = X.iloc[107,:])


figure = plt.figure()
shap.waterfall_plot(shap_object, show=False)
# 补充缺失的两条辅助线
plt.plot([explainer.expected_value[1]-0.053, explainer.expected_value[1]-0.053], 
            [-0.4, 0.6], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
plt.plot([explainer.expected_value[1]+0.183, explainer.expected_value[1]+0.183], [0.6, 1.58], 
            color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

figure.savefig("./Image/waterfall_plot_ZnO_30_DT.jpg",dpi=600,bbox_inches='tight')

# %%

shap_object = ShapObject(base_values = explainer.expected_value[1],
                         values = shap_values[105],
                         feature_names = X.columns,
                         data = X.iloc[105,:])

figure = plt.figure()
shap.waterfall_plot(shap_object, show=False)

figure.savefig("./Image/waterfall_plot_ZnO_106_DT.jpg",dpi=600,bbox_inches='tight')


# %%
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values, X)

# %%
figure = plt.figure()
shap.summary_plot(shap_values, features=X, max_display=10)
figure.savefig("./Image/Local_Shap_summary_feature_DT.jpg",dpi=600,bbox_inches='tight')

# %%
figure = plt.figure()
shap.decision_plot(explainer.expected_value[1],
                   shap_values, X.columns)
figure.savefig("./Image/Local_Shap_summary_decesion.jpg",dpi=600,bbox_inches='tight')

# %% SHAP Dependence Plots

# %% 1. 'Concentration (mg/L)' SHAP Dependence Plots
feature = 'Concentration (mg/L)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_Concentration_DT.jpg",dpi=600,bbox_inches='tight')

# %% 2. 'TEM size (nm)' SHAP Dependence Plots
feature = 'TEM size (nm)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_TEM_DT.jpg",dpi=600,bbox_inches='tight')

# %% 3. Zeta potential (mV) SHAP Dependence Plots
feature = 'Zeta potential (mV)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_zeta_DT.jpg",dpi=600,bbox_inches='tight')



# %% SHAP Main effect Plots
shap_interaction_values = shap.TreeExplainer(model=model).shap_interaction_values(X)[1]
np.array(shap_interaction_values).shape

# %% 1. 'Concentration (mg/L)' SHAP Main effect Plots
feature = 'Concentration (mg/L)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_Concentration_DT.jpg",dpi=600,bbox_inches='tight') 

# %% 2. 'TEM size (nm)' SHAP Main effect Plots
feature = 'TEM size (nm)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_%s_DT.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 2. 'TEM size (nm)' SHAP Main effect Plots
feature = 'TEM size (nm)'

fig, ax = plt.subplots()
ind = list(X.columns).index(feature)
shap_main_values = shap_interaction_values[:,ind,ind]
display_x = X.loc[:,feature]
coef = np.polyfit(display_x.values,shap_main_values, 3)  # 用1次多项式拟合，输出系数从高到0
func = np.poly1d(coef)  # 生成拟合后的函数方程
curve_x = np.linspace(min(display_x),max(display_x),500)
curve_y = func(curve_x)
plt.plot(curve_x,curve_y,linestyle='--',color="slategrey",alpha=0.8)

shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X, ax=ax)
plt.gcf().set_size_inches(4, 2)

plt.text(90, 0.3, 'R$^{2}$=%.2f'%metrics.r2_score(shap_main_values,func(display_x.values)), 
            color='#696969',fontsize='10')

fig.savefig("./Image/SHAP_Main_effects_%s_DT_curve.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 3. Zeta potential (mV) SHAP Main effect Plots

feature = 'Zeta potential (mV)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_%s_DT.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 3. Zeta potential (mV) SHAP Main effect Plots
feature = 'Zeta potential (mV)'

fig, ax = plt.subplots()
ind = list(X.columns).index(feature)
shap_main_values = shap_interaction_values[:,ind,ind]
display_x = X.loc[:,feature]
coef = np.polyfit(display_x.values,shap_main_values, 3)  # 用1次多项式拟合，输出系数从高到0
func = np.poly1d(coef)  # 生成拟合后的函数方程
curve_x = np.linspace(min(display_x),max(display_x),500)
curve_y = func(curve_x)
plt.plot(curve_x,curve_y,linestyle='--',color="slategrey",alpha=0.8)

shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X, ax=ax)
plt.gcf().set_size_inches(4, 2)

plt.text(20, 0.38, 'R$^{2}$=%.2f'%metrics.r2_score(shap_main_values,func(display_x.values)), 
            color='#696969',fontsize='10')

fig.savefig("./Image/SHAP_Main_effects_%s_DT_curve.jpg"%(feature),dpi=600,bbox_inches='tight') 



# %% SHAP interaction values
fig, ax= plt.subplots(figsize = (6, 6))

plt.style.use('default')
tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X.columns[inds]
df_temp2.index = X.columns[inds]

h=sns.heatmap(df_temp2, cmap='Blues', square=True, center=11,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)            
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/Feature_interaction_summary_heatmap_DT.jpg",dpi=600,bbox_inches='tight')

# %% 1. 'Concentration (mg/L)' and 'TEM size (nm)'   SHAP interaction values
fig, ax = plt.subplots(figsize=(4, 2))
feature2 = 'Concentration (mg/L)'
feature1 = 'TEM size (nm)'
shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,display_features=X)
# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#
fig.savefig("./Image/Interact_Concentration_TEM_DT.jpg",dpi=600,bbox_inches='tight')



# %% 1. 'Concentration (mg/L)' PDP one feature
feature = 'Concentration (mg/L)'

pdp_NP_none_M = pdp.pdp_isolate(model=model,
                        dataset=X,
                        model_features=X.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=10)
fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))
#axes['pdp_ax']['_count_ax'].set_xlabel(feature, fontsize=12)
fig.savefig("./Image/Local_PDP_Concentration_DT.jpg",dpi=600,bbox_inches='tight') 

# %% 2. 'TEM size (nm)' PDP one feature
feature = 'TEM size (nm)'

pdp_NP_none_M = pdp.pdp_isolate(model=model,
                        dataset=X,
                        model_features=X.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=20)
fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))
#axes['pdp_ax']['_count_ax'].set_xlabel(feature, fontsize=12)
fig.savefig("./Image/Local_PDP_%s_DT.jpg"%feature,dpi=600,bbox_inches='tight') 


# %% 3. Zeta potential (mV) PDP one feature
feature = "Zeta potential (mV)"

pdp_NP_none_M = pdp.pdp_isolate(model=model,
                        dataset=X,
                        model_features=X.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=20)
fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))
#axes['pdp_ax']['_count_ax'].set_xlabel(feature, fontsize=12)
fig.savefig("./Image/Local_PDP_%s_DT.jpg"%feature,dpi=600,bbox_inches='tight') 


# %% 1. 'Concentration (mg/L)' and 'TEM size (nm)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=['Concentration (mg/L)', 'TEM size (nm)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, ['Concentration (mg/L)', 'TEM size (nm)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5)
)

fig.savefig("./Image/Combined_PDP_Concentration_TEM_DT.jpg",dpi=600,bbox_inches='tight') 

# %% 2. 'Concentration (mg/L)' and 'Zeta potential (mV)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=['Concentration (mg/L)', 'Zeta potential (mV)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, ['Concentration (mg/L)', 'Zeta potential (mV)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5))

fig.savefig("./Image/Combined_PDP_Concentration_Zeta_DT.jpg",dpi=600,bbox_inches='tight') 

# %% 3. 'TEM size (nm)' and 'Zeta potential (mV)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=[ 'TEM size (nm)','Zeta potential (mV)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, [ 'TEM size (nm)','Zeta potential (mV)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5)
)

axes['pdp_inter_ax'].set_xticklabels(axes['pdp_inter_ax'].get_xticklabels(),
                    rotation=45,horizontalalignment='center',)
                    
fig.savefig("./Image/Combined_PDP_TEM_Zeta_DT.jpg",dpi=600,bbox_inches='tight') 


# %% LIME
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_cv.values), 
                                                feature_names=X_cv.columns,
                                                discretize_continuous=True,random_state=42)


# %% ZnO 30 
exp = explainer.explain_instance(np.array(X_test.loc[108,:].values), model.predict_proba,
                                          num_features=8)
lime_list = exp.as_list()

# %%
bars = []
height = []
for i in range(0,len(lime_list)):
    bars.append(lime_list[i][0])
    height.append(lime_list[i][1])
bars = bars[::-1]
height = height[::-1]

# %%
colors = ['#008BFA']
for i in range(1,len(lime_list)):
    colors.append('#008BFA')

for i in range(0,len(lime_list)):
    if height[i] >= 0:
        colors[i] = '#FF0050'

fig, ax= plt.subplots(figsize = (6,2.3))
plt.style.use('seaborn-ticks')
plt.margins(0.05)
plt.grid(linestyle=(0, (1, 6.5)),color='#B0B0B0',zorder=0)
plt.barh(range(0,len(lime_list)), height,color=colors,edgecolor = "none", zorder=3)
plt.yticks(range(0,len(lime_list)), bars,)

ax1=plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(False)

ax1.tick_params(top=False,
               bottom=True,
               left=False,
               right=False)

for i in range(0,len(bars)):
    if abs(height[i])<=0.025:
        if height[i]>0:
            plt.text(height[i]+0.018,i-0.1,"%.2f" %height[i],ha = 'center',color='#FF0050',)
        else:
            plt.text(height[i]-0.018,i-0.1,"%.2f" %height[i],ha = 'center',color='#008BFA',)
    else:
        plt.text(height[i]/2,i-0.1,"%.2f" %height[i],ha = 'center',color='w',)


fig.savefig("./Image/Lime_Local_ZnO_30_DT.jpg",dpi=600,bbox_inches='tight')

# %%
predict_proba_local = model.predict_proba([X_test.loc[108,:]])[0] # 
predict_proba_local = predict_proba_local[::-1]

fig, ax= plt.subplots(figsize = (2,1))
bars = ['High','Low']
plt.style.use('seaborn-ticks')
plt.margins(0.05)
plt.barh(range(0,len(predict_proba_local)), predict_proba_local,color=['#FF0050','#008BFA'])

for i in range(0,len(bars)):
    if predict_proba_local[i]<0.25:
        if i==1:
            plt.text(predict_proba_local[i]+0.12,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='#008BFA',)
        else:
            plt.text(predict_proba_local[i]+0.12,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='#FF0050',)

    else:
        plt.text(predict_proba_local[i]/2,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='w',)


ax.set_xticklabels([])

ax1=plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.tick_params(top=False,
               bottom=False,
               left=False,
               right=False)

plt.yticks(range(0,len(predict_proba_local)), bars)
plt.title('Prediction probabilities')
fig.savefig("./Image/Model_predict_proba_Local_ZnO_30_DT.jpg",dpi=600,bbox_inches='tight')



# %% 106 样本
exp = explainer.explain_instance(np.array(X_test.values)[2], model.predict_proba,
                                          num_features=8)
lime_list = exp.as_list()

# %%
bars = []
height = []
for i in range(0,len(lime_list)):
    bars.append(lime_list[i][0])
    height.append(lime_list[i][1])
bars = bars[::-1]
height = height[::-1]

# %%
colors = ['#008BFA']
for i in range(1,len(lime_list)):
    colors.append('#008BFA')

for i in range(0,len(lime_list)):
    if height[i] >= 0:
        colors[i] = '#FF0050'

fig, ax= plt.subplots(figsize = (6,2.3))
plt.style.use('seaborn-ticks')
plt.margins(0.05)
plt.grid(linestyle=(0, (1, 6.5)),color='#B0B0B0',zorder=0)
plt.barh(range(0,len(lime_list)), height,color=colors,edgecolor = "none", zorder=3)
plt.yticks(range(0,len(lime_list)), bars,)

ax1=plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(False)

ax1.tick_params(top=False,
               bottom=True,
               left=False,
               right=False)

for i in range(0,len(bars)):
    if abs(height[i])<=0.025:
        if height[i]>0:
            plt.text(height[i]+0.018,i-0.1,"%.2f" %height[i],ha = 'center',color='#FF0050',)
        else:
            plt.text(height[i]-0.018,i-0.1,"%.2f" %height[i],ha = 'center',color='#008BFA',)
    else:
        plt.text(height[i]/2,i-0.1,"%.2f" %height[i],ha = 'center',color='w',)


fig.savefig("./Image/Lime_Local_106_DT.jpg",dpi=600,bbox_inches='tight')

# %%
predict_proba_local = model.predict_proba(X_test)[2]
predict_proba_local = predict_proba_local[::-1]

fig, ax= plt.subplots(figsize = (2,1))
bars = ['High','Low']
plt.style.use('seaborn-ticks')
plt.margins(0.05)
plt.barh(range(0,len(predict_proba_local)), predict_proba_local,color=['#FF0050','#008BFA'])

for i in range(0,len(bars)):
    if predict_proba_local[i]<0.25:
        if i==1:
            plt.text(predict_proba_local[i]+0.12,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='#008BFA',)
        else:
            plt.text(predict_proba_local[i]+0.12,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='#FF0050',)

    else:
        plt.text(predict_proba_local[i]/2,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='w',)


ax.set_xticklabels([])

ax1=plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.tick_params(top=False,
               bottom=False,
               left=False,
               right=False)

plt.yticks(range(0,len(predict_proba_local)), bars)
plt.title('Prediction probabilities')
fig.savefig("./Image/Model_predict_proba_Local_106_DT.jpg",dpi=600,bbox_inches='tight')
