# %% 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from sklearn.inspection import permutation_importance
import shap
from pdpbox import pdp
import lime
import lime.lime_tabular

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./dataset_RDW.xlsx",index_col = 0,)
data

# %%


# %% 绘制直方图
plt.figure(figsize=(8,10))
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

for i in range(1,11,1):
    plt.subplot(4,3,i)
    if i in [1,4,10]:
        if i == 1:
            data[data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=1)
            plt.xticks(rotation = 45,horizontalalignment='right')
        else:
            data[data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=0.15)
            plt.xticks(rotation = 0,horizontalalignment='center')
    else:
        plt.hist(data.iloc[:,i-1], facecolor="#FF8C00", edgecolor="black", alpha=0.7)
    plt.xlabel(data.columns[i-1])
    plt.ylabel("Freqency")
    
plt.tight_layout()
plt.savefig("./Image/dataset_visual.jpg",dpi=600,bbox_inches='tight')

# %% 查看变量间的相关性
X = data.drop(columns=['Root dry weight']) # 'TEM size (nm)'
X_corr = X.copy()
corr = X_corr.corr()

fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')

h=sns.heatmap(corr, cmap='Blues',  square=True, center=0.5,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':8})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/X_corr.jpg",dpi=600,bbox_inches='tight')

# %%
X = X.drop(columns=['TEM size SD (nm)']) # 'TEM size (nm)'
X = X.drop(columns=['PdI']) # 'TEM size (nm)'
X_raw = X.copy() # 用于shap可视化
X_raw

# %%
le_composition = LabelEncoder()
le_composition.fit(X['Composition'])
X['Composition'] = le_composition.transform(X['Composition'])
print(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])))

le_morphology = LabelEncoder()
le_morphology.fit(X['Morphology'])
X['Morphology'] = le_morphology.transform(X['Morphology'])
print(list(le_morphology.inverse_transform([0,1])))
X

# %%
target_name = 'Root dry weight'
y = data.loc[:,target_name]
y

# %%
# run the following code from 1 to 10 (random_state) and record model performance and hyperparameters
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=10) 
# the sixth dataset split was used for model interpretation
for train,test in sss.split(X, y):
    X_cv = X.iloc[train]
    y_cv = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]
#np.save('X_cv_index_RDW.npy',X_cv.index)
#np.save('X_test_index_RDW.npy',X_test.index)
X_cv

# %%
parameter = {

    # 1.
    # "min_data_in_leaf":np.arange(1,21,1),
    # "min_sum_hessian_in_leaf": np.arange(1,11,1),
    
    # 2.
    # "max_bin":  np.arange(1,21,1),

    # 3.
    # "max_depth":np.arange(1,11,1),
    # "num_leaves": np.arange(1,21,1),

    # 4.
    # "learning_rate": np.arange(0.001,0.2,0.001),

}

model_gs = lgb.LGBMClassifier(n_jobs=-1,n_estimators=1000,max_cat_to_onehot=9,
                              random_state=42,
                              min_data_in_leaf=10,
                              min_sum_hessian_in_leaf=1,
                              max_bin=5,
                              max_depth=6,
                              num_leaves=9,
                              learning_rate=0.060,
                            )

grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_cv, y_cv,categorical_feature=['Composition','Morphology'])

print('best score: %.3f '%grid_search.best_score_)
print('best_params:', grid_search.best_params_)

LGBM_Gs_best = grid_search.best_estimator_

# %%
print('AUC_CV: %.2f ' %(sum(cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5))
print('F1_CV: %.2f ' %(sum(cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='f1'))/5))
print('Accuracy_CV: %.2f ' %(sum(cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5))

# %%
y_pred_cv = LGBM_Gs_best.predict(X_cv)
y_proba_cv = LGBM_Gs_best.predict_proba(X_cv)[:, 1]

print('Train AUC: %.2f'%metrics.roc_auc_score(y_cv,y_proba_cv))
print('Train F1: %.2f'%metrics.f1_score(y_cv,y_pred_cv))
print('Train Accuracy: %.2f'%metrics.accuracy_score(y_cv,y_pred_cv))
#print('Train MCC: %.2f'%metrics.matthews_corrcoef(y_cv,y_pred))

y_pred = LGBM_Gs_best.predict(X_test)
y_proba = LGBM_Gs_best.predict_proba(X_test)[:, 1]

print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred))
print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))
#print('Test MCC: %.2f'%metrics.matthews_corrcoef(y_test,y_pred))

# %%

# %%
# run the following code from 1 to 10 (random_state) and record model performance and hyperparameters
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=5) 
# the sixth dataset split was used for model interpretation
for train,test in sss.split(X, y):
    X_cv = X.iloc[train]
    y_cv = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

np.save('X_cv_index_RDW.npy',X_cv.index)
np.save('X_test_index_RDW.npy',X_test.index)

model = lgb.LGBMClassifier(n_jobs=-1,n_estimators=1000,max_cat_to_onehot=9,
                              random_state=42,
                              min_data_in_leaf=1,
                              min_sum_hessian_in_leaf=3,
                              max_bin=14,
                              max_depth=4,
                              num_leaves=5,
                              learning_rate=0.081,
                         )

model.fit(X_cv, y_cv,categorical_feature=['Composition','Morphology'])
LightGBM_importance_split = model.feature_importances_
# # save model
# model.booster_.save_model('lgbm_model.txt')
joblib.dump(model, 'lgbmodel.pkl')
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
y_cv_proba = model.predict_proba(X_cv)[:, 1]

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

fpr, tpr, threshold = metrics.roc_curve(y_cv, y_cv_proba)
plt.plot(fpr, tpr, 'hotpink',label = 'Train set')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],c='dimgray',linestyle='--')
margin = 0.03
plt.xlim([0-margin, 1+margin])
plt.ylim([0-margin, 1+margin])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

fig.savefig("./Image/Test_ROC.jpg",dpi=600,bbox_inches='tight')



# %%
figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = LightGBM_importance_split.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         LightGBM_importance_split[sorted_idx], align='center', color="#1E90FF")

plt.title('LightGBM feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=18)

figure.savefig("./Image/LightGBM_importance.jpg",dpi=600,bbox_inches='tight')

# %%
result = permutation_importance(model, X_cv, y_cv, scoring='accuracy', 
                                n_repeats=10, random_state=0, n_jobs=-1)
Permutation_importance = result.importances_mean

figure = plt.figure(figsize=(8,6))
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

figure.savefig("./Image/Permutation_importance.jpg",dpi=600,bbox_inches='tight')

# %%
explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_cv)[1]
global_shap_values = np.abs(shap_values).mean(0)

figure = plt.figure(figsize=(8,6))
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
figure.savefig("./Image//Shap_importance.jpg",dpi=600,bbox_inches='tight')

# %%
# Calculate the relative importance, the maximum is 1
LightGBM_importance_split_relative = LightGBM_importance_split/max(LightGBM_importance_split)
Permutation_importance_relative = Permutation_importance/max(Permutation_importance)
shap_values__relative = global_shap_values/max(global_shap_values)

# Sort by the sum of relative importance
importance_sum = LightGBM_importance_split_relative+Permutation_importance_relative+shap_values__relative
sorted_idx_sum = importance_sum.argsort()
sorted_features = X_cv.columns[sorted_idx_sum][::-1]

np.save('sorted_features.npy',sorted_features.tolist())

importance_df = pd.DataFrame({'Feature':X_cv.columns[sorted_idx_sum],
                    'LightGBM (split)':LightGBM_importance_split_relative[sorted_idx_sum],
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
    importance_df.loc[i,'Method'] = 'LightGBM'
    importance_df.loc[i,'Relative importance value'] = LightGBM_importance_split_relative[sorted_idx_sum][-i-1]

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

figure = plt.figure(figsize=(6,5))
plt.style.use('classic')
bar = sns.barplot(data = importance_df,y='Feature',x='Relative importance value',hue='Method',palette="rocket")
bar.set_ylabel('',fontsize=16)
bar.set_xlabel('Relative importance value',fontsize=16)
bar.set_yticklabels(feature_this_plot,fontsize=16)
plt.legend(loc='lower right')
i=0
plt.margins(0.02)
for p in bar.patches:
    if p.get_width()>=0:
        bar.annotate("%d" %annotate_value[i], xy=(p.get_width(), p.get_y()+p.get_height()/2),
                xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    else:
        bar.annotate("%d" %annotate_value[i], xy=(0, p.get_y()+p.get_height()/2),
        xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    i=i+1
figure.savefig("./Image/Importance_summary.jpg",dpi=600,bbox_inches='tight')


# %%
shap_values = explainer.shap_values(X)[1] # for class 1
shap_excepeted_value = explainer.expected_value[1]
np.save('shap_values', shap_values)
np.save('shap_excepeted_value', shap_excepeted_value)

# %%
class ShapObject:
    
    def __init__(self, base_values, data, values, feature_names):
        self.base_values = base_values # Single value
        self.data = data # Raw feature values for 1 row of data
        self.values = values # SHAP values for the same row of data
        self.feature_names = feature_names # Column names
        
shap_object = ShapObject(base_values = explainer.expected_value[1],
                         values = shap_values[107],
                         feature_names = X.columns,
                         data = X_raw.iloc[107,:])

figure = plt.figure()
shap.waterfall_plot(shap_object, show=False)
# 补充缺失的两条辅助线
plt.plot([explainer.expected_value[1]+0.16, explainer.expected_value[1]+0.16], 
            [0.6, 1.58], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
plt.plot([explainer.expected_value[1]+0.51, explainer.expected_value[1]+0.51], [1.6, 2.58], 
            color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

figure.savefig("./Image/waterfall_plot_ZnO_30.jpg",dpi=600,bbox_inches='tight')

# %%
shap_object = ShapObject(base_values = explainer.expected_value[1],
                         values = shap_values[0],
                         feature_names = X.columns,
                         data = X_raw.iloc[0,:])

figure = plt.figure()
shap.waterfall_plot(shap_object, show=False)

figure.savefig("./Image/waterfall_plot_0.jpg",dpi=600,bbox_inches='tight')

# %%
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values, X)

# %%
figure = plt.figure()
shap.summary_plot(shap_values, features=X, max_display=10)
figure.savefig("./Image/Local_Shap_summary_feature.jpg",dpi=600,bbox_inches='tight')

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
                               display_features=X_raw,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_Concentration.jpg",dpi=600,bbox_inches='tight')

# %% 2. 'TEM size (nm)' SHAP Dependence Plots
feature = 'TEM size (nm)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X_raw,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_TEM.jpg",dpi=600,bbox_inches='tight')

# %% 3. Zeta potential (mV) SHAP Dependence Plots
feature = 'Zeta potential (mV)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X_raw,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_zeta.jpg",dpi=600,bbox_inches='tight')

# %% 4. BET surface area (m2/g) SHAP Dependence Plots
feature = 'BET surface area (m2/g)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X_raw,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_BET.jpg",dpi=600,bbox_inches='tight')

# %% 5.Composition SHAP Dependence Plots
feature = 'Composition'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X_raw,)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')   

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_Composition.jpg",dpi=600,bbox_inches='tight')

# %% 6. Hydrodynamic diameter (nm) SHAP Dependence Plots
feature = 'Hydrodynamic diameter (nm)'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X_raw,)

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_HydroDiameter.jpg",dpi=600,bbox_inches='tight')

# %% 7. Morphology SHAP Dependence Plots
feature = 'Morphology'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 2))

ax_shap = shap.dependence_plot(feature, shap_values, X,ax=ax, show=False,
                               display_features=X_raw,)
ax.set_xticks(range(0,2)) 
ax.set_xticklabels(ax.get_xticklabels(),rotation=0,ha='center') 

plt.rcParams.update({'font.size': 5})
fig.savefig("./Image/SHAP_dependence_Morphology.jpg",dpi=600,bbox_inches='tight')



# %% SHAP Main effect Plots
shap_interaction_values = shap.TreeExplainer(model=model).shap_interaction_values(X)
np.array(shap_interaction_values).shape

# %% 1. 'Concentration (mg/L)' SHAP Main effect Plots
feature = 'Concentration (mg/L)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_Concentration.jpg",dpi=600,bbox_inches='tight') 

# %% 2. 'TEM size (nm)' SHAP Main effect Plots
feature = 'TEM size (nm)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, 
                        X, show=False,display_features=X_raw, ax=ax,)

plt.gcf().set_size_inches(4, 2)
fig.savefig("./Image/SHAP_Main_effects_%s.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 2. TEM size (nm) SHAP Main effect Plots with curve
def piecewise_linear(x, k1, k2, b1, b2):
    return np.piecewise(x, [x <= 21.5], [lambda x:k1*x + b1, lambda x:k2*x + b2])

feature = 'TEM size (nm)'
fig, ax = plt.subplots()
ind = list(X.columns).index(feature)
shap_main_values = shap_interaction_values[:,ind,ind]
display_x = X_raw.loc[:,feature]

k1, k2, b1, b2 = optimize.curve_fit(piecewise_linear, display_x.values, shap_main_values)[0]
y_temp = [piecewise_linear(x_temp, k1, k2, b1, b2) for x_temp in display_x.values]

curve_x = np.linspace(min(display_x),max(display_x),500)
curve_y = [piecewise_linear(x_temp, k1, k2, b1, b2) for x_temp in curve_x]

p_index = np.where(curve_x>21.5)[0][0]

plt.plot(curve_x[0:p_index],curve_y[0:p_index],linestyle='--',color="slategrey",alpha=0.8)
plt.plot(curve_x[p_index::],curve_y[p_index::],linestyle='--',color="slategrey",alpha=0.8)

shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)

plt.text(100, -1, 'R$^{2}$=%.2f'%metrics.r2_score(shap_main_values,y_temp), 
            color='#696969',fontsize='10')

fig.savefig("./Image/SHAP_Main_effects_%s_curve.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 3. Zeta potential (mV) SHAP Main effect Plots
feature = 'Zeta potential (mV)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_%s.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 3. Zeta potential (mV) SHAP Main effect Plots with curve
feature = 'Zeta potential (mV)'
fig, ax = plt.subplots()

ind = list(X.columns).index(feature)
shap_main_values = shap_interaction_values[:,ind,ind]
display_x = X_raw.loc[:,feature]
coef = np.polyfit(display_x.values,shap_main_values, 1)  # 用1次多项式拟合，输出系数从高到0
func = np.poly1d(coef)  # 生成拟合后的函数方程
curve_x = np.linspace(min(display_x),max(display_x),500)
curve_y = func(curve_x)
plt.plot(curve_x,curve_y,linestyle='--',color="slategrey",alpha=0.8)

shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)

plt.text(25, 0.3, 'R$^{2}$=%.2f'%metrics.r2_score(shap_main_values,func(display_x.values)), 
            color='#696969',fontsize='10')

fig.savefig("./Image/SHAP_Main_effects_%s_curve.jpg"%(feature),dpi=600,bbox_inches='tight') 


# %% 4. BET surface area (m2/g) SHAP Main effect Plots
feature = 'BET surface area (m2/g)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_BET.jpg",dpi=600,bbox_inches='tight') 

# %% 5.Composition SHAP Main effect Plots
feature = 'Composition'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')   
fig.savefig("./Image/SHAP_Main_effects_%s.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 6. Hydrodynamic diameter (nm) SHAP Main effect Plots
feature = 'Hydrodynamic diameter (nm)'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)

fig.savefig("./Image/SHAP_Main_effects_%s.jpg"%(feature),dpi=600,bbox_inches='tight') 

# %% 7. Morphology SHAP Main effect Plots
feature = 'Morphology'
fig, ax = plt.subplots()
shap.dependence_plot((feature, feature), shap_interaction_values, X, show=False,display_features=X_raw, ax=ax)
plt.gcf().set_size_inches(4, 2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0,ha='center') 

fig.savefig("./Image/SHAP_Main_effects_%s.jpg"%(feature),dpi=600,bbox_inches='tight') 



# %% SHAP interaction values
fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')
tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X.columns[inds]
df_temp2.index = X.columns[inds]

h=sns.heatmap(df_temp2, cmap='Blues', square=True, center=35,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)            
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/Feature_interaction_summary_heatmap.jpg",dpi=600,bbox_inches='tight')

# %% 1. 'Concentration (mg/L)' and 'Zeta potential (mV)'   SHAP interaction values
fig, ax = plt.subplots(figsize=(4, 2))
feature2 = 'Concentration (mg/L)'
feature1 = 'Zeta potential (mV)'
shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,
                        display_features=X_raw)
# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

fig.savefig("./Image/Interact_Concentration_Zeta.jpg",dpi=600,bbox_inches='tight')

# %% 1. 'Concentration (mg/L)' and 'Zeta potential (mV)'   SHAP interaction values
def logistic(x, a, b, c, d):
    return a/(1+np.exp(b+c*x))+d

feature_names = list(X.columns)

fig, ax = plt.subplots(figsize=(4, 2))
feature1 = 'Zeta potential (mV)'
feature2 = 'Concentration (mg/L)'

ind1 = feature_names.index(feature1)
ind2 = feature_names.index(feature2)
interact_pair_values = shap_interaction_values[:,ind1,ind2]*2

display_x2 = X_raw.loc[:,feature2][X_raw.loc[:,feature2]==100]
display_x1 = X_raw.loc[display_x2.index,feature1]
ineter_plt_100 = interact_pair_values[display_x2.index-1]
a, b, c, d = optimize.curve_fit(logistic, display_x1.values, ineter_plt_100)[0] # the results of curve_fit are strange
fit_pd = pd.DataFrame(columns=['x','y'])
fit_pd['x']= display_x1.values
fit_pd['y']= ineter_plt_100
fit_pd.to_excel('zeta_conc_interact_100.xlsx')
a, b, c, d = [-2.282,4.66,0.5478,0.839] # so we use the results of Curve Fitting Tool in Matlab
y_temp_100 = logistic(display_x1.values,a, b, c, d)
curve_x = np.linspace(min(display_x1),max(display_x1),500)
curve_y = logistic(curve_x,a, b, c, d)
plt.plot(curve_x,curve_y,linestyle='--',color="#E789C3",alpha=0.6,zorder=0)

display_x2 = X_raw.loc[:,feature2][X_raw.loc[:,feature2]==50]
display_x1 = X_raw.loc[display_x2.index,feature1]
ineter_plt_50 = interact_pair_values[display_x2.index-1]
a, b, c, d = optimize.curve_fit(logistic, display_x1.values, ineter_plt_50)[0] 
y_temp_50 = logistic(display_x1.values,a, b, c, d)
curve_x = np.linspace(min(display_x1),max(display_x1),500)
curve_y = logistic(curve_x,a, b, c, d)
plt.plot(curve_x,curve_y,linestyle='--',color="#FC8D62",alpha=0.6,zorder=0)


shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,
                        display_features=X_raw)

# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

plt.text(0, 1.5, 'R$^{2}$=%.2f'%metrics.r2_score(ineter_plt_100,y_temp_100), 
            color='#E789C3',fontsize='10')

plt.text(30, 1.5, 'R$^{2}$=%.2f'%metrics.r2_score(ineter_plt_50,y_temp_50), 
            color='#FC8D62',fontsize='10')

fig.savefig("./Image/Interact_Concentration_Zeta_curve.jpg",dpi=600,bbox_inches='tight')


# %% 2. 'Concentration (mg/L)' and 'TEM size (nm)'   SHAP interaction values
fig, ax = plt.subplots(figsize=(4, 2))
feature1 = 'TEM size (nm)'
feature2 = 'Concentration (mg/L)'
shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,display_features=X_raw)

# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

fig.savefig("./Image/Interact_Concentration_TEM.jpg",dpi=600,bbox_inches='tight')


# %% 2. 'Concentration (mg/L)' and 'TEM size (nm)'   SHAP interaction values
def plateau(x, a, b, c):
    return a*x/(b+x)+c

fig, ax = plt.subplots(figsize=(4, 2))
feature1 = 'TEM size (nm)'
feature2 = 'Concentration (mg/L)'

ind1 = feature_names.index(feature1)
ind2 = feature_names.index(feature2)
interact_pair_values = shap_interaction_values[:,ind1,ind2]*2
display_x2 = X_raw.loc[:,feature2][(X_raw.loc[:,feature2]==50)|(X_raw.loc[:,feature2]==100)]
display_x1 = X_raw.loc[display_x2.index,feature1]
ineter_plt = interact_pair_values[display_x2.index-1]
a, b, c = optimize.curve_fit(plateau, display_x1.values, ineter_plt)[0] 
y_temp = [plateau(x_temp,a, b, c) for x_temp in display_x1.values]
curve_x = np.linspace(min(display_x1),max(display_x1),500)
curve_y = [plateau(x_temp,a, b, c) for x_temp in curve_x]
plt.plot(curve_x,curve_y,linestyle='--',color="#F18B92",alpha=0.8)

shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,display_features=X_raw)

# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

plt.text(100, 1.1, 'R$^{2}$=%.2f'%metrics.r2_score(ineter_plt,y_temp), 
            color='#F18B92',fontsize='10')

fig.savefig("./Image/Interact_Concentration_TEM_curve.jpg",dpi=600,bbox_inches='tight')

# %% 3. 'Concentration (mg/L)' and 'Hydrodynamic diameter (nm)'   SHAP interaction values
feature1 = 'Hydrodynamic diameter (nm)'
feature2 = 'Concentration (mg/L)'

fig, ax = plt.subplots(figsize=(4, 2))
shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,display_features=X_raw)

# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

fig.savefig("./Image/Interact_Concentration_Hydro.jpg",dpi=600,bbox_inches='tight')

# %% 4. 'Concentration (mg/L)' and 'BET surface area (m2/g)'   SHAP interaction values
feature1 = 'BET surface area (m2/g)'
feature2 = 'Concentration (mg/L)'

fig, ax = plt.subplots(figsize=(4, 2))
shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,display_features=X_raw)

# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

fig.savefig("./Image/Interact_Concentration_BET.jpg",dpi=600,bbox_inches='tight')

# %% 5. 'Concentration (mg/L)' and 'Composition'   SHAP interaction values
feature1 = 'Composition'
feature2 = 'Concentration (mg/L)'

fig, ax = plt.subplots(figsize=(4, 2))
shap.dependence_plot((feature1, feature2),shap_interaction_values,X,ax=ax,show=False,display_features=X_raw)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')   
# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap('Set2')#

fig.savefig("./Image/Interact_Concentration_Composition.jpg",dpi=600,bbox_inches='tight')

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
fig.savefig("./Image/Local_PDP_Concentration.jpg",dpi=600,bbox_inches='tight') 

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
fig.savefig("./Image/Local_PDP_%s.jpg"%feature,dpi=600,bbox_inches='tight') 


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
fig.savefig("./Image/Local_PDP_%s.jpg"%feature,dpi=600,bbox_inches='tight') 

# %% 4. BET surface area (m2/g) PDP one feature
feature = "BET surface area (m2/g)"

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
fig.savefig("./Image/Local_PDP_BET.jpg",dpi=600,bbox_inches='tight') 

# %% 5.Composition PDP one feature
feature = "Composition"

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
axes['pdp_ax'].set_xticks(range(0,8,1))
axes['pdp_ax'].set_xticklabels(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])),
                                rotation=45,ha='right')
fig.savefig("./Image/Local_PDP_%s.jpg"%feature,dpi=600,bbox_inches='tight') 

# %% 6. Hydrodynamic diameter (nm) PDP one feature
feature = "Hydrodynamic diameter (nm)"

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
fig.savefig("./Image/Local_PDP_%s.jpg"%feature,dpi=600,bbox_inches='tight') 

# %% 7. Morphology PDP one feature
feature = "Morphology"

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

axes['pdp_ax'].set_xticks(range(0,2,1))
axes['pdp_ax'].set_xticklabels(list(le_morphology.inverse_transform([0,1])),
                                )
fig.savefig("./Image/Local_PDP_%s.jpg"%feature,dpi=600,bbox_inches='tight') 


# %% 1. 'Concentration (mg/L)' and 'TEM size (nm)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=['Concentration (mg/L)', 'TEM size (nm)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, ['Concentration (mg/L)', 'TEM size (nm)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5)
)

fig.savefig("./Image/Combined_PDP_Concentration_TEM.jpg",dpi=600,bbox_inches='tight') 

# %% 2. 'Concentration (mg/L)' and 'Zeta potential (mV)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=['Concentration (mg/L)', 'Zeta potential (mV)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, ['Concentration (mg/L)', 'Zeta potential (mV)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5))

fig.savefig("./Image/Combined_PDP_Concentration_Zeta.jpg",dpi=600,bbox_inches='tight') 

# %% 3. 'Concentration (mg/L)' and 'Hydrodynamic diameter (nm)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=['Concentration (mg/L)', 'Hydrodynamic diameter (nm)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, ['Concentration (mg/L)', 'Hydrodynamic diameter (nm)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5))
    
fig.savefig("./Image/Combined_PDP_Concentration_HydroDiameter.jpg",dpi=600,bbox_inches='tight') 

# %% 4. 'TEM size (nm)' and 'Zeta potential (mV)'   PDP two feature
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
                    
fig.savefig("./Image/Combined_PDP_TEM_Zeta.jpg",dpi=600,bbox_inches='tight') 

# %% 5. 'TEM size (nm)' and 'BET surface area (m2/g)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=[ 'TEM size (nm)','BET surface area (m2/g)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, [ 'TEM size (nm)','BET surface area (m2/g)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5)
)
axes['pdp_inter_ax'].set_xticklabels(axes['pdp_inter_ax'].get_xticklabels(),
                    rotation=45,horizontalalignment='center',)

fig.savefig("./Image/Combined_PDP_TEM_BET.jpg",dpi=600,bbox_inches='tight') 

# %% 6. 'Zeta potential (mV)' and 'BET surface area (m2/g)'   PDP two feature
inter_rf = pdp.pdp_interact(
    model=model, dataset=X, model_features=X.columns, 
    features=[ 'Zeta potential (mV)','BET surface area (m2/g)'])

fig, axes = pdp.pdp_interact_plot(
    inter_rf, [ 'Zeta potential (mV)','BET surface area (m2/g)'], 
    x_quantile=True, plot_type='contour', plot_pdp=False,
    figsize=(4, 5.5)
)

axes['pdp_inter_ax'].set_xticklabels(axes['pdp_inter_ax'].get_xticklabels(),
                    rotation=45,horizontalalignment='center',)

fig.savefig("./Image/Combined_PDP_Zeta_BET.jpg",dpi=600,bbox_inches='tight')


# %% LIME
categorical_names={}
categorical_names[0]=le_composition.classes_
categorical_names[2]=le_morphology.classes_
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_cv.values), 
                                                feature_names=X_cv.columns,
                                                categorical_features=[0,2],
                                                categorical_names = categorical_names,
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

fig, ax= plt.subplots(figsize = (6,4.3))
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


fig.savefig("./Image/Lime_Local_ZnO_30.jpg",dpi=600,bbox_inches='tight')

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
fig.savefig("./Image/Model_predict_proba_Local_ZnO_30.jpg",dpi=600,bbox_inches='tight')



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

fig, ax= plt.subplots(figsize = (6,4.3))
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


fig.savefig("./Image/Lime_Local_106.jpg",dpi=600,bbox_inches='tight')

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
fig.savefig("./Image/Model_predict_proba_Local_106.jpg",dpi=600,bbox_inches='tight')


# %%
