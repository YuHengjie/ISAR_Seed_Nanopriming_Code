# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from imodels import RuleFitClassifier

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
X_cv

# %%
feature_names = X_cv.columns
feature_names

# %%
rulefit = RuleFitClassifier(n_estimators=100, tree_size=4, sample_fract='default', max_rules=40, 
                           memory_par=0.01, tree_generator=RandomForestRegressor(), lin_trim_quantile=0.025, 
                           lin_standardise=True, exp_rand_tree_size=True, include_linear=True, 
                           alpha=None, cv=True, random_state=42)

rulefit.fit(X_cv, y_cv, feature_names=feature_names)
y_pred = rulefit.predict(X_test)
y_cv_pred = rulefit.predict(X_cv)

# %%
y_pred_cv = rulefit.predict(X_cv)
y_proba_cv = rulefit.predict_proba(X_cv)[:, 1]

print('Train AUC: %.2f'%metrics.roc_auc_score(y_cv,y_proba_cv))
print('Train F1: %.2f'%metrics.f1_score(y_cv,y_pred_cv))
print('Train Accuracy: %.2f'%metrics.accuracy_score(y_cv,y_pred_cv))
#print('Train MCC: %.2f'%metrics.matthews_corrcoef(y_cv,y_pred))

y_pred = rulefit.predict(X_test)
y_proba = rulefit.predict_proba(X_test)[:, 1]

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

fig.savefig("./Image/Test_ROC_RuleFit.jpg",dpi=600,bbox_inches='tight')

# %%
rules = rulefit.get_rules()
rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
rules.to_excel("rules_RDW.xlsx")
rules

# %%
def find_mk(input_vars:list, rule:str):

    var_count = 0
    for var in input_vars:
        if var in rule:
            var_count += 1
    return(var_count)

def get_feature_importance(feature_set: list, rule_set: pd.DataFrame, scaled = False):

    feature_imp = list()
    
    rule_feature_count = rule_set.rule.apply(lambda x: find_mk(feature_set, x))

    for feature in feature_set:
        
        # find subset of rules that apply to a feature
        feature_rk = rule_set.rule.apply(lambda x: feature in x)
        
        # find importance of linear features
        linear_imp = rule_set[(rule_set.type=='linear')&(rule_set.rule==feature)].importance.values
        
        # find the importance of rules that contain feature
        rule_imp = rule_set.importance[(rule_set.type=='rule')&feature_rk]
        
        # find the number of features in each rule that contain feature
        m_k = rule_feature_count[(rule_set.type=='rule')&feature_rk]
        
        # sum the linear and rule importances, divided by m_k
        if len(linear_imp)==0:
            linear_imp = 0
        # sum the linear and rule importances, divided by m_k
        if len(rule_imp) == 0:
            feature_imp.append(float(linear_imp))
        else:
            feature_imp.append(float(linear_imp + (rule_imp/m_k).sum()))
        
    if scaled:
        feature_imp = 100*(feature_imp/np.array(feature_imp).max())
    
    return(feature_imp)

# %%
feature_importances = get_feature_importance(X.columns, rules, scaled=False)
importance_df = pd.DataFrame(feature_importances, index = X.columns, columns = ['importance']).sort_values(by='importance',ascending=False)
importance_df

# %%
fig = plt.figure(figsize=(6,2.5))
plt.style.use('classic')
plt.margins(0.02)

plt.barh(importance_df.index[::-1], importance_df.importance[::-1],
          align='center', color="#1E90FF")

plt.title('RuleFit feature importance',fontsize=16)
plt.xlabel('Importance value',fontsize=16)
fig.savefig("./Image/RuleFit_importance.jpg",dpi=600,bbox_inches='tight')

# %%
