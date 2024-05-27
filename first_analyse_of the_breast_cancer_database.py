import pandas as pd
clinical_data_frame = pd.read_excel('/content/breast_DNA_clinical data.xlsx')
print(df[df['SAMPLE_ID'] == 'MB-0897'])
print(clinical_data_frame)
df2 = pd.DataFrame(metabric_genmutacio_data_frame)
merged_df = pd.merge(df, df2, on='SAMPLE_ID', how='inner'
merged_df = merged_df.drop(['Unnamed: 4', 'Unnamed: 3', 'Unnamed: 2', 'Unnamed: 1', 'Unnamed: 0'], axis=1)
import matplotlib.pyplot as plt
plt.hist(merged_df['Overall survival (month)'], bins= 100)
plt.title('Overall survival (month)')
plt.xlabel('Overall survival (month)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show
merged_df.isnull().sum()
plt.scatter(merged_df['Lymph node status (1=positive)'], merged_df['RFS_time (months)']) #korreláció csak a lymph node status és az rfs time között# kíváncsiság, gyakorlás
plt.title('Scatter Plot of Lymph node status vs RFS_time (months)')
plt.xlabel('Lymph node status')
plt.ylabel('Tumor Size')
plt.show()
correlation = merged_df.corr()
overallsurvival_correlation = correlation['Overall survival (month)'].sort_values(ascending=False)

plt.figure(figsize=(16, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation of Features with Overall survival (month)')
plt.show()
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
mean_value_menopausal = merged_df['menopausal_status_post_2_pre0_peri1'].mean()
merged_df['menopausal_status_post_2_pre0_peri1'].fillna(mean_value_menopausal, inplace=True)
mean_value_overall = merged_df['Overall survival (month)'].mean()
merged_df=merged_df.dropna(subset=['Overall survival (month)'])
display(merged_df)
#merged_df['Overall survival (month)'].fillna(mean_value_overall, inplace=True)
mean_value_grade = merged_df['Grade'].mean()
merged_df['Grade'].fillna(mean_value_grade, inplace=True)
mean_value_ts = merged_df['Tumor_size(mm)'].mean()
merged_df['Tumor_size(mm)'].fillna(mean_value_ts, inplace=True)
mean_value_cht = merged_df['Chemotherapy treated'].mean()
merged_df['Chemotherapy treated'].fillna(mean_value_cht, inplace=True)
merged_df.isnull().sum()
def func(overall_survival_month):
    if overall_survival_month <= 3:
        return 1
    if overall_survival_month <= 6:
        return 2
    if overall_survival_month <= 9:
        return 3
    if overall_survival_month <= 12:
        return 4
    if overall_survival_month <= 15:
        return 5
    if overall_survival_month <= 18:
        return 6
    if overall_survival_month <= 21:
        return 7
    if overall_survival_month <=24:
        return 8
    if overall_survival_month <=36:
        return 9
    if overall_survival_month <=48:
        return 10
    if overall_survival_month <=60:
        return 11
    if overall_survival_month <=72:
        return 12
    if overall_survival_month <=84:
        return 13
    if overall_survival_month <=96:
        return 14
    if overall_survival_month <=108:
        return 15
    if overall_survival_month <=120:
        return 16
    if overall_survival_month <=132:
        return 17
    if overall_survival_month <=144:
        return 18
    if overall_survival_month <=156:
        return 19
    if overall_survival_month <=168:
        return 20
    if overall_survival_month <=180:
        return 21
    if overall_survival_month <=192:
        return 22
    if overall_survival_month <=204:
        return 23
    if overall_survival_month <=216:
        return 24
    if overall_survival_month <=228:
        return 25
    if overall_survival_month <=240:
        return 26
    if overall_survival_month >240:
        return 27

X = merged_df[['TMB (high =1)','Tumor_size(mm)','menopausal_status_post_2_pre0_peri1','RFS_status','Age','Chemotherapy_treated', 'Hormone therapy treated','ER status (1=positive)','HER2 status (1=positive)']] #az X azon változók amik hathatnak a túlélésre#
y = merged_df['Overall_survival_int_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf = DecisionTreeRegressor(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = False)
print(f"Négyzetes hiba (MSE): {mse}")
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X.columns)
plt.show()
!pip install shap
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
X = merged_df[['TMB (high =1)','Tumor_size(mm)','menopausal_status_post_2_pre0_peri1', 'RFS_status','Age','Chemotherapy_treated', 'Hormone therapy treated','ER status (1=positive)','HER2 status (1=positive)']]
y = merged_df['Overall_survival_int_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = False)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
reg.coef_
# Calculate SHAP values
explainer = shap.Explainer(reg, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=['TMB (high =1)','Tumor_size(mm)','menopausal_status_post_2_pre0_peri1', 'RFS_status','Age','Chemotherapy_treated', 'Hormone therapy treated','ER status (1=positive)','HER2 status (1=positive)'])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
X = merged_df[['mutációszám','menopausal_status_post_2_pre0_peri1','Tumor_size(mm)', 'Grade', 'RFS_status','Age','Chemotherapy_treated', 'Hormone therapy treated','HER2 status (1=positive)','ER status (1=positive)',]] # Bemeneti változók
y = merged_df['Overall_survival_int_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
clf = SVC(kernel = 'rbf', random_state = 42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score)
print(recall_score)
from sklearn.tree import DecisionTreeClassifier
X = merged_df[['mutációszám','menopausal_status_post_2_pre0_peri1','Tumor_size(mm)', 'Grade', 'RFS_status','Age','Chemotherapy_treated', 'Hormone therapy treated','HER2 status (1=positive)','ER status (1=positive)',]] # Bemeneti változók
y = merged_df['Overall_survival_int_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted'
print(f"F1 Score: {f1}")

