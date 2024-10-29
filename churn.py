import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the data
# Veriyi yükleme
file_path = 'archive/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(file_path)

# 1. Convert TotalCharges column to numeric and handle missing values
# 1. TotalCharges sütununu sayısal hale getirin ve eksik değerleri işleyin
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

# 2. Convert categorical variables to numerical values
# 2. Kategorik verileri sayısal değerlere çevirin
# Drop the customerID column
# customerID sütununu çıkarın
data.drop(columns=['customerID'], inplace=True)

# Convert Churn column to binary values
# Churn sütununu ikili değerlere dönüştürün
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Apply One-Hot Encoding to other categorical columns
# Diğer kategorik sütunlara One-Hot Encoding uygulayın
data = pd.get_dummies(data, drop_first=True)

# 3. Scale the data
# 3. Veriyi Ölçeklendirme
scaler = StandardScaler()
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# 4. Churn Rate Analysis
# 4. Churn Oranı Analizi
plt.figure(figsize=(12, 6))
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Calculate the churn rate
# Churn oranını hesaplayın
churn_rate = data['Churn'].mean() * 100
print("Churn Rate (%):", churn_rate)

# 5. Correlation Analysis with Key Features
# 5. Önemli Sütunlar ile Churn Korelasyon Analizi
# Select key features for correlation analysis
# Korelasyonu analiz etmek için önemli özellikleri seçin
features = ['MonthlyCharges', 'tenure', 'TotalCharges'] + [col for col in data.columns if 'Contract' in col or 'InternetService' in col]
correlation_data = data[features + ['Churn']].corr()

# Visualize the correlation analysis using a heatmap
# Korelasyon analizini heatmap ile görselleştirin
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data[['Churn']].sort_values(by='Churn', ascending=False), annot=True, cmap='coolwarm')
plt.title('Correlation of Key Features with Churn')
plt.show()

# 6. Model Training and Evaluation
# 6. Model Eğitimi ve Değerlendirilmesi
# Separate target variable and features
# Hedef değişken ve özellikleri ayırın
X = data.drop(columns=['Churn'])
y = data['Churn']

# Split the data into training and testing sets
# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
# Lojistik Regresyon modelini başlat ve eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the test set
# Modeli test setinde değerlendirin
y_pred = model.predict(X_test)

# Calculate model evaluation metrics
# Model değerlendirme metriklerini hesaplayın
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Display the results
# Sonuçları çıktı olarak gösterin
print("Logistic Regression Model Metrics:")
print("Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nROC AUC Score:\n", roc_auc)

# Visualize the ROC curve
# ROC eğrisini görselleştirin
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='Logistic Regression (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Initialize and train the Random Forest model
# Random Forest modelini başlat ve eğit
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the Random Forest model on the test set
# Random Forest modeli test setinde değerlendirin
y_pred_rf = rf_model.predict(X_test)

# Calculate Random Forest model evaluation metrics
# Random Forest model değerlendirme metriklerini hesaplayın
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Display Random Forest results
# Random Forest sonuçlarını çıktı olarak gösterin
print("\nRandom Forest Model Metrics:")
print("Model Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("\nClassification Report:\n", classification_rep_rf)
print("\nConfusion Matrix:\n", conf_matrix_rf)
print("\nROC AUC Score:\n", roc_auc_rf)

# Visualize the ROC curve (Random Forest)
# ROC eğrisini görselleştirin (Random Forest)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, color='green', label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend()
plt.show()

# Initialize and train the Gradient Boosting Classifier model
# Gradient Boosting Classifier modelini başlat ve eğit
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the Gradient Boosting model on the test set
# Gradient Boosting modeli test setinde değerlendirin
y_pred_gb = gb_model.predict(X_test)

# Calculate Gradient Boosting model evaluation metrics
# Gradient Boosting model değerlendirme metriklerini hesaplayın
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
classification_rep_gb = classification_report(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
roc_auc_gb = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])

# Display Gradient Boosting results
# Gradient Boosting sonuçlarını çıktı olarak gösterin
print("\nGradient Boosting Model Metrics:")
print("Model Accuracy:", accuracy_gb)
print("Precision:", precision_gb)
print("Recall:", recall_gb)
print("F1 Score:", f1_gb)
print("\nClassification Report:\n", classification_rep_gb)
print("\nConfusion Matrix:\n", conf_matrix_gb)
print("\nROC AUC Score:\n", roc_auc_gb)

# Visualize the ROC curve (Gradient Boosting)
# ROC eğrisini görselleştirin (Gradient Boosting)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr_gb, tpr_gb, color='red', label='Gradient Boosting (AUC = %0.2f)' % roc_auc_gb)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Gradient Boosting')
plt.legend()
plt.show()

# 7. Identify High-Risk Customers
# 7. Yüksek Riskli Müşterileri Belirleme
# Calculate churn probabilities
# Churn olasılıklarını hesaplayın
y_prob = model.predict_proba(X_test)[:, 1]

# Define customers with 70% or higher churn probability as high-risk
# %70 veya üzeri churn olasılığı olan müşterileri yüksek riskli olarak tanımlayın
threshold = 0.7
high_risk_customers = X_test[y_prob >= threshold].copy()
high_risk_customers['Churn_Probability'] = y_prob[y_prob >= threshold]

# Display high-risk customers
# Yüksek riskli müşterileri gösterin
print("\nHigh-Risk Customers:\n")
print(high_risk_customers)
