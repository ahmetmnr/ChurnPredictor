# Customer Churn Prediction Model

## Overview
This project is a customer churn prediction model for a telecommunications company. The term "churn" refers to when a customer cancels their subscription or terminates their services. The goal of this analysis is to predict customer churn using machine learning models, allowing the company to take preventive actions to reduce customer turnover.

### Key Steps
1. **Data Loading and Preprocessing**
   - The data is loaded from a CSV file containing customer details.
   - Preprocessing steps include handling missing values in the data, converting categorical variables to numerical variables using One-Hot Encoding, and scaling numerical features to improve model performance.

2. **Exploratory Data Analysis (EDA)**
   - The churn rate is calculated, which shows the percentage of customers who have left the service.
   - Correlation analysis is performed between the target variable (Churn) and other features to understand key factors contributing to customer churn.

3. **Model Building and Training**
   - Three different machine learning models are trained to predict customer churn:
     - **Logistic Regression**: Used for binary classification to predict churn.
     - **Random Forest and Gradient Boosting**: Used as ensemble learning methods to create stronger models through multiple decision trees.
   - The dataset is split into training and test sets to evaluate model performance.

4. **Model Evaluation**
   - Models are evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC AUC score to determine their effectiveness.
   - ROC curves are plotted for each model to visualize their classification performance.

5. **Identifying High-Risk Customers**
   - The customers with a churn probability of 70% or higher are identified as high-risk customers.
   - This helps the company target these customers with personalized offers to prevent churn.

### Objective
The primary objective of this project is to predict which customers are likely to churn, allowing the company to take strategic actions to improve customer retention and satisfaction.

By using this model, telecommunications companies can enhance their ability to predict customer behavior and proactively reduce churn, ultimately leading to better customer loyalty and improved revenues.

---

# Müşteri Kaybı Tahmin Modeli

## Genel Bakış
Bu proje, bir telekomünikasyon şirketi için müşteri kaybı (churn) tahmin modeli geliştirmektedir. "Churn" terimi, bir müşterinin aboneliğini iptal etmesi veya hizmetlerini sonlandırması anlamına gelir. Bu analiz ile müşteri kaybını makine öğrenimi modelleri kullanarak tahmin etmek ve bu sayede şirketin müşteri kaybını azaltmak için önleyici tedbirler almasına yardımcı olunması hedeflenmektedir.

### Ana Adımlar
1. **Veri Yükleme ve Ön İşleme**
   - Veri, müşteri bilgilerini içeren bir CSV dosyasından yüklenir.
   - Veri ön işleme adımları, eksik değerlerin işlenmesi, kategorik değişkenlerin One-Hot Encoding yöntemiyle sayısal verilere dönüştürülmesi ve model performansını artırmak için sayısal özelliklerin ölçeklendirilmesini içerir.

2. **Keşifsel Veri Analizi (EDA)**
   - Müşteri kaybı oranı hesaplanır ve hizmeti bırakan müşterilerin yüzdesi gösterilir.
   - Hedef değişken (Churn) ile diğer özellikler arasındaki ilişkiyi anlamak amacıyla korelasyon analizi yapılır.

3. **Model Kurma ve Eğitimi**
   - Müşteri kaybını tahmin etmek için üç farklı makine öğrenimi modeli eğitilir:
     - **Lojistik Regresyon**: Müşteri kaybını ikili sınıflandırma problemi olarak tahmin etmek için kullanılır.
     - **Random Forest ve Gradient Boosting**: Birden fazla karar ağacını kullanarak daha güçlü ve genellenebilir modeller oluşturan toplu öğrenme yöntemleridir.
   - Veri seti eğitim ve test olarak ayrılır, böylece modelin performansı değerlendirilebilir.

4. **Model Değerlendirme**
   - Modeller, doğruluk, kesinlik, geri çağırma, F1 skoru ve ROC AUC skoru gibi metrikler kullanılarak değerlendirilir.
   - Her model için ROC eğrileri çizilerek sınıflandırma performansı görsel olarak anlaşılmaya çalışılır.

5. **Yüksek Riskli Müşterilerin Belirlenmesi**
   - %70 ve üzeri churn olasılığına sahip müşteriler yüksek riskli müşteri olarak tanımlanır.
   - Bu sayede şirket, bu müşterilere özel teklifler sunarak müşteri kaybını önlemeye çalışabilir.

### Amaç
Bu projenin birincil amacı, müşteri kaybetme olasılığı yüksek olan müşterileri tahmin etmek ve bu sayede şirketin müşteri memnuniyetini ve bağlılığını artırmak için stratejik adımlar atmasına yardımcı olmaktır.

Bu model sayesinde, telekomünikasyon şirketleri müşteri davranışlarını daha iyi tahmin edebilir ve proaktif önlemler alarak müşteri kaybını azaltabilir, bu da daha iyi müşteri sadakati ve artan gelir anlamına gelir.

