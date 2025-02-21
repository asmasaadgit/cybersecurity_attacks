import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
url = 'https://raw.githubusercontent.com/incribo-inc/cybersecurity_attacks/main/cybersecurity_attacks.csv'
data = pd.read_csv(url)

# Data Preprocessing
# Handle missing values (if any)
data.fillna('', inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
data['Protocol'] = label_encoder.fit_transform(data['Protocol'])
data['Packet Type'] = label_encoder.fit_transform(data['Packet Type'])
data['Traffic Type'] = label_encoder.fit_transform(data['Traffic Type'])
data['Malware Indicators'] = label_encoder.fit_transform(data['Malware Indicators'])
data['Alerts/Warnings'] = label_encoder.fit_transform(data['Alerts/Warnings'])
data['Action Taken'] = label_encoder.fit_transform(data['Action Taken'])
data['Severity Level'] = label_encoder.fit_transform(data['Severity Level'])
data['User Information'] = label_encoder.fit_transform(data['User Information'])
data['Device Information'] = label_encoder.fit_transform(data['Device Information'])
data['Network Segment'] = label_encoder.fit_transform(data['Network Segment'])
data['Geo-location Data'] = label_encoder.fit_transform(data['Geo-location Data'])
data['Proxy Information'] = label_encoder.fit_transform(data['Proxy Information'])
data['Firewall Logs'] = label_encoder.fit_transform(data['Firewall Logs'])
data['IDS/IPS Alerts'] = label_encoder.fit_transform(data['IDS/IPS Alerts'])
data['Log Source'] = label_encoder.fit_transform(data['Log Source'])

# Combine text features into a single feature using TfidfVectorizer
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(data['Payload Data'])

# Combine all features
features = pd.concat([data[['Protocol', 'Packet Type', 'Traffic Type', 'Malware Indicators', 'Anomaly Scores', 'Alerts/Warnings', 'Action Taken', 'Severity Level', 'User Information', 'Device Information', 'Network Segment', 'Geo-location Data', 'Proxy Information', 'Firewall Logs', 'IDS/IPS Alerts', 'Log Source']], pd.DataFrame(text_features.toarray())], axis=1)

# Ensure all feature names are strings
features.columns = features.columns.astype(str)

# Target variable
target = data['Attack Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model selection and training
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
