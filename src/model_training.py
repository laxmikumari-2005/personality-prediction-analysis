import pandas as pd
df=pd.read_csv("data/personality_dataset.csv")

x=df.drop("Personality",axis=1)
y=df["Personality"]

x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()

import joblib
joblib.dump(model,"models/personality_model.pkl")

df.to_csv("dashboard/dashbord_data.csv",index=False)
