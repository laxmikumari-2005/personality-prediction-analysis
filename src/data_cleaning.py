
import pandas as pd
df=pd.read_csv("data/personality_dataset.csv")

print(df.isnull().sum())
df.dropna(inplace=True)
print(df.shape)

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

df['Stage_fear']=encoder.fit_transform(df['Stage_fear'])
df['Drained_after_socializing']=encoder.fit_transform(df['Drained_after_socializing'])
df['Personality']=encoder.fit_transform(df['Personality'])


df.to_csv("data/cleaned_personality_dataset.csv",
          index=False)