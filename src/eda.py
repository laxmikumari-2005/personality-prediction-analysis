
#PIECHART
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("data/personality_dataset.csv")

df['Personality'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title("Personality Distribution")
plt.show()

#CORRELATION HEATMAP
import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()

#BOXPLOT
sns.boxplot(
    x='Personality',
    y='Time_spent_Alone',
    data=df
)

plt.show()