import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

os.makedirs("analytics/plots", exist_ok=True)

df = sns.load_dataset("titanic")

df.to_csv(
    "analytics/titanic.csv",
    index=False
)

print("Titanic dataset loaded successfully.")

print("\nShape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe())

print("\nMissing percentage:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print((missing / len(df) * 100).round(2))

df = df.dropna(subset=["embarked", "embark_town"])
df["age"] = df["age"].fillna(df["age"].median())
df = df.drop(columns=["deck"])

print("\nAfter missing-value handling:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nMissing-value decisions:")
print("embarked: 0.22% → dropped rows")
print("embark_town: 0.22% → dropped rows")
print("age: 19.87% → median imputation")
print("deck: 77.22% → column dropped due to very high missingness")


plt.figure()
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.savefig("analytics/plots/age_distribution.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.boxplot(x=df["age"])
plt.title("Age Boxplot")
plt.savefig("analytics/plots/age_boxplot.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.histplot(df["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.savefig("analytics/plots/fare_distribution.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.boxplot(x=df["fare"])
plt.title("Fare Boxplot")
plt.savefig("analytics/plots/fare_boxplot.png", bbox_inches="tight")
plt.show()
plt.close()


def iqr_outliers(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((df[column] < lower) | (df[column] > upper)).sum()


print("\nIQR Outliers:")
print("Age:", iqr_outliers("age"))
print("Fare:", iqr_outliers("fare"))

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\nFare Statistics:")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

skewness = df["fare"].skew()

print("\nFare Skewness:")
print(skewness)

if fare_mean > fare_median > fare_mode:
    print("Fare is right-skewed.")
elif fare_mean < fare_median < fare_mode:
    print("Fare is left-skewed.")
else:
    print("Fare does not follow a simple mean-median-mode ordering.")


print("\nSurvival by Sex:")
print(df.groupby("sex", observed=True)["survived"].mean())

print("\nSurvival by Passenger Class:")
print(df.groupby("pclass")["survived"].mean())

print("\nSurvival by Sex and Passenger Class:")
print(df.groupby(["sex", "pclass"], observed=True)["survived"].mean())


plt.figure()
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.savefig("analytics/plots/survival_by_sex.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.savefig("analytics/plots/survival_by_class.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival Rate")
plt.savefig("analytics/plots/survival_by_sex_class.png", bbox_inches="tight")
plt.show()
plt.close()


corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = df[corr_cols].corr()

print("\nCorrelation Matrix:")
print(corr.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.savefig("analytics/plots/correlation_heatmap.png", bbox_inches="tight")
plt.show()
plt.close()

upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

pairs = upper.abs().stack().sort_values(ascending=False)

print("\nTwo Strongest Absolute Correlations:")
print(pairs.head(2))


plt.figure()
sns.boxplot(data=df, x="pclass", y="fare", hue="sex")
plt.title("Fare by Passenger Class and Sex")
plt.savefig("analytics/plots/fare_class_sex.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.scatterplot(data=df, x="age", y="fare", hue="survived")
plt.title("Age vs Fare by Survival")
plt.savefig("analytics/plots/age_fare_survival.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.barplot(data=df, x="pclass", y="fare", hue="survived")
plt.title("Average Fare by Class and Survival")
plt.ylabel("Average Fare")
plt.savefig("analytics/plots/average_fare_class_survival.png", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.countplot(data=df, x="pclass", hue="survived")
plt.title("Passenger Class Distribution by Survival")
plt.savefig("analytics/plots/class_survival_distribution.png", bbox_inches="tight")
plt.show()
plt.close()


scaler = StandardScaler()

df[["age", "fare"]] = scaler.fit_transform(
    df[["age", "fare"]]
)

print("\nStandardized Age and Fare:")
print(df[["age", "fare"]].describe().round(3))