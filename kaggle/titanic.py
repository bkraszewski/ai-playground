# what i want to do

# load train data set
# create local test and validation set

# implement simple model using neural net to predict survival
# verify on validation and test set (do not use test set yet)

# create function to save prediction in the kaggle format

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import re

use_neural_net = True
should_show_plot = True


def prepare_data(df):

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # calculate family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    # age group
    df.loc[df["Age"] <= 15, "Age"] = 0
    df.loc[(df["Age"] > 15) & (df["Age"] <= 25), "Age"] = 1
    df.loc[(df["Age"] > 25) & (df["Age"] <= 35), "Age"] = 2
    df.loc[(df["Age"] > 35) & (df["Age"] <= 45), "Age"] = 3
    df.loc[(df["Age"] > 45) & (df["Age"] <= 60), "Age"] = 4
    df.loc[df["Age"] > 60, "Age"] = 5

    # title
    df["Title"] = df["Name"].apply(get_title)
    df["Title"] = df["Title"].replace(
        ["Capt.", "Dr.", "Major.", "Rev.", "Col."], "Officer."
    )
    df["Title"] = df["Title"].replace(
        ["Lady.", "Countess.", "Don.", "Sir.", "Jonkheer.", "Dona."], "Royal."
    )
    df["Title"] = df["Title"].replace(["Mlle.", "Ms."], "Miss.")
    df["Title"] = df["Title"].replace(["Mme."], "Mrs.")

    # cabin type
    df["Cabin"] = df["Cabin"].fillna("Missing")
    df["Cabin"] = df["Cabin"].str[0]

    df = df.drop(["Name", "Ticket"], axis=1)
    all_dummies = pd.get_dummies(
        df, drop_first=True, columns=["Embarked", "Title", "Cabin", "Sex"]
    )

    all_dummies = all_dummies.astype("float32")
    print(all_dummies.head())
    return all_dummies


def get_title(name):
    title_search = re.search(" ([A-Za-z]+\.)", name)

    if title_search:
        return title_search.group(1)
    return ""


def show_plot(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def neural_net(train_dataset, should_show_plot, val_dataset):
    model = Sequential(
        [
            Dense(
                units=X_train.shape[1],
                input_shape=(X_train.shape[1],),
                activation="relu",
            ),
            Dense(units=1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset)

    if should_show_plot:
        show_plot(history)

    predictions = model.predict(test_dataset)
    predictions = (predictions > 0.5).astype(int)
    predictions = predictions.ravel()
    return predictions


train_df = prepare_data(pd.read_csv("train.csv"))

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))

train_dataset = train_dataset.shuffle(len(X_train)).batch(32)
val_dataset = val_dataset.batch(32)


# test part
test_df = prepare_data(pd.read_csv("test.csv"))

missing_cols = set(X_train.columns) - set(test_df.columns)
for c in missing_cols:
    test_df[c] = 0

test_df = test_df[X_train.columns]

test_dataset = tf.data.Dataset.from_tensor_slices(test_df.values).batch(32)

if use_neural_net:
    predictions = neural_net(train_dataset, should_show_plot, val_dataset)
    ids = pd.read_csv("test.csv")["PassengerId"]

    output = pd.DataFrame({"PassengerId": ids, "Survived": predictions.ravel()})
    # print(output.head())
    output.to_csv("submission.csv", index=False)
