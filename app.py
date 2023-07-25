import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def manipulate_df(df):
    '''
    function to transform our data to make it usable for our logistic regression model
    '''
    # Update sex column to numerical
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    # Fill the nan values in the age column
    df['Age'].fillna(value=df['Age'].mean(), inplace=True)
    # Create a first class column
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    # Create a second class column
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    # Create a second class column
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    # Select the desired features
    df = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass', 'Survived']]
    return df


train_df = pd.read_csv("train.csv")
# print(train_df)

manipulated_df = manipulate_df(train_df)
# print(manipulated_df)

features = train_df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass']]
survival = train_df['Survived']

# split the dataset into two parts, 70% of it for training and the remaining 30% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    features, survival, test_size=0.3)

# Scaling the training and testing data
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)
# print(test_features)


# Create and train the model
# create a model and train it with train_features
model = LogisticRegression()
model.fit(train_features, y_train)
# calculate training and testing scores
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
# predict the model on test_features
y_predict = model.predict(test_features)

# print("Training Score: ",train_score)
# print("Testing Score: ",test_score)


# Calculating Confusion Matrix
confusion = confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

st.title("Would you have survived the Titanic Disaster?")
st.subheader(
    "This model will predict if a passenger would survive the Titanic Disaster or not")
st.dataframe(train_df.head(5))
# An alternative to st.table() is st.dataframe().
# st.table(train_df.head(5))

# display the train and test scores
st.subheader("Train Set Score: {}".format(round(train_score, 3)))
st.subheader("Test Set Score: {}".format(round(test_score, 3)))

# Showing the confusion matrix for model
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(['False Negative', 'True Negative', 'True Positive',
        'False Positive'], [FN, TN, TP, FP])
ax.set_xlabel('Confusion matrix')
st.pyplot(fig)


name = st.text_input("Name of Passenger ")
sex = st.selectbox("Sex", options=['Male', 'Female'])
age = st.slider("Age", 1, 100, 1)
p_class = st.selectbox("Passenger Class", options=[
                       'First Class', 'Second Class', 'Third Class'])

sex = 0 if sex == 'Male' else 1
f_class, s_class, t_class = 0, 0, 0
if p_class == 'First Class':
    f_class = 1
elif p_class == 'Second Class':
    s_class = 1
else:
    t_class = 1

input_data = scaler.transform([[sex, age, f_class, s_class, t_class]])
prediction = model.predict(input_data)
predict_probability = model.predict_proba(input_data)

if name != '':
    if prediction[0] == 1:
        st.subheader('Passenger {} would have survived with a probability of {}%'.format(
            name, round(predict_probability[0][1]*100, 3)))
    else:
        st.subheader('Passenger {} would not have survived with a probability of {}%'.format(
            name, round(predict_probability[0][0]*100, 3)))
