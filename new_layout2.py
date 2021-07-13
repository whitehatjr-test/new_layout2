import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO, StringIO

st.set_page_config(layout="wide")

# Loading the dataset.
iris_df = pd.read_csv("https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache()
# Create a function 'prediction()' which accepts SepalLength, SepalWidth, PetalLength, PetalWidth as input and returns species name.
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth):
  species = svc_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species= int(species)
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"



uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data)



with st.beta_expander("Watch youtube video"):
    st.video('https://www.youtube.com/embed/xn92pT6-fRE') 

