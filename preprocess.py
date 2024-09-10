import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load datasets
df = pd.read_csv('articles.csv')

# Check for missing values
print(df.isnull().sum())

# Function for cleaning text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)
    return text

# Apply text cleaning
df['cleaned_article'] = df['full_article'].apply(clean_text)

# Handle missing values if necessary
df = df.dropna(subset=['cleaned_article', 'article_type'])

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df['cleaned_article'],
    df['article_type'],
    test_size=0.2,
    random_state=42,
    stratify=df['article_type']
)
