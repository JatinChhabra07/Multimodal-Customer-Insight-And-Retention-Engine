import pandas as pd
import numpy as np 
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import json

DATA_PATH="data/sample_multimodal_data.csv"
OUTPUT_PATH="data/processed_tensors.npz"

def process_features():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Run data_preprocessing Firstly")
    
    print("Loading Merged Dataset")
    df = pd.read_csv(DATA_PATH)
    print(f"Data Shape {df.shape}")

    # data cleaning

    df = df.drop_duplicates()
    print(f"shape after dropping duplicates: {df.shape}")

    df = df.dropna(subset=['product_category_name'])
    print(f"Shape after handling missing values: {df.shape}")

    # Encoding
    print("Encoding Product Categories")
    category_encoded = LabelEncoder()
    df['category_encoded']=category_encoded.fit_transform(df['product_category_name'])

    num_categories =len(category_encoded.classes_)
    print(f"Unique Product Categories {num_categories}")

    # Scaling
    print("Scaling the Data")
    df['price_log'] = np.log1p(df['price'])
    df['freight_log'] = np.log1p(df['freight_value'])
    df['payment_log'] = np.log1p(df['payment_value'])

    scaler = MinMaxScaler()
    numerical_features = df[['price_log', 'freight_log', 'payment_log']].values
    numerical_data = scaler.fit_transform(numerical_features)

    # Performing NLP
    print("Generating Embeddings")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    texts = df['review_comment_message'].astype(str).tolist()
    text_embeddings = model.encode(texts, show_progress_bar=True)

    # saving
    y = np.where(df['review_score']<=2,1,0)

    category_ids = df['category_encoded'].values

    print(f"Saving TO {OUTPUT_PATH}")

    np.savez_compressed(
        OUTPUT_PATH,
        text_embeddings=text_embeddings,
        numerical_data=numerical_data,
        category_ids=category_ids,
        lables=y
    )

    metadata={
        "num_categories": int(num_categories),
        "num_numerical_features": numerical_data.shape[1],
        "text_embedding_dim": text_embeddings.shape[1]
    }

    with open("data/metadata.json", "w") as f:
        json.dump(metadata,f)
        print("Feature Engineering Complete. Metadata saved.")

if __name__ == "__main__":
    process_features()
