import pandas as pd
import os

def load_and_merge_data(path='data/'):
    "This loads the seperate data files and merge into a multimodel dataset"

    print("Loading datasets...")

    orders = pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv'))
    reviews = pd.read_csv(os.path.join(path, 'olist_order_reviews_dataset.csv'))
    items = pd.read_csv(os.path.join(path, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(path, 'olist_products_dataset.csv'))
    payments = pd.read_csv(os.path.join(path, 'olist_order_payments_dataset.csv'))

    print("Merging datasets...")
    df = orders.merge(reviews, on='order_id',how='left')
    df = df.merge(items, on='order_id',how='left')
    df = df.merge(products, on='product_id', how='left')
    df = df.merge(payments, on='order_id', how='left')

    df_clean = df.dropna(subset=['review_comment_message', 'price'])

    final_df = df_clean[[
        'order_id', 
        'customer_id', 
        'order_status', 
        'review_comment_message',  # NLP Input
        'review_score',            # Target (Label)
        'price',                   # Tabular Input
        'freight_value',           # Tabular Input
        'product_category_name'    # Categorical Input
    ]]

    print(f"Data successfully merged. Final shape: {final_df.shape}")
    return final_df

if __name__=="__main__":
    df = load_and_merge_data()
    df.head(500).to_csv('data/sample_multimodal_data.csv', index=False)
    print("New Data Saved")