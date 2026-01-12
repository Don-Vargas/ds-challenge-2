import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save_datasets(RAW_DATA,TRAIN_DATA,TEST_DATA, test_size=0.2, random_state=42):
    '''
    split original data
    '''
    df = pd.read_csv(RAW_DATA)
    
    # Split into train/test 0.2 (80/20)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save to respective folders
    train_df.to_csv(TRAIN_DATA, index=False)
    test_df.to_csv(TEST_DATA, index=False)
