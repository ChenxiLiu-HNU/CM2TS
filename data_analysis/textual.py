# %%
import pandas as pd
import re
from datetime import datetime

# %%
DATASET = 'Energy'
TIME_SERIES_FILE = f'../data/numerical/{DATASET}/{DATASET}.csv'
SEARCH_TEXT_FILE = f'../data/textual/{DATASET}/{DATASET}_search.csv'

df_text_search = pd.read_csv(SEARCH_TEXT_FILE, header=0, index_col=0)
df_text_search['start_date'] = pd.to_datetime(df_text_search['start_date'])
df_text_search['end_date'] = pd.to_datetime(df_text_search['end_date'])
df_ts = pd.read_csv(TIME_SERIES_FILE, header=0)
df_ts['start_date'] = pd.to_datetime(df_ts['start_date'])
df_ts['end_date'] = pd.to_datetime(df_ts['end_date'])
input_start_dates = df_ts['start_date']
input_end_dates = df_ts['end_date']

# %%
def fetch_search_text(df_text, end_date, text_col="fact", type_tag="#F#", text_len=2, latest_first=True):
    
    if type_tag == "#F#":
        text_info = "Available facts are as follows: "
    elif type_tag == "#In#":
        text_info = "Available insights are as follows: "
    elif type_tag == "#A#":
        text_info = "Available analysis are as follows: "
    elif type_tag == "#SP#":
        text_info = "Available analysis are as follows: "
    elif type_tag == "#LP#":
        text_info = "Available analysis are as follows: "
    
    df_text_filtered = df_text[df_text['end_date'] < end_date].sort_values('end_date')[-text_len:]
    
    if len(df_text_filtered) == 0:
        return "NA"
    
    if latest_first:
        df_text_filtered = df_text_filtered.iloc[::-1]

    extracted_texts = []
    for _, row in df_text_filtered.iterrows():
        extracted_text = row[text_col]
        
        extracted_texts.append(f"{row['start_date'].strftime('%Y-%m-%d')}: {extracted_text}")
        
    text_info = " ".join(extracted_texts)
    # TODO: do we need separation token? E.g., [SEP] for BERT.
    
    text_info = text_info.strip().replace('\n', '')
    
    return text_info

# %%
start_date = datetime(1983, 1, 1)
end_date = datetime(1983, 5, 7)

text_info = fetch_search_text(df_text_search, start_date, end_date)

# %%
SEARCH_FILE = f'../data/combined/{DATASET}/{DATASET}_search.csv'
REPORT_FILE = f'../data/combined/{DATASET}/{DATASET}_report.csv'
# %%
