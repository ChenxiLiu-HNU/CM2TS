import argparse
from collections import defaultdict
import os
import random
import re
import string
import pandas as pd


def generate_random_text(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def align_time_series_text(df_ts, df_text, config):
    freq = abs(df_ts.iloc[1]["date"] - df_ts.iloc[0]["date"])

    new_text_cols = defaultdict(list)
    for _, row in df_ts.iterrows():
        shift = config.text_lookback_shift * freq
        end_date = row["date"] - shift
        start_date = row["date"] - freq * config.lookback_window - shift
        text_out = fetch_text(
            df_text,
            start_date,
            end_date,
            text_col=config.text_col,
            append_timestamp=config.text_append_timestamp,
        )
        for k, v in text_out.items():
            col_name = f"{config.text_source}_{k}_lb{config.lookback_window}"
            new_text_cols[col_name].append(v)

    df_aligned = df_ts.assign(**new_text_cols)

    return df_aligned


def fetch_text(
    df_text,
    start_date,
    end_date,
    n_kept=1,
    text_col="fact,preds",
    latest_first=True,
    append_timestamp=False,
):
    filter_cond = (df_text["end_date"] <= end_date) & (df_text["start_date"] >= start_date)
    df_text_filtered = df_text[filter_cond].sort_values("end_date")

    if latest_first:
        df_text_filtered = df_text_filtered.iloc[::-1]

    df_text_filtered = df_text_filtered[:n_kept]

    cols = text_col.split(",")
    text_out = {}
    for col in cols:
        if len(df_text_filtered) == 0:
            texts = ""
        else:
            extracted_texts = []
            for _, row in df_text_filtered.iterrows():
                extracted_text = row[col]

                if not isinstance(extracted_text, str):
                    extracted_text = ""
                else:
                    extracted_text = extracted_text.strip()
                    if extracted_text == "" or extracted_text.startswith("NA"):
                        extracted_text = ""
                    elif append_timestamp:
                        extracted_text = f"{row['end_date'].strftime('%Y-%m-%d')}: {extracted_text}"

                extracted_texts.append(extracted_text)

            texts = " ".join(extracted_texts).strip()
            # TODO: do we need separation token? E.g., [SEP] for BERT.

            texts = re.sub(r"\s+", " ", texts).strip()
        text_out[col] = texts

    return text_out


def main():
    parser = argparse.ArgumentParser(description="TimeMMDData")

    # data loader
    parser.add_argument("--data", type=str, default="Energy", help="dataset type")
    parser.add_argument(
        "--root_path", type=str, default="./data/", help="root path of the data file"
    )
    parser.add_argument("--text_path", type=str, default="textual/", help="textual data file")
    parser.add_argument("--text_source", type=str, default="report", help="text source")
    parser.add_argument(
        "--time_series_path",
        type=str,
        default="numerical/",
        help="time series data file",
    )
    parser.add_argument(
        "--fusion_path",
        type=str,
        default="combined/",
        help="text time series fusion data file",
    )

    # textual data config
    parser.add_argument(
        "--text_col",
        type=str,
        default="fact,preds",
        help="target column name in text file",
    )
    parser.add_argument(
        "--text_lookback_shift",
        type=int,
        default=0,
        help="shift text time window by n steps",
    )
    parser.add_argument(
        "--lookback_window",
        type=int,
        default=1,
        help="number of text windows to look back",
    )
    parser.add_argument(
        "--text_n_kept",
        type=int,
        default=1,
        help="number of text records to keep",
    )
    parser.add_argument("--text_latest_first", action="store_true", help="latest record first")
    parser.add_argument("--text_append_timestamp", action="store_true", help="append timestamp")

    args = parser.parse_args()

    # load time series and text data
    TIME_SERIES_FILE = os.path.join(
        args.root_path, args.time_series_path, args.data, f"{args.data}.csv"
    )
    TEXT_FILE = os.path.join(
        args.root_path, args.text_path, args.data, f"{args.data}_{args.text_source}.csv"
    )

    df_text = pd.read_csv(TEXT_FILE, header=0, index_col=0)
    df_text["start_date"] = pd.to_datetime(df_text["start_date"])
    df_text["end_date"] = pd.to_datetime(df_text["end_date"])
    df_ts = pd.read_csv(TIME_SERIES_FILE, header=0)
    df_ts["date"] = pd.to_datetime(df_ts["date"])
    df_ts = df_ts.drop(columns=["start_date", "end_date"])

    df_aligned = align_time_series_text(df_ts, df_text, args)

    # write to file
    ALIGNED_OUTPUT_DIR = os.path.join(args.root_path, args.fusion_path, args.data)

    if not os.path.exists(ALIGNED_OUTPUT_DIR):
        os.makedirs(ALIGNED_OUTPUT_DIR)

    ALIGNED_FILE = os.path.join(ALIGNED_OUTPUT_DIR, f"{args.data}_{args.text_source}.csv")
    df_aligned.to_csv(ALIGNED_FILE, sep=",", encoding="utf-8", index=False, header=True)


if __name__ == "__main__":
    main()
