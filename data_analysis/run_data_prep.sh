# python data_align.py --data climate \
#     --text_source search \
#     --text_col fact,preds

# python data_analysis/data_align.py --data Environment \
#     --text_source search \
#     --text_col fact,preds \
#     --lookback_window 36

python data_analysis/data_align.py --data Agriculture \
    --text_source report \
    --text_col fact,preds \
    --lookback_window 36
