$CUDA_VISIBLE_DEVICES = $args[2]

$all_models = @("Transformer")
$start_index = $args[0]
$end_index = $args[1]
$root_paths = @("./data/combined/Environment")
$data_paths = @("Environment_search.csv")
$models = $all_models[$start_index..($end_index-1)]
$d_model = @(16, 32, 64)
$e_layers = @(3)
$d_layers = @(0)
$revin = 0
$text_len = 1
$text_ts = 0
$text_col = @("NA")
$text_random = 0
$text_random_order = 0
$text_emb = 32
$length = $root_paths.Length
$pred_lengths = @(24)
$seeds = @(2024)
$use_fullmodel = 1
$llm_model = "GPT2"
$llm_from_stratch = 0
$fusion_method = "add"
$prompt_weight = @(0)
$history_weight = 0
$connector_dropout = @(0.2)
$learning_rate = @(1e-3, 5e-4, 1e-4)
$train_epochs = @(10, 20)

foreach ($seed in $seeds) {
  foreach ($model_name in $models) {
    for ($i = 0; $i -lt $length; $i++) {
      foreach ($pred_len in $pred_lengths) {
        $root_path = $root_paths[$i]
        $data_path = $data_paths[$i]
        $model_id = [System.IO.Path]::GetFileName($root_path)

        Write-Output "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"

        foreach ($pw in $prompt_weight) {
          foreach ($tcol in $text_col) {
            foreach ($dm in $d_model) {
              foreach ($el in $e_layers) {
                foreach ($dl in $d_layers) {
                  foreach ($cdrop in $connector_dropout) {
                    foreach ($lr in $learning_rate) {
                      foreach ($te in $train_epochs) {
                        & python -u run.py `
                          --task_name long_term_forecast `
                          --is_training 1 `
                          --root_path $root_path `
                          --data_path $data_path `
                          --model_id "${model_id}_${seed}_36_${pred_len}_text${text_len}_fm${fusion_method}_pw${pw}_fullLLM${use_fullmodel}" `
                          --model $model_name `
                          --data custom `
                          --features M `
                          --seq_len 36 `
                          --label_len 0 `
                          --pred_len $pred_len `
                          --des "Exp" `
                          --seed $seed `
                          --type_tag "#F#" `
                          --d_model $dm `
                          --e_layers $el `
                          --d_layers $dl `
                          --rev_in $revin `
                          --text_emb $text_emb `
                          --text_len $text_len `
                          --text_ts $text_ts `
                          --text_col $tcol `
                          --text_random $text_random `
                          --text_random_order $text_random_order `
                          --fusion_method $fusion_method `
                          --prompt_weight $pw `
                          --history_weight $history_weight `
                          --connector_dropout $cdrop `
                          --pool_type "last" `
                          --save_name "results/${model_id}_${seed}_36_${pred_len}_text${text_len}_fm${fusion_method}_pw${pw}_fullLLM${use_fullmodel}" `
                          --llm_model $llm_model `
                          --llm_from_stratch $llm_from_stratch `
                          --use_fullmodel $use_fullmodel `
                          --learning_rate $lr `
                          --train_epochs $te `
                          2>&1 | Tee-Object -FilePath "logs/${model_id}_${seed}_36_${pred_len}_text${text_len}_fm${fusion_method}_pw${pw}_fullLLM${use_fullmodel}.log"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
