export PYTHONPATH=$PYTHONPATH:/path/to/cloned/folder
export CUDA_VISIBLE_DEVICES="[YOUR_GPU_INDICES]"
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model scope_vlm \
    --model_args="pretrained=Gyubeum/SCOPE-VLM-3B-SFT-Qwen2.5-VL,max_pixels=2007040,use_flash_attention_2=True,device_map=auto" \
    --tasks docvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix scope \
    --output_path ./results/docVQA.jsonl


#for lora
    # --model_args="pretrained=Gyubeum/SCOPE-VLM-3B-SFT-Qwen2.5-VL,lora=lorar/path,max_pixels=12845056,use_flash_attention_2=True" \

#Table 3 for qwen multi image
    #--model qwen2_5_vl \
    #--model_args="pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=2007040,use_flash_attention_2=True,device_map=auto" \

#Table 3 for qwen CoS
    #--model scope_vlm \
    #--model_args="pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=2007040,use_flash_attention_2=True,device_map=auto" \


dataset_path: YeMoKoo/DUDE
dataset_kwargs:
  data_files: dude_val.jsonl
task: "DUDE_val"
test_split: train
output_type: generate_until
doc_to_visual: !function utils.DUDE_doc_to_visual
doc_to_text: !function utils.DUDE_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  max_new_tokens: 32
  temperature: 0
  do_sample: False
process_results: !function utils.DUDE_process_results
metric_list:
  - metric: anls
    aggregation: !function utils.DUDE_aggregate_results_anls
    higher_is_better: true
  - metric: accuracy
    aggregation: !function utils.DUDE_aggregate_results_accuracy
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer the question using a single word or phrase."

