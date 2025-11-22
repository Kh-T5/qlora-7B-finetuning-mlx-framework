Project: QLoRA Fine-Tuning of a msitral-7b on Databricks Dolly 15k

Objective: Turn the base model into an instruction-following assistant using parameter-efficient fine-tuning on a human-written dataset.

Data: databricks/databricks-dolly-15k (15k prompt–response pairs, CC-BY-SA 3.0).


Method:
    - QLoRA:
        • Load base model in 4-bit quantization (Apple Silicon backend). 
        • Insert LoRA adapters on attention projection layers. 
    - Fine tuning:
        • Supervised fine-tune on Dolly for a few epochs with small batch size.
        • Save and optionally merge adapters into the base model.

Deliverables:
• Training script 
• Data prep script 
• Inference script 


Environmental dependencies can be found in "env.yml" & "requirements.txt" files
