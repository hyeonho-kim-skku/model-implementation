import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class DINOV2LoRA(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, lora_rank, use_quantization):
        super().__init__()

        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
                llm_int8_skip_modules=["classifier"] # classifier는 4bit 양자화하지 않음
            )
        else:
            bnb_config = None
        
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            device_map="auto",
            ignore_mismatched_sizes=True, # pre-trained 모델의 classifier 무시하고 num_classes에 맞는 classifier 초기화
            quantization_config=bnb_config,
            dtype=torch.float32
        )

        if use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank*2,
            target_modules=["query", "value"],
            bias="none",
            modules_to_save=["classifier"] # classifier는 LoRA 적용하지 않고 학습
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits