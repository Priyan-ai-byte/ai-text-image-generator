from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class TextModel:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove prompt from output
        answer = full_text.replace(prompt, "").strip()

        return answer