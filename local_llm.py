import json
import importlib.util
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass


class LocalBielikLLM:
    def __init__(
        self,
        model_id="speakleash/Bielik-1.5B-v3.0-Instruct",
        local_model_dir="models/Bielik-1.5B-v3.0-Instruct",
        max_new_tokens=700,
        temperature=0.1,
        top_p=0.9,
    ):
        self.model_id = model_id
        self.local_model_dir = os.path.normpath(os.path.abspath(local_model_dir))
        self.max_new_tokens = max_new_tokens # max len of tokens to generate in response
        self.temperature = temperature # lower values make output more deterministic, higher values make it more random
        self.top_p = top_p # nucleus sampling: 0.9 means only consider tokens that make up the top 90% probability mass

        model_path = self._ensure_local_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        from_pretrained_kwargs = {
            "torch_dtype": dtype,
        }

        try:
            has_accelerate = importlib.util.find_spec("accelerate") is not None
            if has_accelerate:
                from_pretrained_kwargs["device_map"] = "auto"
        except Exception:
            pass

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **from_pretrained_kwargs,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "torchvision::nms" in message:
                raise RuntimeError(
                    "Wykryto konflikt bibliotek torch/torchvision (operator torchvision::nms). "
                    "Dla modelu tekstowego Bielik odinstaluj torchvision: "
                    "python -m pip uninstall -y torchvision"
                ) from exc
            raise

        if "device_map" not in from_pretrained_kwargs:
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")

    def _ensure_local_model(self):
        config_path = os.path.join(self.local_model_dir, "config.json")
        if os.path.exists(config_path):
            return self.local_model_dir

        os.makedirs(self.local_model_dir, exist_ok=True)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        try:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.local_model_dir,
                token=token,
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                "Brak dostepu do gated modelu na Hugging Face. "
                "1) Popros o dostep na stronie modelu. "
                "2) Zaloguj sie: huggingface-cli login. "
                "3) Albo ustaw zmienna srodowiskowa HF_TOKEN z tokenem. "
                "4) Uruchom ponownie proces."
            ) from exc
        return self.local_model_dir

    def generate_text(self, prompt):
        messages = [{"role": "user", "content": prompt.strip()}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = model_inputs["input_ids"].to(self.model.device)
        attention_mask = model_inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_json(self, prompt):
        text = self.generate_text(prompt)

        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError("Model nie zwrócił poprawnego JSON")
