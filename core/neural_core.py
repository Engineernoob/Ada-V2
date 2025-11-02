import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.memory import Memory
from core.config import *

# Silence Hugging Face warnings
import transformers
transformers.utils.logging.set_verbosity_error()

# 🧩 AdaNet — trainable neural head (for reinforcement tuning)
class AdaNet(nn.Module):
    def __init__(self, input_size=512, hidden=256, output_size=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, output_size)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# 🧠 AdaCore — combines Qwen backbone + AdaNet + reinforcement
class AdaCore:
    def __init__(self):
        print(f"🧠 Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.lm = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.memory = Memory(limit=MEMORY_LIMIT)
        self.head = AdaNet()
        self.optimizer = optim.Adam(self.head.parameters(), lr=LR)
        self.rewards = []

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lm.to(self.device)
        print(f"✅ AdaCore initialized on {self.device.upper()}")

    # 💬 Conversational inference
        # 💬 Conversational inference (Qwen-3 clean version)
    def infer(self, prompt: str):
        # Build conversational prompt with clear system rule
        messages = [
            {"role": "system", "content": ADA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Prepare the prompt for Qwen's chat template
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True).to(self.device)

        # Generate Ada's reply
        with torch.no_grad():
            outputs = self.lm.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id  # clean stop point
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🧹 Step 1 — remove hidden reasoning or chain-of-thought
        stop_markers = [
            "<think>", "</think>", "I need to respond", "Let me think",
            "I should", "So the user", "The user asked", "To respond",
            "my response should be", "I will", "Okay,", "Firstly,"
        ]
        for phrase in stop_markers:
            if phrase in text:
                text = text.split(phrase)[0].strip()

        # 🧹 Step 2 — clean leftover tokens or metadata
        for marker in ["<|im_start|>", "<|im_end|>", "assistant", "user"]:
            text = text.replace(marker, "").strip()

        # 🧹 Step 3 — collapse double spaces, stray punctuation
        response = " ".join(text.split())

        # 🧠 Update Ada's conversational memory
        self.memory.add(f"You: {prompt}")
        self.memory.add(f"Ada: {response}")

        return response

    # 🧩 Reinforcement update (for /rate)
    def reinforce(self, reward: float):
        self.rewards.append(reward)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        pred = torch.mean(torch.stack([p.mean() for p in self.head.parameters()]))
        loss = -reward_tensor * pred
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Display reinforcement result
        emoji = "🌟" if reward > 0 else ("⚖️" if reward == 0 else "🚫")
        print(f"{emoji} Reinforced with reward {reward:+.2f} | Loss {loss.item():.4f}")

    def load_brain(self, path="storage/models/adacore_head.pth"):
        try:
            self.head.load_state_dict(torch.load(path))
            print("🧠 Loaded trained Ada neural head.")
        except FileNotFoundError:
            print("⚠️ No trained Ada head found; starting fresh.")

    def save_brain(self, path="storage/models/adacore_head.pth"):
        torch.save(self.head.state_dict(), path)
        print(f"💾 Ada neural head saved to {path}")
