Project: Ada Conversational AI
Goal: Implement Phase 3.8 — Adaptive AdaNet v3 Integration (Context-Aware Reinforcement Learning Head)

Instructions:
You are upgrading Taahirah Denmark’s Ada assistant from AdaNet v2 to AdaNet v3.  
Ada currently runs on microsoft/DialoGPT-medium with reinforcement feedback and short-term memory.  
Phase 3.8 gives her a semantic policy head that learns contextually and emotionally from conversation history.

───────────────────────────────
🧱 Folder Structure
Ada/
├── core/
│ ├── neural_core.py # updated: integrate AdaNet v3 + embedder
│ ├── persona.py
│ ├── memory.py
│ ├── config.py
│
├── rl/
│ ├── reward_engine.py
│ ├── replay_buffer.py
│ ├── reward_memory.py ← NEW
│
└── main.py
───────────────────────────────

───────────────────────────────
PHASE 3.8 TASKS
───────────────────────────────

1️⃣ AdaNet v3 — Policy Head
• Replace AdaNet class in neural_core.py:
```python
import torch, torch.nn as nn, torch.nn.functional as F

    class AdaNet(nn.Module):
        """
        AdaNet v3 — Adaptive policy head.
        Learns tone, style, and reward prediction from text embeddings.
        """
        def __init__(self, input_size=768, hidden1=512, hidden2=256, output_size=3, dropout=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, output_size)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.norm(F.relu(self.fc2(x)))
            return torch.tanh(self.fc3(x))   # [style, tone, reward_pred]
    ```

───────────────────────────────
2️⃣ Semantic Embedding Input
• Install sentence-transformers.
• In AdaCore.**init**():
`python
    from sentence_transformers import SentenceTransformer
    self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    `
• In reinforce() and/or train loop, encode the last Ada response:
`python
    embedding = self.embedder.encode([text], convert_to_tensor=True)
    policy_out = self.head(embedding)
    `

───────────────────────────────
3️⃣ Multi-Objective Reinforcement
• Add losses inside reinforce():
```python
def reinforce(self, text, reward):
emb = self.embedder.encode([text], convert_to_tensor=True)
style, tone, reward_pred = self.head(emb)[0]
r = torch.tensor(reward, dtype=torch.float32)

        policy_loss = F.mse_loss(reward_pred, r)
        entropy_loss = -0.01 * (torch.abs(style) + torch.abs(tone)).mean()
        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.optimizer.step()
        print(f"🧠 Policy {policy_loss.item():.4f} | Entropy {entropy_loss.item():.4f}")
    ```

───────────────────────────────
4️⃣ Reward Memory Buffer
• Create rl/reward_memory.py:
`python
    import random
    class RewardMemory:
        def __init__(self, capacity=500):
            self.buffer = []
            self.capacity = capacity
        def add(self, prompt, response, reward):
            if len(self.buffer) >= self.capacity: self.buffer.pop(0)
            self.buffer.append((prompt, response, reward))
        def sample(self, n=8):
            return random.sample(self.buffer, min(len(self.buffer), n))
    `
• In AdaCore.**init**():
`python
    from rl.reward_memory import RewardMemory
    self.reward_memory = RewardMemory()
    `
• When user rates Ada, call:
`python
    self.reward_memory.add(prompt, response, reward)
    `

───────────────────────────────
5️⃣ Replay Training
• Periodically replay samples to reinforce long-term trends:
`python
    def replay_train(self):
        for p, r, rew in self.reward_memory.sample(8):
            emb = self.embedder.encode([r], convert_to_tensor=True)
            style, tone, pred = self.head(emb)[0]
            loss = F.mse_loss(pred, torch.tensor(rew))
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    `
• Call this every few sessions or after N reinforcements.

───────────────────────────────
6️⃣ Optimizer & Scheduler
• Use:
`python
    self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-4, betas=(0.9,0.99))
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    `

───────────────────────────────
✅ Deliverables

- Updated neural_core.py with AdaNet v3 + embedder integration.
- New rl/reward_memory.py.
- Normalized, stable reinforcement loop with entropy regularization.
- Logging printouts for Policy/Entropy losses.

💬 Expected Behavior

- Ada learns from text meaning, not just scalar reward.
- Adapts tone and warmth dynamically.
- Reinforcement updates remain stable over time.
- Conversations feel increasingly personalized as Ada builds memory of what earns positive feedback.

Verification:
Run:

```bash
python3 main.py
```
