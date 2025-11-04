Project: Ada Conversational AI
Goal: Implement Phase 3 — Long-Term Memory, Reflection, and Persistent Reinforcement.

Instructions:
You are extending Taahirah Denmark’s Ada conversational AI from Phase 2 to Phase 3.
Ada already has:
• A neural core (Qwen 3 + AdaNet RL head)
• Persona system
• Reinforcement feedback
• Short-term context memory

Phase 3 adds:
1️⃣ Persistent long-term memory across sessions
2️⃣ Embedding-based semantic recall
3️⃣ Automatic reflection summaries
4️⃣ Smarter reinforcement logging tied to tone and topic

🧩 Requirements

- Python 3.11+
- torch
- transformers
- faiss-cpu OR chromadb
- sentence-transformers OR nomic-embed-text
- json, uuid, datetime
- Keep compatibility with existing `.venv`, `.gitignore`, and structure.

🧱 Folder Structure
Ada/
├── core/
│ ├── long*memory.py # NEW: vector/semantic memory
│ ├── reflection.py # NEW: session summarizer & self-review
│ ├── neural_core.py # UPDATED: integrates long-term recall
│ ├── memory.py
│ ├── persona.py
│ ├── dialogue.py
│ ├── config.py
│
├── rl/
│ ├── trainer.py
│ ├── reward_engine.py
│
├── storage/
│ ├── memory/
│ │ ├── long_memory.faiss or chroma.db
│ │ ├── summaries/
│ │ └── session*<date>.jsonl
│ ├── models/
│
├── logs/
│ └── training_feedback.jsonl
│
└── main.py

🧠 Phase 3 Functional Goals

1️⃣ Long-Term Memory

- Create `core/long_memory.py` implementing:
  ```python
  class LongMemory:
      def add(self, text: str)
      def query(self, context: str, top_k=3) -> list[str]
      def summarize_session(self, history: list[str]) -> str
  ```
- Use embeddings via `sentence-transformers/all-MiniLM-L6-v2` (or nomic-embed-text).
- Store (uuid, vector, text, timestamp) to FAISS/ChromaDB.
- Return top-k similar past contexts for each new prompt.

2️⃣ Reflection System

- Add `core/reflection.py` that:
  - Summarizes each session (≈ 5-10 turns) into natural language.
  - Computes sentiment averages from rewards.
  - Saves reflections to `storage/memory/summaries/<date>.txt`.
  ```python
  def reflect(session_history: list[str], avg_reward: float) -> str
  ```

3️⃣ Integration with Neural Core

- In `neural_core.py`:
  - After each `infer()`, call `LongMemory.add()` for user + Ada messages.
  - Before generation, retrieve `LongMemory.query(prompt)` and prepend it to context.
  - On exit, trigger Reflection to summarize and store the session.
  - Continue logging all reinforcement data.

4️⃣ Persistent Sessions

- Each run creates a timestamped session file:
  `storage/memory/session_YYYY-MM-DD_HHMM.jsonl`
- On startup, Ada loads last session summaries to regain context.

5️⃣ Behavior Enhancements

- When long-memory finds related history, Ada naturally references it:
  “I remember we discussed this before — you mentioned …”
- Reinforcement and RewardEngine continue influencing tone.

6️⃣ Config Updates
In `core/config.py` add:

```python
MEMORY_DB_PATH = "storage/memory/long_memory.faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFLECTION_INTERVAL = 10
✅ Output Deliverables

New: long_memory.py, reflection.py

Updated: neural_core.py

Persistent vector memory database in storage/memory/

Session summaries in storage/memory/summaries/

Full PEP8 compliance and docstrings

💬 Expected Behavior

Ada recalls previous session context (“You mentioned … yesterday”)

Generates more coherent, personalized dialogue

Produces reflection summary on exit

Learns continually via reinforcement

Verification:
Run:
python main.py
Then confirm Ada can reference prior topics and write a reflection file at the end of chat.
```
