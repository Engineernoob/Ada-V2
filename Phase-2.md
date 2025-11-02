Project: Ada Conversational AI
Goal: Upgrade Ada to Phase 2 — Contextual Conversation, Personality Modes, and Reinforcement Expansion.

Instructions:
You are building Phase 2 of Taahirah Denmark’s personal AI assistant, Ada.
Ada is a text-based conversational agent with reinforcement learning and a small neural head (AdaNet).
She must now gain emotional awareness, context memory, persona modes, and improved reinforcement behavior.

🧩 System Requirements

- Python 3.11+
- torch
- transformers
- rich
- json, datetime
- Maintain .venv, .gitignore, and folder structure.

🧱 Folder Layout
Create or update the following structure:
Ada/
├── core/
│ ├── neural_core.py
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
│ ├── models/
│
├── logs/
│ └── training_feedback.jsonl
│
└── main.py

🧠 Phase 2 Functional Goals

1. **Persona System**

   - Create `core/persona.py` defining multiple tone profiles (friendly, mentor, creative, analyst).
   - Allow dynamic persona switching by setting `CURRENT_PERSONA`.
   - Integrate persona parameters (temperature, top_p, tone) into `neural_core.py`.

2. **Context Memory**

   - Expand `core/memory.py` to support short-term recall and session summarization.
   - Save messages to `storage/memory/session.jsonl`.
   - Update `AdaCore.infer()` to prepend the last 6 turns as conversational context.

3. **Reinforcement Expansion**

   - Create `rl/reward_engine.py` that scores implicit emotional sentiment from user messages (+1 to –1 scale).
   - Modify `AdaCore.reinforce()` to handle both explicit `/rate` rewards and automatic feedback from `RewardEngine.analyze()`.

4. **Dialogue Interface**

   - Build `core/dialogue.py` with a CLI chat loop (quit/exit to stop).
   - Integrate AdaCore for message inference and memory updates.

5. **Main Entrypoint**

   - Update `main.py` to import `DialogueManager` from `core/dialogue.py`.
   - Run Ada in interactive text mode by default.

6. **Config Updates**
   - Add `ADA_SYSTEM_PROMPT` and other constants in `core/config.py`.
   - Ensure system prompt enforces “no internal reasoning or system echo”.

💬 Behavior Guidelines

- Ada should reply warmly, contextually, and emotionally.
- She remembers the last few exchanges.
- She adapts slightly based on sentiment and persona mode.
- Never echo system messages or reasoning text.
- Log all conversations and feedback.

✅ Output Deliverables

- Fully functional Ada v2.0 (Phase 2)
- Updated code for `neural_core.py`, `memory.py`, `persona.py`, `reward_engine.py`, `dialogue.py`, and `main.py`
- Ensure compatibility with existing AdaNet RL functions
- Keep imports clean, comments concise, and formatting PEP8-compliant.

When complete, display a summary of the added files, their purposes, and the entry command:
`python main.py`
