import json, os, datetime
LOG_FILE = "logs/training_feedback.jsonl"

def log_feedback(prompt, response, reward):
    os.makedirs("logs", exist_ok=True)
    entry = {
        "time": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "reward": reward
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
