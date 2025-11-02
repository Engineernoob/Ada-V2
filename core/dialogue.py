class DialogueManager:
    def __init__(self, core):
        self.core = core

    def chat(self):
        print("✨ Ada v2.1 online — type /rate <value> to reward, or 'quit' to exit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Ada: Until next time ✨")
                break
            if user_input.startswith("/rate"):
                try:
                    reward = float(user_input.split()[1])
                    self.core.reinforce(reward)
                except Exception:
                    print("⚠️ Usage: /rate <number>")
                continue
            response = self.core.infer(user_input)
            print(f"Ada: {response}\n")
