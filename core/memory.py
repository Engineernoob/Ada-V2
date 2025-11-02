class Memory:
    def __init__(self, limit=3):
        self.history = []
        self.limit = limit
    def add(self, text: str):
        self.history.append(text)
        if len(self.history) > self.limit:
            self.history.pop(0)
    def recall(self):
        return self.history
