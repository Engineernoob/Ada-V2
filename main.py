from core.neural_core import AdaCore
from core.dialogue import DialogueManager

def main():
    ada = AdaCore()
    DialogueManager(ada).chat()

if __name__ == "__main__":
    main()
