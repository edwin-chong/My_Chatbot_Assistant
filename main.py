from Chatbot import ChatBot

def _main_menu(self):
    """Loop the main menu until the user chooses to exit."""
    while True:
        user_choice = input("What do you want to do?\n  1. Process a text file\n  2. Remove Pinecone index\nType 'exit' to quit\n")
        if user_choice == '1':
            _process_text_file(self)
        elif user_choice == '2':
            self._clear_index()
        elif user_choice == 'exit':
            print("Exiting program.")
            quit()
        else:
            print("Invalid choice. Please try again.")

def _process_text_file(self):
    """Handle processing of a text file."""
    file_name = input("Please enter the file path: ")
    file_path = "./data/" + file_name
    self.process_file(file_path, file_name)
    _question_loop()

def _question_loop():
    """Loop to handle user questions."""
    while True:
        user_question = input("Ask me anything (Enter 'exit' or 'quit' to go back to main menu): ")
        if user_question.lower() in ['exit', 'quit']:
            break
        result = bot.rag_chain.invoke(user_question)
        print(result)

bot = ChatBot()
_main_menu(bot)