import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
import time
import random
import matplotlib.pyplot as plt
import traceback

class AIBenchmark:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2:3b", temperature=0.7, top_p=1.0)
        self.max_context_window = 20000
        self.used_tokens = 0
        self.retention_score = 100
        self.retention_history = [100]
        self.total_interactions = 0
        self.context = []
        self.token_limit_warning = False
        self.token_limit_threshold = 0.95  # % of max_context_window
        self.seed = None
        self.seed_value = None
        self.seed_first_remembered = None
        self.seed_last_remembered = None
        self.seed_forgotten_at = None
        self.interactions_since_last_seed_mention = 0

    def set_seed(self, seed):
        self.seed = seed
        self.seed_value = seed
        self.seed_first_remembered = None
        self.seed_last_remembered = None
        self.seed_forgotten_at = None
        self.interactions_since_last_seed_mention = 0
        self.context.append(f"System: Remember this seed value: {seed}")

    def chat(self, message, history):
        self.total_interactions += 1
        
        # Periodically ask about the seed
        if self.seed_value and self.total_interactions % 4 == 0:
            message += f"\n\nAlso, can you tell me what the seed value is?"
        
        self.context.append(f"Human: {message}")
        context_str = '\n'.join(self.context[-50:])
        
        template = """
        You are an AI assistant engaged in a conversation. Please answer the question based on the context provided.
        If you're asked about a seed value, only mention it if it's the exact value you were told to remember.
        
        Context:
        {context}
        
        Human: {message}
        AI:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = RunnableSequence(prompt | self.model)

        result = chain.invoke({
            "context": context_str,
            "message": message,
        })

        message_tokens = len(message.split())
        response_tokens = len(str(result).split())
        self.used_tokens += message_tokens + response_tokens
        self.update_retention_score()
        self.check_token_limit()

        self.context.append(f"AI: {result}")

        if self.used_tokens > self.max_context_window * 0.9:
            self.token_limit_warning = True

        # Check for seed retention
        if self.seed_value:
            seed_mentioned = self.check_seed_retention(str(result))
            if seed_mentioned:
                if self.seed_first_remembered is None:
                    self.seed_first_remembered = self.total_interactions
                self.seed_last_remembered = self.total_interactions
                self.interactions_since_last_seed_mention = 0
                if not self.seed:  # If it was previously forgotten, consider it remembered again
                    self.seed = self.seed_value
                    self.seed_forgotten_at = None
            else:
                self.interactions_since_last_seed_mention += 1
                if self.seed and self.interactions_since_last_seed_mention > 5:
                    # Consider it forgotten, but keep the seed_value
                    self.seed = None
                    self.seed_forgotten_at = self.total_interactions

        return str(result)
        
    def check_seed_retention(self, response):
        # Check if the exact seed is mentioned
        if self.seed_value in response:
            return True
        
        # Check for phrases indicating remembrance of the seed
        remembrance_phrases = [
            "The seed value is",
            "The seed you mentioned is",
            "You asked me to remember the seed",
            "The seed I was told to remember is"
        ]
        for phrase in remembrance_phrases:
            if phrase in response and self.seed_value in response.split(phrase)[1]:
                return True
        
        return False

    def get_seed_metrics(self):
        if not self.seed_value:
            return "No seed set"
        if not self.seed:
            return f"Seed: {self.seed_value}\nForgotten at: Interaction {self.seed_forgotten_at}\nLast remembered: Interaction {self.seed_last_remembered}"
        if self.seed_first_remembered is None:
            return f"Seed: {self.seed_value}\nNot yet remembered"
        return f"Seed: {self.seed_value}\nFirst remembered: Interaction {self.seed_first_remembered}\nLast remembered: Interaction {self.seed_last_remembered}"

    def update_retention_score(self):
        self.retention_score = max(0, 100 - (self.used_tokens / self.max_context_window * 100))
        self.retention_history.append(self.retention_score)

    def get_retention_score(self):
        status = 'green' if self.retention_score > 66 else 'yellow' if self.retention_score > 33 else 'red'
        return f"{self.retention_score:.2f}% ({status})"

    def check_token_limit(self):
        if self.used_tokens > self.max_context_window * self.token_limit_threshold:
            self.token_limit_warning = True
        else:
            self.token_limit_warning = False

    def get_token_metrics(self):
        return f"Total Tokens: {self.used_tokens}\nTotal Interactions: {self.total_interactions}\nAvg Tokens per Interaction: {self.used_tokens / max(1, self.total_interactions):.2f}\nToken Limit Warning: {'Yes' if self.token_limit_warning else 'No'}"

    def reset(self):
        self.__init__()
        return "Conversation and context reset."

    def set_temperature(self, temp):
        self.model.temperature = temp
        return f"Temperature set to {temp}"

    def set_top_p(self, top_p):
        self.model.top_p = top_p
        return f"Top P set to {top_p}"

    def set_max_tokens(self, max_tokens):
        self.max_context_window = int(max_tokens)
        return f"Max tokens set to {max_tokens}"

    def set_personality(self, personality):
        if personality == "Balanced":
            self.model.temperature = 0.7
            self.model.top_p = 1.0
        elif personality == "Creative":
            self.model.temperature = 0.9
            self.model.top_p = 0.9
        elif personality == "Precise":
            self.model.temperature = 0.3
            self.model.top_p = 0.6
        elif personality == "Code-focused":
            self.model.temperature = 0.2
            self.model.top_p = 0.5
        return f"Personality set to {personality}"

benchmark = AIBenchmark()

preset_prompts = [
    "Explain the concept of quantum entanglement.",
    "Write a short story about a time traveler.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis in plants.",
    "Discuss the impact of social media on modern society.",
    "Explain the basics of machine learning algorithms."
]

def generate_follow_up(response):
    follow_ups = [
        f"That's interesting. Can you elaborate on the part about {random.choice(response.split())}?",
        "How does this relate to real-world applications?",
        "What are some common misconceptions about this topic?",
        "Can you provide a concrete example to illustrate this concept?",
        "How has our understanding of this topic evolved over time?",
        "What are some potential future developments in this area?",
        "How does this compare to similar concepts in other fields?"
    ]
    return random.choice(follow_ups)

def plot_retention_history():
    plt.figure(figsize=(10, 4))
    plt.plot(benchmark.retention_history, marker='o')
    plt.title("Retention Score History")
    plt.xlabel("Interaction")
    plt.ylabel("Retention Score")
    plt.ylim(0, 100)
    plt.grid(True)
    return plt

def run_batch_process(chatbot, status, stop_threshold):
    try:
        status = "Initializing batch process..."
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), "", benchmark.get_seed_metrics()

        initial_prompt = random.choice(preset_prompts)
        status = f"Starting batch process with initial prompt: {initial_prompt}"
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), "", benchmark.get_seed_metrics()

        response = benchmark.chat(initial_prompt, chatbot)
        chatbot.append((initial_prompt, response))
        status = f"Initial response received. Retention score: {benchmark.get_retention_score()}"
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), "", benchmark.get_seed_metrics()

        interaction_count = 1
        while benchmark.retention_score > stop_threshold and not benchmark.token_limit_warning:
            follow_up = generate_follow_up(response)
            status = f"Interaction {interaction_count}: Generated follow-up question: {follow_up}"
            yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), "", benchmark.get_seed_metrics()

            response = benchmark.chat(follow_up, chatbot)
            chatbot.append((follow_up, response))
            status = f"Interaction {interaction_count}: Response received. Retention score: {benchmark.get_retention_score()}"
            warning = "Approaching token limit!" if benchmark.token_limit_warning else ""
            yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), warning, benchmark.get_seed_metrics()

            interaction_count += 1

        status = f"Batch processing completed. Final retention score: {benchmark.get_retention_score()}"
        chatbot.append((None, "Batch processing completed."))
        warning = "Token limit reached. Consider resetting the conversation." if benchmark.token_limit_warning else ""
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), warning, benchmark.get_seed_metrics()

    except Exception as e:
        error_message = f"Error in batch processing: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_message)
        chatbot.append((None, error_message))
        status = "An error occurred during batch processing. Check the chat for details."
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics(), "", benchmark.get_seed_metrics()

with gr.Blocks() as demo:
    gr.Markdown("# AI Context Retention Benchmark")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=600)
            msg = gr.Textbox(label="Message")
            
            with gr.Row():
                preset_buttons = [gr.Button(prompt) for prompt in preset_prompts]
            
            retention_plot = gr.Plot(label="Retention Score History")
            
            with gr.Row():
                batch_button = gr.Button("Start Batch Processing")
                clear = gr.Button("Reset")

            status = gr.Textbox(label="Batch Status", value="Ready to start batch processing.")

        with gr.Column(scale=1):
            seed_input = gr.Textbox(label="Seed Value")
            set_seed_button = gr.Button("Set Seed")
            seed_metrics = gr.Textbox(label="Seed Metrics", value="No seed set")
            
            retention_score = gr.Textbox(label="Current Retention Score", value=benchmark.get_retention_score())
            token_metrics = gr.Textbox(label="Token Metrics", value=benchmark.get_token_metrics())
            token_warning = gr.Textbox(label="Token Limit Warning", value="")
            
            stop_threshold = gr.Slider(0, 100, value=30, step=1, label="Stop Threshold (%)")
            
            gr.Markdown("## Model Parameters")
            personality = gr.Radio(["Balanced", "Creative", "Precise", "Code-focused"], label="Personality", value="Balanced")
            temperature = gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0, 1, value=1.0, step=0.1, label="Top P")
            max_tokens = gr.Slider(250, 20000, value=1280, step=100, label="Max Tokens")

            with gr.Accordion("Parameter Info", open=False):
                gr.Markdown("""
                **Temperature (0-1):** Controls randomness. Lower values make the model more deterministic, higher values make it more creative.
                
                **Top P (0-1):** Alternative to temperature. Controls diversity by considering only the most probable tokens. Lower values focus on likely tokens, higher values allow more diversity.
                
                **Max Tokens (10000-20000):** Maximum length of the context window. Higher values allow longer conversations but may affect performance.

                **Personalities:**
                - Balanced: Default settings (Temperature: 0.7, Top P: 1.0)
                - Creative: Higher randomness for more diverse outputs (Temperature: 0.9, Top P: 0.9)
                - Precise: Lower randomness for more focused outputs (Temperature: 0.3, Top P: 0.6)
                - Code-focused: Very low randomness, optimized for code generation (Temperature: 0.2, Top P: 0.5)
                """)

    def respond(message, chat_history):
        bot_message = benchmark.chat(message, chat_history)
        chat_history.append((message, bot_message))
        warning = "Approaching token limit! Consider resetting the conversation." if benchmark.token_limit_warning else ""
        return "", chat_history, benchmark.get_retention_score(), plot_retention_history(), benchmark.get_token_metrics(), warning, benchmark.get_seed_metrics()

    def reset_all():
        result = benchmark.reset()
        return [], result, benchmark.get_retention_score(), None, benchmark.get_token_metrics(), "", "Balanced", 0.7, 1.0, 20000, "Ready to start batch processing.", "No seed set"

    def set_seed(seed):
        benchmark.set_seed(seed)
        return benchmark.get_seed_metrics()

    def update_personality(personality):
        benchmark.set_personality(personality)
        if personality == "Balanced":
            return 0.7, 1.0
        elif personality == "Creative":
            return 0.9, 0.9
        elif personality == "Precise":
            return 0.3, 0.6
        elif personality == "Code-focused":
            return 0.2, 0.5

    msg.submit(respond, [msg, chatbot], [msg, chatbot, retention_score, retention_plot, token_metrics, token_warning, seed_metrics])
    clear.click(reset_all, outputs=[chatbot, msg, retention_score, retention_plot, token_metrics, token_warning, personality, temperature, top_p, max_tokens, status, seed_metrics])
    set_seed_button.click(set_seed, inputs=[seed_input], outputs=[seed_metrics])

    personality.change(update_personality, personality, [temperature, top_p])
    temperature.change(benchmark.set_temperature, temperature, None)
    top_p.change(benchmark.set_top_p, top_p, None)
    max_tokens.change(benchmark.set_max_tokens, max_tokens, None)

    for button in preset_buttons:
        button.click(lambda x: x, button, msg).then(
            respond, [msg, chatbot], [msg, chatbot, retention_score, retention_plot, token_metrics, token_warning]
        )

    batch_button.click(run_batch_process, [chatbot, status, stop_threshold], [chatbot, retention_score, retention_plot, status, token_metrics, token_warning, seed_metrics])

if __name__ == "__main__":
    demo.launch()