import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
import time
import random
import matplotlib.pyplot as plt
import traceback
import asyncio

class AIBenchmark:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2:3b", temperature=0.7, top_p=1.0)
        self.max_tokens = 5000
        self.context = ""
        self.retention_score = 100
        self.retention_history = [100]  # Start with initial 100% score
        self.total_tokens = 0
        self.total_interactions = 0

    def chat(self, message, history):
        template = """
        Answer the question below. 
        Here is the conversation history:
        {context}
        Question: {message}
        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = RunnableSequence(prompt | self.model)

        result = chain.invoke({
            "context": self.context,
            "message": message,
        })

        # Truncate the result to max_tokens
        result = ' '.join(str(result).split()[:self.max_tokens])

        # Update retention score and token count
        self.retention_score = max(0, self.retention_score - 5)
        self.retention_history.append(self.retention_score)
        self.total_tokens += len(message.split()) + len(result.split())
        self.total_interactions += 1

        self.context += f"\nHuman: {message}\nAI: {result}"
        return str(result)

    def get_retention_score(self):
        status = 'green' if self.retention_score > 66 else 'yellow' if self.retention_score > 33 else 'red'
        return f"{self.retention_score}% ({status})"

    def get_token_metrics(self):
        return f"Total Tokens: {self.total_tokens}\nTotal Interactions: {self.total_interactions}\nAvg Tokens per Interaction: {self.total_tokens / max(1, self.total_interactions):.2f}"

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
        self.max_tokens = int(max_tokens)
        return f"Max tokens set to {max_tokens}"

    def set_personality(self, personality):
        if personality == "Balanced":
            self.model.temperature = 0.7
            self.model.top_p = 1.0
            self.max_tokens = 5000
        elif personality == "Creative":
            self.model.temperature = 0.9
            self.model.top_p = 0.9
            self.max_tokens = 5000
        elif personality == "Precise":
            self.model.temperature = 0.3
            self.model.top_p = 0.6
            self.max_tokens = 5000
        elif personality == "Code-focused":
            self.model.temperature = 0.2
            self.model.top_p = 0.5
            self.max_tokens = 5000
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
    plt.plot(benchmark.retention_history, marker='o')  # Added marker for clarity
    plt.title("Retention Score History")
    plt.xlabel("Interaction")
    plt.ylabel("Retention Score")
    plt.ylim(0, 100)
    plt.grid(True)
    return plt

def run_batch_process(chatbot, status, stop_threshold):
    try:
        status = "Initializing batch process..."
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

        initial_prompt = random.choice(preset_prompts)
        status = f"Starting batch process with initial prompt: {initial_prompt}"
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

        response = benchmark.chat(initial_prompt, chatbot)
        chatbot.append((initial_prompt, response))
        status = f"Initial response received. Retention score: {benchmark.get_retention_score()}"
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

        interaction_count = 1
        while benchmark.retention_score > stop_threshold:
            follow_up = generate_follow_up(response)
            status = f"Interaction {interaction_count}: Generated follow-up question: {follow_up}"
            yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

            response = benchmark.chat(follow_up, chatbot)
            chatbot.append((follow_up, response))
            status = f"Interaction {interaction_count}: Response received. Retention score: {benchmark.get_retention_score()}"
            yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

            interaction_count += 1

        status = f"Batch processing completed. Final retention score: {benchmark.get_retention_score()}"
        chatbot.append((None, "Batch processing completed."))
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

    except Exception as e:
        error_message = f"Error in batch processing: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_message)
        chatbot.append((None, error_message))
        status = "An error occurred during batch processing. Check the chat for details."
        yield chatbot, benchmark.get_retention_score(), plot_retention_history(), status, benchmark.get_token_metrics()

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
            retention_score = gr.Textbox(label="Current Retention Score", value=benchmark.get_retention_score())
            token_metrics = gr.Textbox(label="Token Metrics", value=benchmark.get_token_metrics())
            
            stop_threshold = gr.Slider(0, 100, value=30, step=1, label="Stop Threshold (%)")
            
            gr.Markdown("## Model Parameters")
            personality = gr.Radio(["Balanced", "Creative", "Precise", "Code-focused"], label="Personality", value="Balanced")
            temperature = gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0, 1, value=1.0, step=0.1, label="Top P")
            max_tokens = gr.Slider(10, 5000, value=5000, step=10, label="Max Tokens")

            with gr.Accordion("Parameter Info", open=False):
                gr.Markdown("""
                **Temperature (0-1):** Controls randomness. Lower values make the model more deterministic, higher values make it more creative.
                
                **Top P (0-1):** Alternative to temperature. Controls diversity by considering only the most probable tokens. Lower values focus on likely tokens, higher values allow more diversity.
                
                **Max Tokens (10-5000):** Maximum length of the model's response. Higher values allow longer responses but may affect context retention.

                **Personalities:**
                - Balanced: Default settings (Temperature: 0.7, Top P: 1.0, Max Tokens: 5000)
                - Creative: Higher randomness for more diverse outputs (Temperature: 0.9, Top P: 0.9, Max Tokens: 5000)
                - Precise: Lower randomness for more focused outputs (Temperature: 0.3, Top P: 0.6, Max Tokens: 5000)
                - Code-focused: Very low randomness, optimized for code generation (Temperature: 0.2, Top P: 0.5, Max Tokens: 5000)
                """)

    def respond(message, chat_history):
        bot_message = benchmark.chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history, benchmark.get_retention_score(), plot_retention_history(), benchmark.get_token_metrics()

    def reset_all():
        result = benchmark.reset()
        return [], result, benchmark.get_retention_score(), None, benchmark.get_token_metrics(), "Balanced", 0.7, 1.0, 5000, "Ready to start batch processing."

    def update_personality(personality):
        benchmark.set_personality(personality)
        if personality == "Balanced":
            return 0.7, 1.0, 5000
        elif personality == "Creative":
            return 0.9, 0.9, 5000
        elif personality == "Precise":
            return 0.3, 0.6, 5000
        elif personality == "Code-focused":
            return 0.2, 0.5, 5000

    msg.submit(respond, [msg, chatbot], [msg, chatbot, retention_score, retention_plot, token_metrics])
    clear.click(reset_all, outputs=[chatbot, msg, retention_score, retention_plot, token_metrics, personality, temperature, top_p, max_tokens, status])

    personality.change(update_personality, personality, [temperature, top_p, max_tokens])
    temperature.change(benchmark.set_temperature, temperature, None)
    top_p.change(benchmark.set_top_p, top_p, None)
    max_tokens.change(benchmark.set_max_tokens, max_tokens, None)

    for button in preset_buttons:
        button.click(lambda x: x, button, msg).then(
            respond, [msg, chatbot], [msg, chatbot, retention_score, retention_plot, token_metrics]
        )

    batch_button.click(run_batch_process, [chatbot, status, stop_threshold], [chatbot, retention_score, retention_plot, status, token_metrics])

if __name__ == "__main__":
    demo.launch()