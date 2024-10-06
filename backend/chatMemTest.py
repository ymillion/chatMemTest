import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
import random
import traceback

class AIBenchmark:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2:3b", temperature=0.7, top_p=1.0, max_tokens=1280)
        self.max_tokens = 1280
        self.context_window_size = 0
        self.max_context_window = 128000
        self.used_tokens = 0
        self.total_interactions = 0
        self.context = []
        self.batch_running = False
        self.last_batch_question = None
        self.memkey = None
        self.memkey_value = None
        self.memkey_first_remembered = None
        self.memkey_last_remembered = None
        self.memkey_forgotten_at = None
        self.interactions_since_last_memkey_mention = 0
        self.memkey_query_frequency = 3
    
    def set_query_frequency(self, frequency):
        self.memkey_query_frequency = int(frequency)
        return f"MemKey query frequency set to {frequency}"

    def set_memkey(self, memkey):
        self.memkey = memkey
        self.memkey_value = memkey
        self.memkey_first_remembered = None
        self.memkey_last_remembered = None
        self.memkey_forgotten_at = None
        self.interactions_since_last_memkey_mention = 0
        self.context.append(f"System: Remember this MemKey value: {memkey}")

    def chat(self, message, history):
        self.total_interactions += 1

        message_tokens = len(message.split())

        if self.memkey and self.total_interactions % self.memkey_query_frequency == 0:
            message += f"\n\nAlso, can you tell me what the MemKey value is?"
            message_tokens = len(message.split())
        
        self.context.append(f"Human: {message}")
        context_str = '\n'.join(self.context[-50:])

        template = """
        You are an AI assistant engaged in a conversation. Please answer the question based on the context provided.
        If you're asked about a MemKey value, mention it only if you're certain it's the exact value you were told to remember.
        
        Context:
        {context}
        
        Human: {message}
        AI:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = RunnableSequence(prompt | self.model)

        result = chain.invoke({
            "context": context_str,
            "message": message,
        }, config={"max_tokens": self.max_tokens})

        response_tokens = len(str(result).split())

        new_size = self.context_window_size + message_tokens + response_tokens
        self.context_window_size = min(new_size, self.max_context_window)

        self.used_tokens += message_tokens + response_tokens

        self.context.append(f"AI: {result}")

        if self.memkey:
            memkey_mentioned = self.check_memkey_retention(str(result))
            if memkey_mentioned:
                if self.memkey_first_remembered is None:
                    self.memkey_first_remembered = self.total_interactions
                self.memkey_last_remembered = self.total_interactions
            elif self.memkey_first_remembered is not None and self.total_interactions - self.memkey_last_remembered > 3:
                self.memkey = None

        return str(result)
        
    def check_memkey_retention(self, response):
        if self.memkey in response:
            return True
        
        remembrance_phrases = [
            "The MemKey value is",
            "The MemKey you mentioned is",
            "You asked me to remember the MemKey",
            "The MemKey I was told to remember is"
        ]
        for phrase in remembrance_phrases:
            if phrase in response and self.memkey in response.split(phrase)[1]:
                return True
        
        return False

    def get_memkey_metrics(self):
        base_info = f"MemKey: {self.memkey_value}\n" if self.memkey_value else ""
        if not self.memkey:
            return ("forgotten", f"{base_info}No MemKey set or MemKey forgotten")
        if self.memkey_first_remembered is None:
            return ("set", f"{base_info}Not yet remembered")
        return ("remembered", f"{base_info}First remembered: Interaction {self.memkey_first_remembered}\nLast remembered: Interaction {self.memkey_last_remembered}")

    def get_token_metrics(self):
        return f"Total Tokens: {self.used_tokens}\nTotal Interactions: {self.total_interactions}\nAvg Tokens per Interaction: {self.used_tokens / max(1, self.total_interactions):.2f}"

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
        return f"Max tokens per response set to {max_tokens}"

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
    "Explain the basics of machine learning algorithms.",
    "How does blockchain technology work?",
    "Describe the water cycle and its importance to the environment.",
    "What are the key principles of effective leadership?",
    "Explain the theory of relativity in simple terms.",
    "How do vaccines work to prevent diseases?",
    "Discuss the cultural significance of mythology in ancient civilizations.",
]

def generate_follow_up(response):
    follow_ups = [
        f"How does this relate to real-world applications?",
        "What are some common misconceptions about this topic?",
        "Can you provide a concrete example to illustrate this concept?",
        "How has our understanding of this topic evolved over time?",
        "What are some potential future developments in this area?",
        "How does this compare to similar concepts in other fields?",
        "Are there any ethical considerations related to this topic?",
        "How might this concept impact society in the coming years?",
        "Can you explain this in a way that a beginner would understand?",
        "What are some challenges or limitations in this area?",
        "How do experts in the field typically approach this subject?",
        "Are there any interesting historical anecdotes related to this topic?",
        "How does this concept apply in different cultural contexts?",
        "What are some practical applications of this knowledge?"
    ]
    return random.choice(follow_ups)

def update_memkey_display(memkey_status, memkey_message):
    if memkey_status == "forgotten":
        return gr.update(value=memkey_message, label="⚠️ MemKey Status ⚠️", interactive=False)
    elif memkey_status == "set":
        return gr.update(value=memkey_message, label="MemKey Status", interactive=False)
    else:  # "remembered"
        return gr.update(value=memkey_message, label="MemKey Status", interactive=False)

def run_batch_process(chatbot, status):
    try:
        if not benchmark.batch_running:
            benchmark.batch_running = True
            status = "Initializing batch process..."
            yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))

            if benchmark.last_batch_question is None:
                benchmark.last_batch_question = random.choice(preset_prompts)

            status = f"Starting batch process with prompt: {benchmark.last_batch_question}"
            yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))

        while benchmark.batch_running:
            response = benchmark.chat(benchmark.last_batch_question, chatbot)
            chatbot.append((benchmark.last_batch_question, response))
            status = f"Response received. Total interactions: {benchmark.total_interactions}"
            yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))

            memkey_status, _ = benchmark.get_memkey_metrics()
            if memkey_status == "forgotten":
                status = "MemKey forgotten. Stopping batch process."
                benchmark.batch_running = False
                yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))
                break

            if not benchmark.batch_running:
                status = "Batch processing stopped by user."
                yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))
                break

            benchmark.last_batch_question = generate_follow_up(response)
            status = f"Generated follow-up question: {benchmark.last_batch_question}"
            yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))

        chatbot.append((None, "Batch processing completed."))
        yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))

    except Exception as e:
        error_message = f"Error in batch processing: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_message)
        chatbot.append((None, error_message))
        status = "An error occurred during batch processing. Check the chat for details."
        yield chatbot, status, benchmark.get_token_metrics(), update_memkey_display(*benchmark.get_memkey_metrics()), update_context_window_size(str(benchmark.context_window_size))

def start_batch():
    benchmark.batch_running = True
    if benchmark.last_batch_question is None:
        benchmark.last_batch_question = random.choice(preset_prompts)

def stop_batch():
    benchmark.batch_running = False

with gr.Blocks() as demo:
    gr.Markdown("# AI Context Retention Benchmark")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                memkey_input = gr.Textbox(label="MemKey Value")
                set_memkey_button = gr.Button("Set MemKey")
                memkey_display = gr.Textbox(label="MemKey Status", interactive=False) 
            status = gr.Textbox(label="Batch Status", value="Ready to start batch processing.")  
            chatbot = gr.Chatbot(label="Conversation", height=600)
            msg = gr.Textbox(label="Message")
                                  
            with gr.Row():
                preset_buttons = [gr.Button(prompt) for prompt in preset_prompts]

        with gr.Column(scale=1):
            context_window_size_display = gr.HTML(
                value="<div><strong>Current Context Window Size:</strong><h2 style='text-align: center; font-weight: bold;'>0</h2></div>"
            )
            max_context_window_slider = gr.Slider(0, 128000, value=benchmark.max_context_window, step=100, label="Max Context Window")
            
            gr.Markdown("## Batch Processing")
            token_metrics = gr.Textbox(label="Token Metrics", value=benchmark.get_token_metrics())
            query_frequency_slider = gr.Slider(1, 10, value=benchmark.memkey_query_frequency, step=1, label="MemKey Query Frequency")
            gr.Markdown("*MemKey Query Frequency: How often (in number of interactions) the model is asked about the MemKey.*")
            
            start_batch_button = gr.Button("Start/Resume Batch Processing")
            stop_batch_button = gr.Button("Stop Batch Processing")
            clear = gr.Button("Reset")   
   
            gr.Markdown("## Model Parameters")
            personality = gr.Radio(["Balanced", "Creative", "Precise", "Code-focused"], label="Personality", value="Balanced")
            temperature = gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0, 1, value=1.0, step=0.1, label="Top P")
            max_tokens_per_response = gr.Slider(50, 16384, value=1280, step=10, label="Max Tokens Per Response")

    def update_context_window_size(size):
        return f"<div><strong>Current Context Window Size:</strong><h2 style='text-align: center; font-weight: bold;'>{size}</h2></div>"

    def respond(message, chat_history):
        bot_message = benchmark.chat(message, chat_history)
        chat_history.append((message, bot_message))
        memkey_status, memkey_message = benchmark.get_memkey_metrics()
        return "", chat_history, benchmark.get_token_metrics(), update_memkey_display(memkey_status, memkey_message), update_context_window_size(str(benchmark.context_window_size))


    def reset_all():
        benchmark.reset()
        benchmark.batch_running = False
        benchmark.last_batch_question = None
        memkey_status, memkey_message = benchmark.get_memkey_metrics()
        return [], "Conversation and context reset.", benchmark.get_token_metrics(), "Balanced", 0.7, 1.0, 1280, "Ready to start batch processing.", update_memkey_display(memkey_status, memkey_message), update_context_window_size("0"), benchmark.max_context_window, benchmark.memkey_query_frequency

    def set_memkey(memkey):
        benchmark.set_memkey(memkey)
        memkey_status, memkey_message = benchmark.get_memkey_metrics()
        return update_memkey_display(memkey_status, memkey_message)

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

    msg.submit(respond, [msg, chatbot], [msg, chatbot, token_metrics, memkey_display, context_window_size_display])
    clear.click(reset_all, outputs=[
            chatbot, msg, token_metrics, personality, temperature, top_p, 
            max_tokens_per_response, status, memkey_display, 
            context_window_size_display, max_context_window_slider, 
            query_frequency_slider
        ])
    set_memkey_button.click(set_memkey, inputs=[memkey_input], outputs=[memkey_display])

    personality.change(update_personality, personality, [temperature, top_p])
    temperature.change(benchmark.set_temperature, temperature, None)
    top_p.change(benchmark.set_top_p, top_p, None)
    max_tokens_per_response.change(benchmark.set_max_tokens, max_tokens_per_response, None)
    query_frequency_slider.change(benchmark.set_query_frequency, query_frequency_slider, None)

    for button in preset_buttons:
        button.click(lambda x: x, button, msg).then(
            respond, [msg, chatbot], [msg, chatbot, token_metrics, memkey_display, context_window_size_display]
        )

    max_context_window_slider.change(
        lambda value: setattr(benchmark, 'max_context_window', value),
        inputs=max_context_window_slider,
        outputs=None
    )

    start_batch_button.click(start_batch).then(
        run_batch_process, 
        inputs=[chatbot, status], 
        outputs=[chatbot, status, token_metrics, memkey_display, context_window_size_display]
    )

    stop_batch_button.click(stop_batch)

if __name__ == "__main__":
    demo.launch()