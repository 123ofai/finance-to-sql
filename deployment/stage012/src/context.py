import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import inference_v2
from mlx_lm import load, generate

# ---- Load LLM (quantized recommended) ----
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
"""model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
"""
model, tokenizer = load("microsoft/Phi-3-mini-4k-instruct")


# ---- Session Manager ----
class ChatSession:
    def __init__(self, session_id: str, max_turns: int = 50):
        self.session_id = session_id
        self.history = []   # store {"role": "user"/"assistant", "content": "..."}
        self.max_turns = max_turns

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def build_prompt(self, user_query: str) -> str:
        system = (
            "You are a user query rewriter. Expand the user's latest question into a to include conversation history for context."
            "Don't try to answer the question, instead only rewrite the query!"
            "Also, don't change any factual information on what is being asked. Keep it brief."
        )

        prompt = f"[System]\n{system}\n\n[Conversation History]\n"
        for turn in self.history:
            role = turn['role'].capitalize()
            prompt += f"{role}: {turn['content']}\n"

        prompt += f"\n[Current User Query]\nUser: {user_query}\n\n[Task]\nRewrite the current user query so that it includes context from conversation history.\nAssistant:"
        return prompt

# ---- LLM Helper ----
def rewrite_query(session: ChatSession, user_query: str) -> str:
    prompt = session.build_prompt(user_query)
    rewritten = generate(model, tokenizer, prompt, max_tokens=500)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False
    )
    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
    """
    
    # Extract only assistant part (basic cleanup)
    rewritten = rewritten.split("Assistant:")[-1].strip()
    return rewritten

def generate_reply(history, user_query, rewritten_query, sql_result):
    system = "Summarise the given information and reply in natural language."
    prompt = f"[System]\n{system}\n\n"

    # Include conversation history
    prompt += "[Conversation History]\n"
    for turn in history:
        prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"

    # Add current query + SQL context
    prompt += f"\n[Current Turn]\n"
    prompt += f"User asked: \"{user_query}\"\n"
    prompt += f"Rewritten Query: {rewritten_query}\n"
    prompt += f"SQL Result:\n{sql_result}\n\n"

    prompt += "[Task]\nBased on the query and results, reply in natural, conversational English.\nAssistant:"

    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    """
    output = generate(model, tokenizer, prompt, max_tokens=500)

    return output


# ---- Main loop ----
if __name__ == "__main__":
    #resources = inference_v2.model_fn('../')
    session = ChatSession("test")

    while True:
        user_query = input("User: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        # Step 1: Rewrite with LLM
        rewritten = rewrite_query(session, user_query)
        print(f"[Rewritten Query] {rewritten}")
        print('----------------------')
        input_data = {
            "query": rewritten, 
            "taxonomy": "", 
            "currency": "", 
            "schema": "" , 
            "entity_id":"", 
            "scenario": "",
            "nature": ""
        }

        # Step 2: Feed into your existing SQL pipeline
        #db_result = inference_v2.predict_fn(input_data, resources)
        db_result = '7503734'

        # Step 3: LLM User O/P
        reply = generate_reply(session.history, user_query, rewritten, db_result)
        print(f"[Output] {reply}")


        # Step 4: Update session
        session.add_turn("user", user_query)
        session.add_turn("assistant", rewritten)
        session.add_turn("system response", reply)

