import os
from llama_cpp import Llama


class SmallLLM:
    def __init__(self):
        self.model_path = os.path.join("models", "Dolphin3.0-Llama3.1-8B-Q8_0.gguf")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please check the path.")

        print(f"SmallRAG: Loading {os.path.basename(self.model_path)} into RAM...")


        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            verbose=False
        )

    def generate_response(self, prompt, max_tokens=512, stop=["Q:", "\n", "User:"]):
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            echo=False
        )
        return output["choices"][0]["text"].strip()

    def generate_rag_answer(self, query, context):
        system_prompt = (
            "You are SmallRAG, a helpful AI assistant. Use the following pieces of "
            "retrieved context to answer the user's question. If you don't know the "
            "answer based on the context, say that you don't know. Keep it concise.\n\n"
            f"Context: {context}"
        )

        full_prompt = f"{system_prompt}\n\nQuestion: {query}\nAnswer: "

        return self.generate_response(full_prompt)