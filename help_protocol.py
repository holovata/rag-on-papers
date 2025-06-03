import os
import sys
import yaml
import json
import re

from src.data_processing.papers_vectorization import get_top_relevant_documents
from langchain_ollama import OllamaLLM


def refine_query(query: str, prompt_template: str = None) -> dict:
    """
    Refines the user's query using LLM.
    Returns a dict with key "refined_query".
    """
    if prompt_template:
        prompt = prompt_template.format(query=query)
    else:
        prompt = f"""
Given the research query: "{query}",
please rephrase it into a clearer, more formal research question.

Return ONLY a valid JSON object with the following structure:
{{
    "refined_query": "string"
}}

Example:
{{
    "refined_query": "What are the key installation steps for the software?"
}}
"""
    llm = OllamaLLM(model="llama3.2:latest")
    response = llm.invoke(prompt)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group()
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print("Error parsing JSON in refine_query:", e)
        result = {"refined_query": query}
    return result


def answer_query_from_guide(query: str, prompt_template: str = None) -> str:
    """
    Retrieves the top relevant document chunks and then uses an LLM to generate an answer.
    If prompt_template is provided, it is used to build the final prompt.
    """
    top_chunks = get_top_relevant_documents(query, top_n=3,
                                            embeddings_folder=r"C:\Work\diplom2\rag_on_papers\data\user_manual\embeddings")
    if not top_chunks:
        return "No relevant information found in the guide."
    context_text = "\n\n".join(chunk["content"] for chunk in top_chunks)
    llm = OllamaLLM(model="llama3.2:latest")

    if prompt_template:
        # Используем плейсхолдеры {refined_query} и {guide_chunks} из YAML-шаблона.
        prompt = prompt_template.format(refined_query=query, guide_chunks=context_text)
    else:
        prompt = f"""
Based on the following document chunks from the user guide, answer the query:
{query}

Context:
{context_text}

Answer:
        """
    return llm.invoke(prompt)


class UserGuidePipeline:
    def __init__(self, protocol_path):
        self.protocol = self.load_protocol(protocol_path)
        self.context = {}

    def load_protocol(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def resolve_parameters(self, params: dict):
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                var_name = value[2:-2].strip()
                resolved[key] = self.context.get(var_name, value)
            else:
                resolved[key] = value
        return resolved

    def run_pipeline(self, user_query: str):
        self.context["user_query"] = user_query
        for step in self.protocol["pipeline"]:
            stage = step["stage"]
            func_name = step["function"]
            params = self.resolve_parameters(step.get("parameters", {}))
            self.context.update(params)
            print(f"\n=== {stage.upper()} ===")
            print(f"CoT: {step['cot_template'].format(**self.context)}")
            # Вызываем функцию и сохраняем результат
            func = globals().get(func_name) or getattr(self, func_name, None)
            if not func:
                print(f"[ERROR] Function {func_name} not found!")
                continue
            result = func(**params)
            self.context[f"{func_name}_output"] = result

            # Пример обработки для разных этапов:
            if stage == "query_refinement" and isinstance(result, dict):
                if "refined_query" in result:
                    self.context["refined_query"] = result["refined_query"]
                print("\n[Refined Query]")
                print(result.get("refined_query"))
            elif stage == "retrieval" and isinstance(result, list):
                self.context["num_chunks"] = len(result)
                print("\nTOP RELEVANT CHUNKS:")
                for idx, art in enumerate(result[:3], 1):
                    print(f"{idx}. Chunk Index: {art.get('chunk_index')}, Similarity: {art.get('similarity'):.2f}")
                    snippet = art.get("content", "")[:150].replace("\n", " ")
                    print(f"   Content snippet: {snippet}...")
            elif stage == "qa_over_guide":
                print("\nFINAL ANSWER:")
                print(result)
            yield (stage, self.context.copy())
        yield ("done", self.context.copy())


if __name__ == "__main__":
    pipeline = UserGuidePipeline("help_protocol.yaml")
    print("=== QA for User Guide ===")
    user_query = input("Enter your query for the user guide: ").strip()
    output = pipeline.run_pipeline(user_query)
