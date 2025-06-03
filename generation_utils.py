import json
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def generate_clarification(articles, draft, query, prompt_template=None):
    if not draft:
        return {"questions": ["No draft available to clarify."]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    if prompt_template:
        prompt = PromptTemplate.from_template(prompt_template).format(query=query, draft=draft)
    else:
        prompt = (
            f"You are a scientific assistant. Given this draft answer for '{query}', "
            "list up to 3 clarifying questions as JSON."
        )
    ai = llm.invoke([("user", prompt)])
    text = ai.content if hasattr(ai, "content") else ai
    try:
        return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
    except Exception:
        return {"questions": ["Could you specify which aspect needs more detail?"]}


def generate_ideas(articles, query, prompt_template=None):
    if not articles:
        return {"ideas": ["No articles found. Please broaden your query or topic."]}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    if prompt_template:
        prompt = PromptTemplate.from_template(prompt_template).format(query=query, num_articles=len(articles))
    else:
        prompt = f"Generate 3 research ideas as JSON for query: {query}."
    ai = llm.invoke([("user", prompt)])
    text = ai.content if hasattr(ai, "content") else ai
    try:
        return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
    except Exception:
        return {
            "ideas": [
                "Explore interdisciplinary connections with optimization.",
                "Assess historical breakthroughs in this field.",
                "Design a comparative methodology study.",
            ]
        }
