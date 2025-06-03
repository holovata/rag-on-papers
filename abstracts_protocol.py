import yaml
import json
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pymongo.errors import NetworkTimeout

from src.retrieval.abstracts_retrieval_MONGO import get_top_relevant_articles

load_dotenv()

# ---------- helpers -------------------------------------------------- #
def generate_clarification(articles, draft, query, prompt_template=None):
    """Запитуємо до 3 уточнень від користувача."""
    if not draft:
        return {"questions": ["No draft available to clarify."]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    prompt = PromptTemplate.from_template(prompt_template).format(
        query=query, draft=draft
    )
    resp = llm.invoke([("user", prompt)]).content
    try:
        json_str = resp[resp.find("{") : resp.rfind("}") + 1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        qs = re.findall(r'"([^"]+)"', resp)
        return {"questions": qs[:3] or ["Could you specify which aspect needs more detail?"]}


def generate_ideas(articles, query, prompt_template=None):
    """Повертає {"ideas": [...]} — завжди 3 ідеї."""
    if not articles:
        return {"ideas": ["No articles found. Please broaden your query or topic."]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = (
        PromptTemplate.from_template(prompt_template).format(
            query=query, num_articles=len(articles)
        )
        if prompt_template
        else f"Generate 3 research ideas as JSON for query: {query}."
    )
    resp = llm.invoke([("user", prompt)]).content
    try:
        return json.loads(re.search(r"\{.*\}", resp, re.DOTALL).group())
    except Exception:
        return {
            "ideas": [
                "Explore interdisciplinary connections with optimization.",
                "Assess historical breakthroughs in this field.",
                "Design a comparative methodology study.",
            ]
        }


# ---------- main processor ------------------------------------------ #
class ProtocolProcessor:
    def __init__(self, protocol_path: str, enable_refinement: bool = False):
        self.protocol = self._load_protocol(protocol_path)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.enable_refinement = enable_refinement
        self.context: dict = {}

    @staticmethod
    def _load_protocol(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # -- utils -------------------------------------------------------- #
    def _resolve_parameters(self, params):
        """Підставляємо значення з self.context у {{...}}."""
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                parts = v[2:-2].split(".")
                obj = self.context
                for p in parts:
                    obj = obj.get(p, {})
                resolved[k] = obj or v
            else:
                resolved[k] = v
        return resolved

    @staticmethod
    def _format_articles(arts):
        return "\n".join(
            f"{i + 1}. {a['title']} (authors: {a.get('authors','N/A')})\n"
            f"   Abstract: {a.get('abstract','')[:100]}...\n"
            f"   PDF: {a.get('pdf_url','')}"
            for i, a in enumerate(arts)
        )

    # -- pipeline ----------------------------------------------------- #
    def run_pipeline(self, user_query: str):
        self.context = {"user_query": user_query, "first_query": user_query}

        for step in self.protocol["pipeline"]:
            stage = step["stage"]
            # Пропускаємо refinement-стадії, якщо вимкнено
            if not self.enable_refinement and stage in ("clarification", "retrieval_refined", "second_synthesis"):
                continue

            params = self._resolve_parameters(step.get("parameters", {}))
            self.context.update(params)

            # Печатаем логи
            print(step.get("actions", "").format(**self.context))

            # ---------- RETRIEVAL (початковий або refined) -------------
            if stage == "retrieval":
                # Здесь params содержит keys: query, top_n, min_similarity
                arts = self._safe_retrieve(params)
                self.context["initial_articles"] = arts
                self.context.update(
                    {
                        "num_articles":   len(arts),
                        "articles_data":  self._format_articles(arts),
                    }
                )

            elif stage == "retrieval_refined":
                arts = self._safe_retrieve(params)
                self.context["refined_articles"] = arts
                self.context.update(
                    {
                        "num_articles":   len(arts),
                        "articles_data":  self._format_articles(arts),
                    }
                )

            # ---------- FIRST SYNTHESIS --------------------------------
            elif stage == "first_synthesis":
                prompt = PromptTemplate.from_template(step["prompt_template"]).format(
                    **self.context
                )
                self.context["first_synthesis_output"] = self._stream_llm(prompt)

            # ---------- CLARIFICATION ---------------------------------
            elif stage == "clarification":
                clar = generate_clarification(
                    self.context["initial_articles"],
                    self.context.get("first_synthesis_output", ""),
                    user_query,
                    prompt_template=step["prompt_template"],
                )
                for q in clar.get("questions", []):
                    print(" •", q)
                extra = input("\nYour answer (or Enter to skip): ").strip()

                self.context.update(
                    {
                        "extra":         extra,
                        "refined_query": f"{user_query} {extra}" if extra else user_query,
                    }
                )

            # ---------- SECOND SYNTHESIS -------------------------------
            elif stage == "second_synthesis":
                # Обʼєднані статті без дублів
                all_articles = self._merge_unique(
                    self.context.get("initial_articles", []),
                    self.context.get("refined_articles", []),
                )
                self.context.update(
                    {
                        "num_articles":  len(all_articles),
                        "articles_data": self._format_articles(all_articles),
                    }
                )
                prompt = PromptTemplate.from_template(step["prompt_template"]).format(
                    **self.context
                )
                self.context["second_synthesis_output"] = self._stream_llm(prompt)

            # ---------- IDEATION (після першої або другої синтез) ------
            elif stage == "ideation":
                # Если был refinement — берём refined_query, иначе user_query
                query_for_ideas = self.context.get("refined_query", user_query)
                arts_for_ideas = (
                    self.context.get("refined_articles")
                    or self.context.get("initial_articles")
                    or []
                )
                ideas = generate_ideas(
                    arts_for_ideas, query_for_ideas, prompt_template=step["prompt_template"]
                )
                self.context["ideas_output"] = ideas
                print("\nIDEAS:", json.dumps(ideas, ensure_ascii=False, indent=2))

        return self.context

    # -- helpers ------------------------------------------------------ #
    def _safe_retrieve(self, params):
        """
        Обёртка вокруг get_top_relevant_articles.
        Ловим NetworkTimeout или другой Exception и возвращаем [].
        """
        try:
            arts = get_top_relevant_articles(
                query=params.get("query"),
                top_n=params.get("top_n", 3),
                min_similarity=params.get("min_similarity", 0.0),
                num_candidates=params.get("num_candidates", 150)
            )
        except NetworkTimeout as e:
            print(f"[retrieval] NetworkTimeout: {e}")
            arts = []
        except Exception as e:
            print(f"[retrieval] Error: {e}")
            arts = []

        for i, a in enumerate(arts[:5], 1):
            print(f"{i}. {a['id']} (sim={a.get('similarity', 0.0):.2f})")
        return arts

    def _stream_llm(self, prompt):
        """
        Стримим ответ LLM, печатаем построчно и накапливаем в одну строку.
        """
        streamed = ""
        for chunk in self.llm.stream([("user", prompt)]):
            print(chunk.content, end="", flush=True)
            streamed += chunk.content
        print()
        return streamed

    @staticmethod
    def _merge_unique(list_a, list_b):
        """
        Объединяет два списка статей, убирая дубликаты по полю "id".
        """
        seen, merged = set(), []
        for art in list_a + list_b:
            if art["id"] not in seen:
                seen.add(art["id"])
                merged.append(art)
        return merged


# -------------------------------------------------------------------- #
if __name__ == "__main__":
    enable = input("Enable refinement mode? (y/n): ").lower().startswith("y")
    processor = ProtocolProcessor("abstracts_protocol.yaml", enable_refinement=enable)
    q = input("Enter your research query: ").strip()
    processor.run_pipeline(q)
