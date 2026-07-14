import json
from openai import OpenAI

class QueryParser:
    def __init__(self, base_url="http://127.0.0.1:1234/v1", api_key="not-needed"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def parse_query(self, query):
        """
        Parses a natural language query into context and specific fashion items.
        Returns a dict: {"context": "...", "items": ["...", "..."]}
        """
        prompt = f"""You are a specialized fashion query parser. 
Analyze the following user query and extract two things:
1. "context": The environment, location, or general background vibe (e.g. 'office', 'park', 'formal setting', 'casual weekend'). If none, return empty string.
2. "items": A list of specific clothing items and their attributes. CRITICAL: ONLY extract items explicitly mentioned in the query. Do NOT hallucinate or guess items (like 'yellow raincoat' or 'red tie') if they are not literally in the query text. If no specific clothing items are mentioned, return an empty list [].

Return ONLY a valid JSON object in this exact format:
{{"context": "...", "items": ["...", "..."]}}

Query: "{query}"
"""
        try:
            response = self.client.chat.completions.create(
                model="local-model", # The model name doesn't matter for LM Studio/Ollama usually if there's only one loaded
                messages=[
                    {"role": "system", "content": "You are a helpful JSON-only output assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            # Sometimes models wrap json in ```json ... ```
            content = content.replace('```json', '').replace('```', '').strip()
            return json.loads(content)
        except Exception as e:
            print(f"LLM Parsing failed: {e}. Falling back to heuristic.")
            # Fallback heuristic if local LLM is down
            return {"context": query, "items": [query]}
