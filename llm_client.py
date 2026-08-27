import os
import json
import logging
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.provider = None
        self.api_key = None
        self.model = None
        self.url = None
        
        if os.environ.get("OPENAI_API_KEY"):
            self.provider = "openai"
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.model = "gpt-4o-mini"
            self.url = "https://api.openai.com/v1/chat/completions"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.provider = "anthropic"
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.model = "claude-3-5-haiku-latest"
            self.url = "https://api.anthropic.com/v1/messages"
        elif os.environ.get("GEMINI_API_KEY"):
            self.provider = "gemini"
            self.api_key = os.environ.get("GEMINI_API_KEY")
            self.model = "gemini-1.5-flash"
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        elif os.environ.get("MISTRAL_API_KEY"):
            self.provider = "mistral"
            self.api_key = os.environ.get("MISTRAL_API_KEY")
            self.model = "mistral-small-latest"
            self.url = "https://api.mistral.ai/v1/chat/completions"
        elif os.environ.get("GROQ_API_KEY"):
            self.provider = "groq"
            self.api_key = os.environ.get("GROQ_API_KEY")
            self.model = "llama-3.1-70b-versatile"
            self.url = "https://api.groq.com/openai/v1/chat/completions"
        elif os.environ.get("DEEPSEEK_API_KEY"):
            self.provider = "deepseek"
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
            self.model = "deepseek-chat"
            self.url = "https://api.deepseek.com/v1/chat/completions"
        else:
            logger.warning("No recognized API key found (OPENAI, ANTHROPIC, GEMINI, MISTRAL, GROQ, DEEPSEEK). LLM calls will fail.")

        if self.provider:
            logger.info(f"Initialized Unified LLMClient using {self.provider.upper()} ({self.model})")

    def _complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        if not self.provider:
            return ""

        try:
            if self.provider in ["openai", "mistral", "groq", "deepseek"]:
                body = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500
                }
                
                if json_mode:
                    if self.provider in ["openai", "mistral", "groq", "deepseek"]:
                        body["response_format"] = {"type": "json_object"}
                
                req = urllib.request.Request(
                    self.url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                )
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
                
            elif self.provider == "anthropic":
                body = {
                    "model": self.model,
                    "max_tokens": 500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
                req = urllib.request.Request(
                    self.url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "x-api-key": self.api_key, 
                        "anthropic-version": "2023-06-01", 
                        "Content-Type": "application/json"
                    }
                )
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
                
            elif self.provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                body = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}
                }
                if json_mode:
                    body["generationConfig"]["responseMimeType"] = "application/json"
                    
                req = urllib.request.Request(
                    self.url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
                
        except Exception as e:
            logger.error(f"LLM API Error ({self.provider}): {e}")
            return ""

    def generate_message(self, strategy: dict, context_bundle: dict) -> str:
        system_prompt = (
            "You are Vera, a business growth assistant. Your goal is to simplify complex problems "
            "and provide actionable, helpful advice for business needs.\n"
            "RULES:\n"
            "- Speak in simple, jargon-free English so anyone can understand easily.\n"
            "- Be direct, helpful, and concise.\n"
            "- Never invent metrics or facts. Only use the provided evidence.\n"
            "- Include exactly one clear Call To Action (CTA).\n"
            f"- Match the tone: {strategy.get('tone', 'simple_english')}\n"
            f"- Use language: {strategy.get('language', 'en')}"
        )
        
        user_prompt = f"""
        Objective: {strategy.get('objective')}
        Recipient: {strategy.get('recipient')}
        Trigger: {strategy.get('trigger')}
        Evidence: {json.dumps(strategy.get('evidence'))}
        Recommended Action: {strategy.get('recommended_action')}
        
        Draft a short WhatsApp message fulfilling this strategy.
        """
        
        logger.info("Calling LLM (%s) for message generation...", self.provider)
        response_text = self._complete(system_prompt, user_prompt, json_mode=False)
        return response_text.strip() if response_text else ""
        
    def classify_intent(self, conversation_history: list, latest_reply: str) -> dict:
        system_prompt = (
            "You are an intent classifier. Given a conversation history and a new reply, "
            "classify the intent of the latest reply into one of these states:\n"
            "INTERESTED, NOT_INTERESTED, OFF_TOPIC, CONFUSED, READY.\n"
            "Return a strictly valid JSON object with 'intent' (string), 'confidence' (float), and 'reasoning' (string).\n"
            "Do not output markdown block ticks, just the raw JSON."
        )
        
        user_prompt = f"""
        Conversation History:
        {json.dumps(conversation_history, indent=2)}
        
        Latest Reply: "{latest_reply}"
        """
        
        fallback = {
            "intent": "INTERESTED",
            "confidence": 0.0,
            "reasoning": "Fallback due to parse error or API failure"
        }
        
        try:
            response_text = self._complete(system_prompt, user_prompt, json_mode=True)
            if not response_text:
                return fallback
                
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            
            valid_intents = ["INTERESTED", "NOT_INTERESTED", "OFF_TOPIC", "CONFUSED", "READY"]
            if result.get("intent") not in valid_intents:
                result["intent"] = "INTERESTED"
                
            return result
        except Exception as e:
            logger.error(f"Error parsing intent classification: {e}")
            fallback["reasoning"] = f"Parse error: {str(e)}"
            return fallback
