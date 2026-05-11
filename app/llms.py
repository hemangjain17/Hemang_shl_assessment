import os
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

def call_gemini(model_name: str, messages: list, response_schema=None) -> str:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
    
    formatted_messages = []
    # format messages to gemini format
    for m in messages:
        role = 'user' if m['role'] == 'user' else 'model'
        formatted_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=m['content'])]))
    
    config_args = {"temperature": 0.0}
    if response_schema:
        config_args["response_mime_type"] = "application/json"
        config_args["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_args)

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=formatted_messages,
        config=config
    )
    return response.text

def call_groq(model_name: str, messages: list, response_format=None) -> str:
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not set.")
    
    kwargs = {"temperature": 0.0, "max_tokens": 4096}
    if response_format:
        kwargs["response_format"] = response_format
        
    completion = groq_client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )
    return completion.choices[0].message.content

def call_synthesizer(messages: list) -> str:
    # Try Groq first (better structured JSON output)
    try:
        model_name = "openai/gpt-oss-20b"
        print(f"Calling Groq with model: {model_name}")
        return call_groq(model_name, messages, response_format={"type": "json_object"})
    except Exception as e:
        print(f"Groq failed: {e}. Falling back to Gemini 2.5 Pro.")
        model_name = "gemini-2.5-pro"
        return call_gemini(model_name, messages)

def embed_text(texts: list[str]) -> list[list[float]]:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
        
    embeddings = []
    import time
    for text in texts:
        result = gemini_client.models.embed_content(
            model='gemini-embedding-2',
            contents=text,
            config={'output_dimensionality': 768}
        )
        
        if isinstance(result.embeddings, list):
             embeddings.append(result.embeddings[0].values)
        else:
             embeddings.append(result.embeddings.values)
             
        # Add a small delay to avoid rate limits
        time.sleep(0.05)
        
    return embeddings
