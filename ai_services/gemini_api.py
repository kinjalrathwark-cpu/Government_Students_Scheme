from google import genai
from dotenv import load_dotenv
import os
from typing import Union,List,Dict,Any
load_dotenv()

def generate_answer(user_message: str, context: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Generates a grounded response to the user's message using retrieved context from ChromaDB.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."

    client = genai.Client(api_key=api_key)
    

    # Format context if passed as retrieved chunks
    if isinstance(context, list):
        if not context:
            return "I couldn't find any relevant admission or syllabus information in the knowledge base for your query."

        formatted_chunks = []
        for i, chunk in enumerate(context):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Document")
            page = meta.get("page", "?")
            text = chunk.get("text", "")
            formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
        context_str = "\n\n".join(formatted_chunks)
    else:
        context_str = str(context)

    prompt = f"""You are a helpful assistant for the Gujarat Student Yojana & Scholarship Guide.
       
     

Context:
{context_str}

User Question:
{user_message}
"""

    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.7-flash")

    try:
        response = client.interactions.create(
            model=model_name,
            input=prompt,
        )
        return response.output_text
    except Exception as e:
        # Fallback to standard generate_content if interactions fails
        try:
            fallback_model = "gemini-2.5-flash"
            fb_response = client.models.generate_content(
                model=fallback_model,
                contents=prompt,
            )
            return fb_response.text
        except Exception as inner_e:
            return f"An error occurred while generating the answer: {str(e)}"

























# def generate_answer(user_message, text):
#     client = genai.client(
#         api_key=os.getenv("genai_api_key")
#     )
#     # client = genai.client(api_key = "AQ.Ab8RN6La6flnngV7OelRYVi3LAsslnIuv2J1X19b6eOjtQjRAA")
#     prompt=f"""
#         You have to generate answer from the given text {text} only.
#         user question: {user_message}"""

#     response = client.interactions.create(
#         model="gemini-3.5-flash-lite",     
#         input=prompt,
#     )
    

#     return response.output_text







