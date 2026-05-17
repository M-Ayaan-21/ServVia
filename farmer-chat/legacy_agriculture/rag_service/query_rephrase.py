"""
Query rephrasing for better retrieval
"""
import logging
from legacy_agriculture.rag_service.openai_service import make_openai_request
from django_core.config import Config

logger = logging.getLogger(__name__)


async def rephrase_query(original_query, chat_history=None):
    """
    Rephrase user query for better vector database retrieval.
    Uses the fast model — query rephrasing is a simple reformulation task.
    """
    try:
        prompt = f"""Convert this health query into concise medical search terms.
Preserve the medical meaning accurately. Do NOT assume the answer should be home remedies.
Output only the search terms, nothing else.

Query: {original_query}

Medical search terms:"""

        response, error, retries = await make_openai_request(
            prompt, model=Config.GROQ_FAST_MODEL
        )
        
        if response and response.choices:
            rephrased = response.choices[0].message.content.strip()
            logger.info(f"Query rephrased: '{original_query}' -> '{rephrased}'")
            return rephrased
        else:
            return original_query
    
    except Exception as e:
        logger.error(f"Query rephrase error: {e}", exc_info=True)
        return original_query
