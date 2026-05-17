import os, re
import asyncio
import datetime
import random
import logging
from openai import AsyncOpenAI, RateLimitError as OAIRateLimitError, APITimeoutError as OAIAPITimeoutError, InternalServerError as OAIInternalServerError
from groq import AsyncGroq, RateLimitError as GroqRateLimitError, APITimeoutError as GroqAPITimeoutError, InternalServerError as GroqInternalServerError

from django_core.config import Config

logger = logging.getLogger("ServVia.LLM")
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from legacy_agriculture.common.constants import Constants


async def _call_groq_fallback(
    prompt_message: str,
    temperature: float = 0,
    model: str = None,
) -> tuple:
    """
    Groq fallback — used when OpenAI fails or quota is exhausted.
    Returns the same (response, exception_string, retries) tuple shape.
    """
    model = model or Config.GROQ_MODEL
    try:
        async with AsyncGroq(api_key=Config.GROQ_API_KEY) as groq_client:
            response = await groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_message}],
                temperature=temperature,
            )
            return response, "", 0
    except Exception as e:
        return None, f"Groq fallback failed: {e}", 0


async def make_openai_request(
    prompt_message,
    model=None,
    temperature=0,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
):
    """
    Make a chat completion request — OpenAI GPT-4o primary, Groq fallback.

    Strategy:
      - Try OpenAI first with exponential backoff on rate limits.
      - If OpenAI fails (rate limit exhausted or error): fall back to Groq.
    """
    oai_model = Config.OAI_MODEL
    # If a Groq model was passed explicitly (legacy call sites), ignore it and use OAI_MODEL.
    # Callers that need Groq explicitly should call _call_groq_fallback directly.

    exception_string = ""
    retries = 0
    delay = initial_delay

    async with AsyncOpenAI(api_key=Config.OPEN_AI_KEY) as oai_client:
        while retries < max_retries:
            try:
                response = await oai_client.chat.completions.create(
                    model=oai_model,
                    messages=[{"role": "user", "content": prompt_message}],
                    temperature=temperature,
                )
                logger.info(f"OpenAI {oai_model} responded successfully.")
                return response, exception_string, retries

            except OAIRateLimitError as e:
                exception_string += str(e) + "\n"
                logger.warning(f"OpenAI rate limit (retry {retries + 1}/{max_retries}): {e}")
                delay *= exponential_base * (1 + jitter * random.random())
                await asyncio.sleep(delay)
                retries += 1

            except (OAIAPITimeoutError, OAIInternalServerError) as e:
                exception_string += str(e) + "\n"
                logger.warning(f"OpenAI error (retry {retries + 1}/{max_retries}): {e}")
                delay *= exponential_base * (1 + jitter * random.random())
                await asyncio.sleep(delay)
                retries += 1

            except Exception as e:
                exception_string += str(e) + "\n"
                logger.error(f"OpenAI unexpected error: {e}")
                break

    # OpenAI failed — fall back to Groq
    logger.warning(f"OpenAI failed after {retries} retries — falling back to Groq ({Config.GROQ_MODEL}).")
    result = await _call_groq_fallback(prompt_message, temperature=temperature)
    if result[0] is None:
        logger.error("Both OpenAI and Groq failed.")
        return None, exception_string + "\n" + result[1], retries
    return result


# Strip chain-of-thought reasoning traces from deepseek-r1 outputs.
_THINK_TAG_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


async def make_reasoner_request(
    prompt_message: str,
    temperature: float = 0.6,
    max_tokens: int = 1500,
) -> str:
    """
    Call the primary LLM for reasoning tasks (OpenAI GPT-4o).
    Falls back to Groq reasoning model if OpenAI fails.
    """
    try:
        async with AsyncOpenAI(api_key=Config.OPEN_AI_KEY) as oai_client:
            response = await oai_client.chat.completions.create(
                model=Config.OAI_MODEL,
                messages=[{"role": "user", "content": prompt_message}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI reasoner failed: {e} — falling back to Groq.")
        try:
            reasoner_model = getattr(Config, "GROQ_REASONER_MODEL", Config.GROQ_MODEL)
            async with AsyncGroq(api_key=Config.GROQ_API_KEY) as groq_client:
                response = await groq_client.chat.completions.create(
                    model=reasoner_model,
                    messages=[{"role": "user", "content": prompt_message}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = response.choices[0].message.content or ""
                return _THINK_TAG_RE.sub("", raw).strip()
        except Exception as fe:
            logger.error(f"Groq reasoner fallback also failed: {fe}")
        return ""
