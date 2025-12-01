import datetime
import asyncio
from asgiref.sync import sync_to_async

from django_core.config import Config
from rag_service.openai_service import make_openai_request

try:
    from django_core.servvia_prompts import RESPONSE_GEN_PROMPT as SERVVIA_RESPONSE_PROMPT, get_user_profile_context
except Exception:
    SERVVIA_RESPONSE_PROMPT = None
    get_user_profile_context = None


@sync_to_async
def get_user_profile_from_db(email_id):
    """Async-safe profile retrieval"""
    try:
        from user_profile.models import UserProfile
        profile = UserProfile.objects.get(email=email_id)
        return {
            'allergies': profile.get_allergies_list(),
            'medical_conditions': profile.get_conditions_list(),
            'current_medications': profile.get_medications_list(),
            'first_name': profile.first_name,
        }
    except Exception as e:
        print(f"Profile fetch error: {e}")
        return None


async def setup_prompt(user_name, context_chunks, rephrased_query, email_id=None, user_profile=None, system_prompt=Config.RESPONSE_GEN_PROMPT):
    """
    Setup generation response prompt with user profile context
    """
    prompt_name_1 = user_name if user_name else "User"
    prompt_template = SERVVIA_RESPONSE_PROMPT or system_prompt
    
    # Get user profile text
    user_profile_text = "No health profile available."
    
    if user_profile and get_user_profile_context:
        user_profile_text = get_user_profile_context(user_profile)
    elif email_id and get_user_profile_context:
        profile_data = await get_user_profile_from_db(email_id)
        if profile_data:
            user_profile_text = get_user_profile_context(profile_data)
            if not user_name and profile_data.get('first_name'):
                prompt_name_1 = profile_data['first_name']
    
    # Format the prompt
    try:
        response_prompt = prompt_template.format(
            name_1=prompt_name_1,
            context=context_chunks,
            input=rephrased_query,
            user_profile=user_profile_text,
        )
    except KeyError as e:
        print(f"Prompt format error: {e}")
        response_prompt = f"User {prompt_name_1} asked: {rephrased_query}\n\nContext: {context_chunks}\n\nProvide a helpful response:"
    
    return response_prompt


async def generate_query_response(original_query, user_name, context_chunks, rephrased_query, email_id=None, user_profile=None):
    """
    Generate final response with user profile context
    """
    response_map = {}
    llm_response = None
    response_gen_start = None
    response_gen_end = None

    generation_completion_tokens = 0
    generation_prompt_tokens = 0
    generation_total_tokens = 0

    response_gen_exception = None
    response_gen_retries = 0

    response_map.update({
        "response": llm_response,
        "original_query": original_query,
        "rephrased_query": rephrased_query,
        "generation_start_time": response_gen_start,
        "generation_end_time": response_gen_end,
        "completion_tokens": generation_completion_tokens,
        "prompt_tokens": generation_prompt_tokens,
        "total_tokens": generation_total_tokens,
        "response_gen_exception": response_gen_exception,
        "response_gen_retries": response_gen_retries,
    })

    response_prompt = await setup_prompt(user_name, context_chunks, rephrased_query, email_id, user_profile)

    response_gen_start = datetime.datetime.now()
    generated_response, response_gen_exception, response_gen_retries = await make_openai_request(response_prompt)
    response_gen_end = datetime.datetime.now()

    if generated_response:
        llm_response = generated_response.choices[0].message.content
        usage = getattr(generated_response, "usage", None)
        if usage:
            generation_completion_tokens = getattr(usage, "completion_tokens", 0)
            generation_prompt_tokens = getattr(usage, "prompt_tokens", 0)
            generation_total_tokens = getattr(usage, "total_tokens", 0)

    response_map.update({
        "response": llm_response,
        "original_query": original_query,
        "rephrased_query": rephrased_query,
        "generation_start_time": response_gen_start,
        "generation_end_time": response_gen_end,
        "completion_tokens": generation_completion_tokens,
        "prompt_tokens": generation_prompt_tokens,
        "total_tokens": generation_total_tokens,
        "response_gen_exception": response_gen_exception,
        "response_gen_retries": response_gen_retries,
    })

    return response_map
