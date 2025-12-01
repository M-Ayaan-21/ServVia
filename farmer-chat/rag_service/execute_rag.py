import asyncio
import logging
from datetime import datetime

from generation.generate_response import generate_query_response
from rag_service.content_retrieval import retrieve_content
from rag_service.query_rephrase import rephrase_query

logger = logging.getLogger(__name__)


def execute_rag_pipeline(
    query_in_english,
    input_language_detected,
    email_id,
    user_name=None,
    message_id=None,
    chat_history=None,
    user_profile=None,
):
    """
    Execute the complete RAG pipeline with user profile integration
    
    Args:
        query_in_english: User query in English
        input_language_detected: Detected language code
        email_id: User email
        user_name: User's name
        message_id: Unique message ID
        chat_history: Previous chat messages
        user_profile: User health profile dict with allergies, conditions, medications
    
    Returns:
        Tuple of (response_map, message_data)
    """
    response_map = {}
    message_data_update = {}
    
    original_query = query_in_english
    rephrased_query = query_in_english
    
    retrieval_start = None
    retrieval_end = None
    retrieved_chunks = {}
    context_chunks = ""
    
    logger.info(f"Starting RAG pipeline for: {query_in_english}")
    
    try:
        # Step 1: Rephrase query for better retrieval
        logger.info("Step 1: Rephrasing query...")
        rephrased_query = asyncio.run(
            rephrase_query(original_query, chat_history or [])
        )
        logger.info(f"Rephrased query: {rephrased_query}")
        
    except Exception as e:
        logger.error(f"Query rephrasing failed: {e}", exc_info=True)
        rephrased_query = original_query
    
    try:
        # Step 2: Retrieve relevant content chunks
        logger.info("Step 2: Retrieving content from vector database...")
        retrieval_start = datetime.now()
        
        retrieved_chunks = retrieve_content(
            rephrased_query,
            email_id,
            top_k=10
        )
        
        retrieval_end = datetime.now()
        
        if retrieved_chunks and 'chunks' in retrieved_chunks:
            logger.info(f"Retrieved {len(retrieved_chunks['chunks'])} chunks")
            
            # Combine chunks into context string
            context_chunks = "\n\n".join([
                chunk.get('text', '') 
                for chunk in retrieved_chunks['chunks'][:5]  # Use top 5 chunks
            ])
            
            logger.info(f"Context length: {len(context_chunks)} characters")
        else:
            logger.warning("No content retrieved from vector database")
            context_chunks = ""
        
    except Exception as e:
        logger.error(f"Content retrieval failed: {e}", exc_info=True)
        retrieval_end = datetime.now()
        retrieved_chunks = {}
        context_chunks = ""
    
    # Update response map with retrieval data
    response_map.update({
        'retrieval_start': retrieval_start,
        'retrieval_end': retrieval_end,
        'retrieved_chunks': retrieved_chunks,
    })
    
    try:
        # Step 3: Generate response with user profile
        logger.info("Step 3: Generating personalized response with OpenAI...")
        
        if user_profile:
            logger.info(f"Using profile for {user_name}: {user_profile.get('allergies', [])}")
        
        generated_response = asyncio.run(
            generate_query_response(
                original_query,
                user_name,
                context_chunks,
                rephrased_query,
                email_id,
                user_profile,
            )
        )
        
        final_response = generated_response.get('response')
        
        if final_response:
            logger.info(f"Response generated successfully: {len(final_response)} characters")
        else:
            logger.warning("No response generated")
            final_response = "I apologize, but I couldn't generate a response. Please try again."
        
        # Update response map with generation data
        response_map.update({
            'generated_final_response': final_response,
            'generation_start_time': generated_response.get('generation_start_time'),
            'generation_end_time': generated_response.get('generation_end_time'),
            'completion_tokens': generated_response.get('completion_tokens', 0),
            'prompt_tokens': generated_response.get('prompt_tokens', 0),
            'total_tokens': generated_response.get('total_tokens', 0),
        })
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)
        response_map.update({
            'generated_final_response': "I'm having trouble generating a response right now. Please try again.",
            'generation_error': str(e),
        })
    
    # Prepare message data for database (future use)
    message_data_update = {
        'message_id': message_id,
        'query': original_query,
        'rephrased_query': rephrased_query,
        'response': response_map.get('generated_final_response'),
        'retrieval_time': (retrieval_end - retrieval_start).total_seconds() if retrieval_start and retrieval_end else 0,
        'chunks_retrieved': len(retrieved_chunks.get('chunks', [])) if retrieved_chunks else 0,
    }
    
    logger.info("RAG pipeline completed successfully")
    
    return response_map, message_data_update
