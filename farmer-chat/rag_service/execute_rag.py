"""
ServVia 2.0 - Agentic RAG Pipeline (Production Ready)
======================================================
Intelligent health assistant pipeline with:
- Contextual conversation memory
- Drug-herb interaction safety
- Evidence-based verification
- Chronobiological recommendations

Author: ServVia Team
Version: 2.0. 0
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from generation.generate_response import generate_query_response
from rag_service.content_retrieval import retrieve_content
from rag_service.query_rephrase import rephrase_query

logger = logging. getLogger(__name__)

# Import conversation manager
try:
    from servvia2. conversation. manager import conversation_manager
    CONVERSATION_ENABLED = True
except ImportError:
    conversation_manager = None
    CONVERSATION_ENABLED = False
    logger.warning("Conversation manager not available")


# =============================================================================
# EMERGENCY DETECTION SYSTEM
# =============================================================================

EMERGENCY_RESPONSES = {
    'cardiac_arrest': """🚨 **EMERGENCY - CARDIAC ARREST / NOT BREATHING**

**CALL 112 (India) / 911 (US) / 999 (UK) IMMEDIATELY**

**CPR Steps (Hands-Only for untrained):**

1. **CHECK** - Tap shoulders firmly, shout "Are you OK?"
2. **CALL** - If no response, call emergency services immediately
3. **PUSH** - Start chest compressions:
   - Place heel of hand on center of chest
   - Push hard and fast (at least 2 inches deep)
   - Rate: 100-120 compressions per minute
   - Allow full chest recoil between compressions

4. **If trained in CPR:**
   - After 30 compressions, give 2 rescue breaths
   - Tilt head back, lift chin, pinch nose
   - Breathe into mouth until chest rises

5. **CONTINUE** until help arrives or person responds

**🔴 This is a life-threatening emergency. Every second counts.**""",

    'choking': """🚨 **EMERGENCY - CHOKING**

**CALL 112 (India) / 911 (US) / 999 (UK)**

**For ADULTS - Heimlich Maneuver:**

1. Stand behind the person
2. Make a fist with one hand
3. Place fist just above the belly button
4. Grasp fist with other hand
5. Give quick, upward thrusts
6.  Repeat until object is expelled

**For INFANTS (under 1 year):**
1. Place face-down on your forearm
2. Give 5 back blows between shoulder blades
3. Turn over, give 5 chest thrusts
4. Repeat until object comes out

**If person becomes unconscious, start CPR.**

**🔴 This is a life-threatening emergency.**""",

    'cardiac': """🚨 **EMERGENCY - POSSIBLE HEART ATTACK**

**CALL 112 (India) / 911 (US) / 999 (UK) IMMEDIATELY**

**While waiting for help:**

1. Have the person sit or lie in a comfortable position
2.  Loosen any tight clothing
3. If available and not allergic, give aspirin (325mg, chew don't swallow)
4. Stay calm and reassure the person
5. Be prepared to perform CPR if they become unresponsive
6. Do NOT let them walk or exert themselves

**Warning signs:**
- Chest pain or pressure
- Pain spreading to arm, jaw, or back
- Shortness of breath
- Cold sweat, nausea
- Lightheadedness

**🔴 Do NOT drive yourself to the hospital. Wait for ambulance.**""",

    'stroke': """🚨 **EMERGENCY - POSSIBLE STROKE**

**CALL 112 (India) / 911 (US) / 999 (UK) IMMEDIATELY**

**Remember F.A.S. T. :**

- **F**ace: Ask them to smile.  Does one side droop?
- **A**rms: Ask them to raise both arms. Does one drift down?
- **S**peech: Ask them to repeat a phrase. Is it slurred? 
- **T**ime: If ANY of these signs, call emergency immediately! 

**While waiting:**
1. Note the TIME symptoms started (critical for treatment)
2. Keep them calm and lying down
3. Do NOT give food, water, or medication
4.  Loosen tight clothing
5. If unconscious, place in recovery position

**🔴 Time is brain.  Every minute matters.**""",

    'mental_health': """🚨 **You Are Not Alone - Help Is Available**

**Please reach out to someone right now:**

**Crisis Helplines:**
- 🇮🇳 India: iCall - 9152987821 | Vandrevala Foundation - 1860-2662-345
- 🇺🇸 USA: 988 (Suicide & Crisis Lifeline)
- 🇬🇧 UK: 116 123 (Samaritans)
- 🌍 International: findahelpline.com

**If you or someone is in immediate danger:**
Call 112 (India) / 911 (US) / 999 (UK)

**Remember:**
- This feeling is temporary
- You matter and your life has value
- Professional help works - millions have recovered
- Reaching out is a sign of strength, not weakness

**Please talk to someone.  We care about you.  💙**""",

    'poisoning': """🚨 **EMERGENCY - POISONING / OVERDOSE**

**CALL POISON CONTROL IMMEDIATELY:**
- 🇮🇳 India: 1800-11-6117 (AIIMS)
- 🇺🇸 USA: 1-800-222-1222
- 🇬🇧 UK: 111

**Do NOT:**
- Induce vomiting unless told to by poison control
- Give anything to eat or drink
- Wait for symptoms to appear

**Do:**
1. Call poison control immediately
2. Have the container/substance ready to describe
3. Know the person's age and weight
4. Note the time of exposure
5. If on skin, remove clothing and rinse with water
6. If in eyes, rinse with water for 15-20 minutes

**🔴 This is a medical emergency. Call immediately.**""",

    'allergic_reaction': """🚨 **EMERGENCY - SEVERE ALLERGIC REACTION (ANAPHYLAXIS)**

**CALL 112 (India) / 911 (US) / 999 (UK)**

**If person has an EpiPen:**
1. Remove blue safety cap
2. Press orange tip firmly into outer thigh
3.  Hold for 10 seconds
4.  Note the time

**While waiting for help:**
1.  Have person lie down with legs elevated
2.  Loosen tight clothing
3. If vomiting, turn on side
4. Stay with them constantly
5. Be ready to perform CPR

**Signs of anaphylaxis:**
- Difficulty breathing, wheezing
- Swelling of face, lips, tongue
- Rapid heartbeat
- Dizziness or fainting
- Hives or rash

**🔴 Anaphylaxis can be fatal within minutes. Call immediately.**""",

    'severe_bleeding': """🚨 **EMERGENCY - SEVERE BLEEDING**

**CALL 112 (India) / 911 (US) / 999 (UK)**

**Immediate steps:**

1. **Apply direct pressure** - Use clean cloth, press firmly
2. **Don't remove the cloth** - Add more layers on top if soaked
3. **Elevate** - Raise injured area above heart level if possible
4. **Apply pressure to pressure points** if direct pressure doesn't work
5. **Use tourniquet** only as last resort for life-threatening limb bleeding

**Do NOT:**
- Remove embedded objects
- Apply tourniquet unless trained and necessary
- Stop applying pressure to check the wound

**🔴 Call emergency services immediately for severe bleeding.**""",
}


def check_emergency(query: str) -> Optional[str]:
    """
    Check if query indicates an emergency situation. 
    Returns emergency type or None. 
    """
    query_lower = query. lower()
    
    emergency_keywords = {
        # Cardiac arrest / breathing
        'cardiac_arrest': ['cpr', 'not breathing', 'stopped breathing', 'no pulse', 'unconscious not breathing'],
        'choking': ['choking', 'cant breathe', "can't breathe", 'something stuck throat', 'heimlich'],
        'cardiac': ['heart attack', 'chest pain', 'chest pressure', 'heart pain'],
        'stroke': ['stroke', 'face drooping', 'arm weakness', 'slurred speech', 'sudden confusion'],
        'mental_health': ['suicide', 'kill myself', 'want to die', 'end my life', 'self harm', 'hurt myself'],
        'poisoning': ['poisoning', 'overdose', 'swallowed poison', 'took too many pills', 'drank bleach'],
        'allergic_reaction': ['anaphylaxis', 'allergic reaction severe', 'cant breathe allergy', 'throat closing'],
        'severe_bleeding': ['severe bleeding', 'wont stop bleeding', 'blood everywhere', 'arterial bleeding'],
    }
    
    for emergency_type, keywords in emergency_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                return emergency_type
    
    return None


def get_emergency_response(emergency_type: str) -> str:
    """Get the appropriate emergency response"""
    return EMERGENCY_RESPONSES.get(
        emergency_type, 
        """🚨 **EMERGENCY DETECTED**

**CALL 112 (India) / 911 (US) / 999 (UK)**

Please describe the emergency to the dispatcher. 
Stay calm and follow their instructions.

**This is a medical emergency. Professional help is essential.**"""
    )


# =============================================================================
# DRUG-HERB INTERACTION DATABASE
# =============================================================================

INTERACTION_DATABASE = {
    'ginger': {
        'drugs': ['aspirin', 'ibuprofen', 'warfarin', 'blood thinner', 'coumadin', 'plavix', 'clopidogrel', 'anticoagulant'],
        'severity': 'HIGH',
        'reason': 'Ginger inhibits platelet aggregation (blood thinning effect).  Combined with anticoagulants, this significantly increases bleeding risk - bruising, prolonged bleeding from cuts, or internal bleeding.',
        'alternatives': ['Peppermint oil (topical)', 'Lavender aromatherapy', 'Cold compress', 'Chamomile tea'],
    },
    'turmeric': {
        'drugs': ['aspirin', 'warfarin', 'blood thinner', 'coumadin', 'metformin', 'diabetes medication', 'insulin'],
        'severity': 'HIGH',
        'reason': 'Curcumin has antiplatelet effects and can lower blood sugar. Risk of bleeding with anticoagulants and hypoglycemia with diabetes medications.',
        'alternatives': ['Boswellia (for inflammation)', 'Cold compress', 'Rest and elevation', 'Omega-3 foods'],
    },
    'garlic': {
        'drugs': ['aspirin', 'warfarin', 'blood thinner', 'hiv medication', 'saquinavir'],
        'severity': 'MODERATE',
        'reason': 'Garlic has blood-thinning properties. Use only small culinary amounts with blood thinners.',
        'alternatives': ['Onion (milder effect)', 'Oregano', 'Thyme'],
    },
    'ashwagandha': {
        'drugs': ['thyroid medication', 'levothyroxine', 'synthroid', 'sedative', 'benzodiazepine', 'immunosuppressant'],
        'severity': 'HIGH',
        'reason': 'Ashwagandha stimulates thyroid function and has sedative properties. May interfere with thyroid medication dosing and compound sedative effects.',
        'alternatives': ['Chamomile tea (for stress)', 'Lavender aromatherapy', 'Deep breathing exercises', 'Brahmi'],
    },
    'licorice': {
        'drugs': ['blood pressure medication', 'bp medicine', 'antihypertensive', 'diuretic', 'digoxin', 'heart medication'],
        'severity': 'HIGH',
        'reason': 'Glycyrrhizin in licorice raises blood pressure and depletes potassium.  Counteracts BP medications and can cause dangerous heart rhythms with digoxin.',
        'alternatives': ['Honey (for sore throat)', 'Slippery elm', 'Marshmallow root'],
    },
    'ginseng': {
        'drugs': ['warfarin', 'blood thinner', 'diabetes medication', 'metformin', 'insulin', 'antidepressant', 'maoi'],
        'severity': 'MODERATE',
        'reason': 'Ginseng affects blood clotting and blood sugar levels. Has stimulant properties that may interact with MAOIs.',
        'alternatives': ['Green tea (moderate amounts)', 'Peppermint tea', 'Amla'],
    },
    'st johns wort': {
        'drugs': ['antidepressant', 'ssri', 'birth control', 'contraceptive', 'hiv medication', 'immunosuppressant', 'warfarin'],
        'severity': 'CRITICAL',
        'reason': "St. John's Wort induces liver enzymes (CYP450), dramatically reducing effectiveness of many medications. Risk of serotonin syndrome with SSRIs.",
        'alternatives': ['Lavender', 'Chamomile', 'Exercise', 'Light therapy'],
    },
    'valerian': {
        'drugs': ['sedative', 'benzodiazepine', 'sleep medication', 'ambien', 'alcohol'],
        'severity': 'HIGH',
        'reason': 'Valerian has sedative effects that compound with other CNS depressants, risking over-sedation or respiratory depression.',
        'alternatives': ['Chamomile tea', 'Warm milk', 'Lavender aromatherapy', 'Sleep hygiene practices'],
    },
    'kava': {
        'drugs': ['alcohol', 'sedative', 'benzodiazepine', 'antidepressant', 'levodopa'],
        'severity': 'CRITICAL',
        'reason': 'Kava has significant hepatotoxicity risk and compounds with other sedatives. Can interfere with dopamine medications.',
        'alternatives': ['Chamomile', 'Passionflower', 'Lavender'],
    },
    'ginkgo': {
        'drugs': ['aspirin', 'warfarin', 'blood thinner', 'nsaid', 'ibuprofen'],
        'severity': 'HIGH',
        'reason': 'Ginkgo inhibits platelet activating factor, increasing bleeding risk with anticoagulants.',
        'alternatives': ['Brahmi (for cognitive support)', 'Green tea', 'Omega-3 fatty acids'],
    },
    'echinacea': {
        'drugs': ['immunosuppressant', 'cyclosporine', 'corticosteroid'],
        'severity': 'MODERATE',
        'reason': 'Echinacea stimulates the immune system, potentially counteracting immunosuppressive therapy.',
        'alternatives': ['Vitamin C foods', 'Zinc lozenges', 'Rest and hydration'],
    },
}


def check_interactions(herbs: List[str], medications: List[str]) -> List[Dict]:
    """
    Check for drug-herb interactions.
    Returns list of interaction warnings. 
    """
    warnings = []
    
    if not herbs or not medications:
        return warnings
    
    for herb in herbs:
        herb_lower = herb. lower(). strip()
        
        if herb_lower in INTERACTION_DATABASE:
            interaction_data = INTERACTION_DATABASE[herb_lower]
            
            for med in medications:
                med_lower = med.lower().strip()
                
                for dangerous_drug in interaction_data['drugs']:
                    if dangerous_drug in med_lower or med_lower in dangerous_drug:
                        warnings.append({
                            'herb': herb,
                            'medication': med,
                            'severity': interaction_data['severity'],
                            'reason': interaction_data['reason'],
                            'alternatives': interaction_data['alternatives'],
                        })
                        break  # Found match, move to next medication
    
    return warnings


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_entities(query: str) -> Dict[str, List[str]]:
    """
    Extract health conditions, herbs, and medications from query.
    """
    query_lower = query. lower()
    
    # Health conditions
    conditions = []
    condition_keywords = {
        'headache': ['headache', 'head hurts', 'head pain', 'head ache', 'migraine'],
        'fever': ['fever', 'temperature', 'feverish', 'high temperature'],
        'cold': ['cold', 'runny nose', 'sneezing', 'stuffy nose', 'congestion'],
        'cough': ['cough', 'coughing', 'dry cough', 'wet cough'],
        'nausea': ['nausea', 'nauseous', 'feeling sick', 'queasy', 'want to vomit'],
        'indigestion': ['indigestion', 'bloating', 'gas', 'stomach upset', 'acidity', 'heartburn', 'acid reflux'],
        'sore throat': ['sore throat', 'throat pain', 'throat hurts', 'scratchy throat'],
        'anxiety': ['anxiety', 'anxious', 'worried', 'nervous', 'panic'],
        'stress': ['stress', 'stressed', 'overwhelmed', 'tension'],
        'insomnia': ['insomnia', 'cant sleep', "can't sleep", 'trouble sleeping', 'sleepless'],
        'fatigue': ['fatigue', 'tired', 'exhausted', 'no energy', 'weakness'],
        'joint pain': ['joint pain', 'arthritis', 'joints hurt', 'knee pain', 'joint ache'],
        'back pain': ['back pain', 'backache', 'back hurts', 'lower back'],
        'toothache': ['toothache', 'tooth pain', 'tooth hurts', 'dental pain'],
        'acne': ['acne', 'pimples', 'breakout', 'zits', 'blemishes'],
        'constipation': ['constipation', 'constipated', 'cant poop'],
        'diarrhea': ['diarrhea', 'loose stools', 'upset stomach'],
    }
    
    for condition, keywords in condition_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                if condition not in conditions:
                    conditions. append(condition)
                break
    
    # Herbs
    herbs = []
    herb_list = [
        'ginger', 'turmeric', 'peppermint', 'garlic', 'honey', 'tulsi', 'basil',
        'ashwagandha', 'chamomile', 'cinnamon', 'clove', 'licorice', 'ginseng',
        'valerian', 'neem', 'amla', 'fennel', 'cumin', 'coriander', 'fenugreek',
        'mint', 'lavender', 'eucalyptus', 'tea tree', 'aloe vera', 'coconut oil',
        'ginkgo', 'echinacea', 'elderberry', 'brahmi', 'giloy', 'triphala',
        'moringa', 'shatavari', 'ajwain', 'cardamom', 'black pepper',
    ]
    
    for herb in herb_list:
        if herb in query_lower:
            herbs. append(herb)
    
    # Medications
    medications = []
    medication_keywords = {
        'aspirin': ['aspirin', 'disprin', 'ecosprin'],
        'ibuprofen': ['ibuprofen', 'advil', 'motrin', 'brufen'],
        'paracetamol': ['paracetamol', 'acetaminophen', 'tylenol', 'crocin', 'dolo'],
        'warfarin': ['warfarin', 'coumadin'],
        'blood thinner': ['blood thinner', 'blood thinners', 'anticoagulant'],
        'metformin': ['metformin', 'glycomet', 'glucophage'],
        'insulin': ['insulin'],
        'diabetes medication': ['diabetes medication', 'diabetes medicine', 'sugar medicine'],
        'blood pressure medication': ['blood pressure', 'bp medicine', 'bp medication', 'antihypertensive'],
        'thyroid medication': ['thyroid', 'levothyroxine', 'synthroid', 'thyroxine', 'eltroxin'],
        'antidepressant': ['antidepressant', 'ssri', 'prozac', 'zoloft', 'lexapro'],
        'sedative': ['sedative', 'sleeping pill', 'sleep medication', 'benzodiazepine'],
        'statin': ['statin', 'atorvastatin', 'cholesterol medicine'],
        'pan d': ['pan d', 'pantoprazole', 'pantop', 'ppi'],
        'omeprazole': ['omeprazole', 'omez', 'prilosec'],
    }
    
    for med_name, keywords in medication_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                if med_name not in medications:
                    medications.append(med_name)
                break
    
    return {
        'conditions': conditions,
        'herbs': herbs,
        'medications': medications,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def execute_rag_pipeline(
    query_in_english: str,
    input_language_detected: str,
    email_id: str,
    user_name: str = None,
    message_id: str = None,
    chat_history: List = None,
    user_profile: Dict = None,
) -> Tuple[Dict, Dict]:
    """
    Main RAG pipeline execution. 
    
    This is the core function that:
    1.  Checks for emergencies
    2.  Manages conversation context
    3.  Detects drug-herb interactions
    4. Retrieves relevant knowledge
    5.  Generates safe, personalized responses
    6. Validates with Trust Engine
    7. Adds chronobiological context
    
    Args:
        query_in_english: User's query (translated to English if needed)
        input_language_detected: Detected input language
        email_id: User's email for context tracking
        user_name: User's name for personalization
        message_id: Message ID for logging
        chat_history: Previous chat history
        user_profile: User's health profile (allergies, conditions, medications)
    
    Returns:
        Tuple of (response_map, message_data_update)
    """
    
    response_map = {}
    original_query = query_in_english
    retrieval_start = datetime.now()
    retrieval_end = datetime.now()
    
    # ==========================================================================
    # LOGGING
    # ==========================================================================
    logger.info("=" * 70)
    logger. info("ServVia 2.0 Agentic Pipeline Started")
    logger. info(f"Query: {original_query}")
    logger. info(f"User: {email_id}")
    logger.info("=" * 70)
    
    # ==========================================================================
    # EXTRACT USER PROFILE DATA
    # ==========================================================================
    allergies = []
    profile_conditions = []
    profile_medications = []
    
    if user_profile:
        allergies = user_profile.get('allergies', []) or []
        profile_conditions = user_profile.get('medical_conditions', []) or []
        profile_medications = user_profile.get('current_medications', []) or []
    
    logger.info(f"Profile - Allergies: {allergies}, Conditions: {profile_conditions}, Meds: {profile_medications}")
    
    # ==========================================================================
    # STEP 1: EMERGENCY DETECTION
    # ==========================================================================
    emergency_type = check_emergency(original_query)
    
    if emergency_type:
        logger. info(f"🚨 EMERGENCY DETECTED: {emergency_type}")
        
        emergency_response = get_emergency_response(emergency_type)
        
        response_map['generated_final_response'] = emergency_response
        response_map['intent'] = 'emergency'
        response_map['emergency_type'] = emergency_type
        
        return response_map, {
            'message_id': message_id,
            'query': original_query,
            'response': emergency_response,
            'intent': 'emergency',
        }
    
    # ==========================================================================
    # STEP 2: EXTRACT ENTITIES FROM CURRENT QUERY
    # ==========================================================================
    current_entities = extract_entities(original_query)
    logger.info(f"Extracted entities: {current_entities}")
    
    # ==========================================================================
    # STEP 3: UPDATE CONVERSATION CONTEXT
    # ==========================================================================
    context_changes = {'added': [], 'removed': []}
    accumulated_context = {'herbs': [], 'medications': [], 'conditions': []}
    
    if CONVERSATION_ENABLED and conversation_manager and email_id:
        # Update context (handles additions and removals)
        context_changes = conversation_manager.update_context(email_id, original_query)
        
        # Store user message
        conversation_manager.add_message(email_id, 'user', original_query)
        
        # Get accumulated context
        accumulated_context = conversation_manager.get_context(email_id)
        
        if context_changes. get('removed'):
            logger.info(f"🔄 User STOPPED: {context_changes['removed']}")
        if context_changes.get('added'):
            logger.info(f"➕ User ADDED: {context_changes['added']}")
    
    # ==========================================================================
    # STEP 4: MERGE ALL CONTEXT
    # ==========================================================================
    
    # Get stopped medications for filtering
    stopped_medications = []
    if context_changes.get('removed'):
        stopped_medications = [
            r. replace('medication: ', '').lower()
            for r in context_changes['removed']
            if 'medication:' in r
        ]
    
    # Merge herbs
    all_herbs = list(set(
        accumulated_context. get('herbs', []) +
        current_entities['herbs']
    ))
    
    # Merge medications (excluding stopped ones, including profile meds)
    all_medications = list(set(
        accumulated_context.get('medications', []) +
        current_entities['medications'] +
        profile_medications
    ))
    
    # Merge conditions
    all_conditions = list(set(
        accumulated_context.get('conditions', []) +
        current_entities['conditions']
    ))
    
    # Determine CURRENT condition (priority: current query > most recent)
    current_condition = None
    if current_entities['conditions']:
        current_condition = current_entities['conditions'][0]
    elif all_conditions:
        current_condition = all_conditions[-1]  # Most recent
    
    logger.info(f"📋 All herbs: {all_herbs}")
    logger. info(f"📋 All medications: {all_medications}")
    logger.info(f"📋 Stopped medications: {stopped_medications}")
    logger.info(f"📋 All conditions: {all_conditions}")
    logger.info(f"📋 Current condition: {current_condition}")
    
    # ==========================================================================
    # STEP 5: CHECK DRUG-HERB INTERACTIONS
    # ==========================================================================
    interaction_warnings = []
    safety_instructions = ""
    
    if all_herbs and all_medications:
        interaction_warnings = check_interactions(all_herbs, all_medications)
        
        if interaction_warnings:
            logger.info(f"⚠️ INTERACTIONS FOUND: {len(interaction_warnings)}")
            
            safety_instructions = "\n\n🚨 CRITICAL SAFETY ALERTS 🚨\n"
            
            for warning in interaction_warnings:
                logger.info(f"   - {warning['herb']} + {warning['medication']} = {warning['severity']}")
                
                safety_instructions += f"""
**{warning['severity']} SEVERITY - DO NOT RECOMMEND {warning['herb']. upper()}**
User is taking: {warning['medication']}
Risk: {warning['reason']}
Safe alternatives to suggest: {', '.join(warning['alternatives'])}

You MUST:
1.  Clearly state that {warning['herb']} should NOT be used with {warning['medication']}
2.  Explain the specific risk in simple terms
3.  Recommend these safe alternatives instead

"""
    
    # ==========================================================================
    # STEP 6: QUERY REPHRASING
    # ==========================================================================
    rephrased_query = original_query
    
    try:
        rephrased_query = asyncio.run(
            rephrase_query(original_query, chat_history or [])
        )
        logger.info(f"Rephrased: {rephrased_query}")
    except Exception as e:
        logger.warning(f"Query rephrasing failed: {e}")
    
    # ==========================================================================
    # STEP 7: KNOWLEDGE RETRIEVAL
    # ==========================================================================
    context_chunks = ""
    chunks_list = []
    
    try:
        retrieval_start = datetime.now()
        
        retrieved = retrieve_content(rephrased_query, email_id, top_k=10)
        
        retrieval_end = datetime.now()
        
        if retrieved:
            chunks_list = retrieved. get('chunks', [])
            
            # Combine chunks into context
            chunk_texts = []
            for chunk in chunks_list[:5]:
                text = chunk.get('text', '') or chunk.get('content', '') or str(chunk)
                chunk_texts.append(text)
            
            context_chunks = "\n\n".join(chunk_texts)
        
        logger.info(f"Retrieved {len(chunks_list)} chunks")
        
    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        retrieval_end = datetime.now()
    
    # ==========================================================================
    # STEP 8: GENERATE RESPONSE
    # ==========================================================================
    llm_response = ""
    
    try:
        logger.info("Generating response with OpenAI...")
        
        # Get conversation history
        conversation_history = ""
        if CONVERSATION_ENABLED and conversation_manager and email_id:
            conversation_history = conversation_manager.get_formatted_history(email_id)
        
        generated = asyncio.run(
            generate_query_response(
                original_query=original_query,
                user_name=user_name,
                context_chunks=context_chunks,
                rephrased_query=rephrased_query,
                email_id=email_id,
                user_profile=user_profile,
                safety_instructions=safety_instructions,
                conversation_context={
                    'conditions': all_conditions,
                    'herbs': all_herbs,
                    'medications': all_medications,
                    'current_condition': current_condition,
                    'history': conversation_history,
                },
                context_changes=context_changes,
            )
        )
        
        llm_response = generated. get('response', '')
        
        logger.info(f"Response generated: {len(llm_response)} characters")
        
        response_map. update({
            'generation_start_time': generated.get('generation_start_time'),
            'generation_end_time': generated.get('generation_end_time'),
            'completion_tokens': generated.get('completion_tokens', 0),
            'prompt_tokens': generated.get('prompt_tokens', 0),
            'total_tokens': generated. get('total_tokens', 0),
        })
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        llm_response = "I'm having trouble generating a response right now. Please try again in a moment."
    
    # Store assistant response
    if CONVERSATION_ENABLED and conversation_manager and email_id:
        conversation_manager.add_message(email_id, 'assistant', llm_response)
    
    # ==========================================================================
    # STEP 9: TRUST ENGINE VALIDATION
    # ==========================================================================
    verified_herbs = []
    
    try:
        logger.info("Running Trust Engine validation...")
        
        from servvia2.trust_engine. engine import TrustEngine
        
        trust_engine = TrustEngine()
        
        # Verify response with CURRENT condition and CURRENT medications
        results, global_warnings = trust_engine. verify_response(
            llm_response=llm_response,
            query=original_query,
            user_conditions=profile_conditions,
            user_medications=all_medications,
            user_allergies=allergies,
            current_condition=current_condition,  # Pass the actual condition! 
        )
        
        # Collect verified herbs
        for r in results:
            if r. is_valid and not r.is_hallucination:
                verified_herbs.append(r.herb_name)
        
        logger.info(f"Verified herbs: {verified_herbs}")
        
        # Filter out warnings for stopped medications
        if stopped_medications:
            logger.info(f"Filtering warnings for stopped medications: {stopped_medications}")
            
            # Filter global warnings
            filtered_warnings = []
            for warning in global_warnings:
                should_keep = True
                for stopped in stopped_medications:
                    if stopped in warning. lower():
                        should_keep = False
                        logger.info(f"   Filtered out warning about: {stopped}")
                        break
                if should_keep:
                    filtered_warnings.append(warning)
            global_warnings = filtered_warnings
            
            # Filter interaction notes from results
            for r in results:
                if r.interaction_note:
                    for stopped in stopped_medications:
                        if stopped in r.interaction_note.lower():
                            logger.info(f"   Removed interaction note for: {stopped}")
                            r.interaction_note = None
                            break
        
        # Track verified herbs in conversation context
        if CONVERSATION_ENABLED and conversation_manager and email_id and verified_herbs:
            for herb in verified_herbs:
                conversation_manager.update_context(email_id, f"recommended {herb}")
        
        # Add validation section to response
        if results:
            validation_section = trust_engine.format_validation_section(results, global_warnings)
            llm_response = llm_response + validation_section
        
        response_map['trust_engine_verified'] = True
        response_map['verified_count'] = len(verified_herbs)
        
    except Exception as e:
        logger.warning(f"Trust Engine validation failed: {e}")
        response_map['trust_engine_verified'] = False
    
    # ==========================================================================
    # STEP 10: CHRONOBIOLOGICAL CONTEXT
    # ==========================================================================
    if verified_herbs:
        try:
            logger. info("Adding chronobiological context...")
            
            from servvia2.chronobiology.engine import CircadianEngine
            
            chrono_engine = CircadianEngine()
            
            # Get seasonal context
            seasonal = chrono_engine. get_seasonal_context(latitude=20.0)
            
            # Get timing advice for remedies
            timing_advice = chrono_engine.format_timing_advice(verified_herbs[:3])
            
            # Build seasonal section
            chrono_section = f"\n\n**Seasonal Wellness ({seasonal['season_name']}):**\n"
            chrono_section += f"_{seasonal['dosha_focus']}_\n\n"
            
            # Get beneficial herbs (filter out allergies)
            beneficial = seasonal. get('beneficial_herbs', [])
            safe_beneficial = [
                h for h in beneficial
                if h. lower() not in [a.lower() for a in allergies]
            ]
            
            if safe_beneficial:
                chrono_section += f"**Beneficial this season:** {', '.join(safe_beneficial[:4])}\n"
            
            # Add diet tip
            if seasonal.get('diet_tips'):
                chrono_section += f"**Diet tip:** {seasonal['diet_tips'][0]}\n"
            
            # Append to response
            llm_response = llm_response + timing_advice + chrono_section
            
        except Exception as e:
            logger. warning(f"Chronobiology context failed: {e}")
    
    # ==========================================================================
    # FINALIZE RESPONSE
    # ==========================================================================
    response_map['generated_final_response'] = llm_response
    response_map['retrieval_start'] = retrieval_start
    response_map['retrieval_end'] = retrieval_end
    
    message_data_update = {
        'message_id': message_id,
        'query': original_query,
        'rephrased_query': rephrased_query,
        'response': llm_response,
        'chunks_retrieved': len(chunks_list),
        'current_condition': current_condition,
    }
    
    logger.info("=" * 70)
    logger.info("ServVia 2. 0 Pipeline Completed Successfully")
    logger. info("=" * 70)
    
    return response_map, message_data_update
