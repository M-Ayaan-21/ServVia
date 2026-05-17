"""
ServVia — Fast Symptom Cluster Triage
======================================

Deterministic pattern detection for serious medical conditions that need
evaluation rather than home remedies.

This module intercepts the pipeline BEFORE RAG retrieval. When a known serious
symptom cluster is detected, a single targeted LLM response is generated
directly — bypassing RAG (which returns "home remedy" content) and the
multi-agent loop (which is designed for Tier 1 refinement, not Tier 2 routing).

Architecture impact:
  Serious cluster detected → single LLM call → response  (~2s, 1 LLM call)
  No cluster detected      → RAG + multi-agent           (existing path)

Cluster tiers:
  is_life_threatening=True  → _EMERGENCY_PROMPT   (call emergency services now)
  is_mental_health=True     → _MENTAL_HEALTH_PROMPT (crisis, warmth-first)
  is_life_threatening=False → _CLUSTER_PROMPT     (see doctor, timeframe given)

Author: ServVia Engineering
Version: 2.0.0
"""

import logging
from typing import Optional

logger = logging.getLogger("ServVia.Triage")


# ─────────────────────────────────────────────────────────────────────────────
# SERIOUS SYMPTOM CLUSTERS
#
# Fields:
#   id                  — unique identifier
#   is_life_threatening — True = acute emergency, use _EMERGENCY_PROMPT
#   is_mental_health    — True = crisis response, use _MENTAL_HEALTH_PROMPT
#   signals             — keyword fragments to match in the user query
#   min_signals         — minimum distinct matches to trigger
#   condition           — human-readable condition name
#   mechanism           — clinical explanation in plain English
#   tests               — what to ask the doctor for
#   urgency             — how soon to seek care
#   escalate_if         — conditions that upgrade urgency to ER now
#   immediate_actions   — what to do right now (life-threatening clusters only)
# ─────────────────────────────────────────────────────────────────────────────

TIER2_CLUSTERS = [

    # =========================================================================
    # IMMEDIATE LIFE-THREATENING EMERGENCIES
    # =========================================================================

    {
        "id": "heart_attack",
        "is_life_threatening": True,
        "signals": [
            "chest pain", "chest tightness", "chest pressure", "chest discomfort",
            "crushing chest", "squeezing chest", "heavy chest", "chest heaviness",
            "left arm pain", "left arm numb", "left arm tingling", "left arm heavy",
            "jaw pain", "jaw tightness", "pain in jaw",
            "heart attack", "myocardial infarction",
            "pain radiating to arm", "pain radiating to jaw", "pain in my arm and chest",
            "sweating with chest", "cold sweat chest",
            "nausea with chest pain", "vomiting with chest pain",
            "shortness of breath with chest", "breathless and chest pain",
        ],
        "min_signals": 1,
        "condition": "a possible heart attack (myocardial infarction)",
        "mechanism": (
            "A heart attack occurs when blood supply to part of the heart muscle is blocked, "
            "usually by a blood clot in a coronary artery. Every minute without blood flow "
            "causes more heart muscle to die permanently. Chest pain or pressure — especially "
            "with pain radiating to the left arm or jaw, sweating, or shortness of breath — "
            "is a cardiac emergency until proven otherwise. Do not wait to see if it gets better."
        ),
        "tests": "ECG (electrocardiogram) and cardiac troponin blood test — done in the ER",
        "urgency": "CALL EMERGENCY SERVICES (112 / 102 / 911) RIGHT NOW — do not drive yourself",
        "escalate_if": [
            "any chest pain, pressure, or tightness — this is already the escalation trigger",
            "pain spreads to your left arm, jaw, neck, or back",
            "sudden sweating, nausea, or lightheadedness with chest discomfort",
            "shortness of breath with or without chest pain",
            "a sudden feeling of doom, extreme anxiety, or that something is very wrong",
        ],
        "immediate_actions": [
            "Call emergency services (112 / 102 / 911) immediately — do not wait",
            "Sit or lie down in the most comfortable position — do not walk around or exert yourself",
            "Chew (do not swallow whole) one adult aspirin 300–325mg if not allergic and no active bleeding",
            "If prescribed nitroglycerin, take it now as directed",
            "Loosen any tight clothing around your chest and neck",
            "Unlock the front door so paramedics can enter without delay",
            "Stay on the phone with emergency services until help arrives",
        ],
    },

    {
        "id": "stroke",
        "is_life_threatening": True,
        "signals": [
            "face drooping", "face droop", "one side of face drooping", "face numb",
            "facial numbness", "face weakness", "face falling",
            "arm weakness", "arm numb", "arm numbness", "arm won't move",
            "one arm weak", "sudden weakness one side", "weakness one side body",
            "slurred speech", "slurring speech", "can't speak", "speech difficulty",
            "speech suddenly changed", "words not coming out", "speech slurred",
            "sudden confusion", "sudden inability to speak", "sudden inability to understand",
            "sudden vision loss", "sudden blurred vision", "sudden double vision",
            "sudden severe headache", "worst headache of my life", "thunderclap headache",
            "sudden dizziness", "sudden loss of balance", "can't walk suddenly",
            "stroke", "mini stroke", "tia", "transient ischemic",
        ],
        "min_signals": 1,
        "condition": "a possible stroke or TIA (transient ischemic attack)",
        "mechanism": (
            "A stroke occurs when blood supply to part of the brain is cut off — either by "
            "a clot (ischaemic stroke, 85% of cases) or a burst blood vessel. Brain cells begin "
            "dying within minutes. The FAST test captures the key signs: Face drooping, Arm weakness, "
            "Speech difficulty, Time to call. A TIA ('mini stroke') has the same symptoms but "
            "resolves — it is a critical warning sign that a major stroke may follow within days. "
            "Clot-busting treatment works only if given within 4.5 hours of symptom onset."
        ),
        "tests": "CT scan and MRI of the brain — done in the ER immediately on arrival",
        "urgency": "CALL EMERGENCY SERVICES (112 / 102 / 911) NOW — note the exact time symptoms started",
        "escalate_if": [
            "any face drooping, arm weakness, or speech difficulty — this is already the trigger",
            "sudden severe headache described as the worst of your life",
            "sudden loss of vision, double vision, or difficulty seeing in one or both eyes",
            "sudden inability to walk, severe dizziness, or loss of balance",
        ],
        "immediate_actions": [
            "Call emergency services (112 / 102 / 911) immediately",
            "Note the exact time the symptoms started — this determines treatment options",
            "Do NOT give food, water, or medication — stroke can cause swallowing difficulties",
            "If conscious, lay them on their side (recovery position) in case of vomiting",
            "Do NOT let them 'sleep it off' — the clot-busting window closes in 4.5 hours",
            "Use FAST to describe symptoms to the dispatcher: Face, Arm, Speech, Time",
            "Stay with them and keep them calm and still until help arrives",
        ],
    },

    {
        "id": "anaphylaxis",
        "is_life_threatening": True,
        "signals": [
            "throat swelling", "throat is swelling", "my throat is swelling",
            "throat closing", "throat tightening", "throat feels tight",
            "can't swallow", "difficulty swallowing suddenly",
            "tongue swelling", "swollen tongue", "lip swelling with breathing",
            "can't breathe after eating", "can't breathe after bee sting",
            "can't breathe after taking medication", "sudden severe breathing difficulty",
            "allergic reaction severe", "severe allergic reaction",
            "anaphylaxis", "anaphylactic",
            "hives and breathing difficulty", "rash and can't breathe",
            "epipen", "epinephrine injection",
            "bee sting and breathing", "wasp sting and breathing",
            "nut allergy reaction", "shellfish allergy reaction",
            "face swelling suddenly", "lips swelling rapidly",
        ],
        "min_signals": 1,
        "condition": "anaphylaxis — a severe, life-threatening allergic reaction",
        "mechanism": (
            "Anaphylaxis is a sudden, life-threatening allergic reaction where the immune system "
            "floods the body with chemicals, causing the throat to swell, airways to narrow, and "
            "blood pressure to drop. Without treatment it can be fatal within minutes. "
            "Common triggers: peanuts, tree nuts, shellfish, insect stings, and certain medications. "
            "The only effective treatment is epinephrine (adrenaline). Antihistamines alone "
            "will not stop anaphylaxis."
        ),
        "tests": "No test needed — this is a clinical emergency. Treatment first, tests later.",
        "urgency": "CALL EMERGENCY SERVICES (112 / 102 / 911) NOW — anaphylaxis can be fatal within minutes",
        "escalate_if": [
            "any throat swelling, tongue swelling, or difficulty breathing — this is already critical",
            "skin turning blue, grey, or ashen",
            "loss of consciousness or collapse",
            "symptoms returning after an EpiPen (biphasic reaction) — needs observation for 4+ hours",
        ],
        "immediate_actions": [
            "Call emergency services (112 / 102 / 911) immediately",
            "If an EpiPen (epinephrine auto-injector) is available, use it NOW — outer thigh, through clothing if needed",
            "Lay the person flat with legs elevated — unless struggling to breathe, in which case let them sit up",
            "If they stop breathing, begin CPR",
            "A second EpiPen can be used after 5–15 minutes if symptoms return and help has not arrived",
            "Do NOT give antihistamines as the primary treatment — they do not stop anaphylaxis",
            "Do NOT let them stand or walk — keep them still and calm",
        ],
    },

    {
        "id": "meningitis",
        "is_life_threatening": True,
        "signals": [
            # neck signals
            "stiff neck", "neck stiffness", "stiff neck fever", "neck can't bend",
            "chin to chest", "cannot touch chin to chest",
            # light sensitivity signals
            "sensitivity to light", "light hurts eyes", "photophobia",
            "light is painful", "can't open eyes in light",
            # headache + fever combo signals
            "severe headache stiff neck", "worst headache and stiff neck",
            "severe headache vomiting and fever",
            # rash signals (any one alone is enough with min_signals=1 override below, but
            # in combination with fever/headache they clearly indicate meningococcal disease)
            "purple rash", "petechiae", "rash doesn't fade", "rash glass test",
            "non-blanching rash", "spots that don't fade under pressure",
            "red spots won't fade", "dark spots on skin with fever",
            # direct naming
            "meningitis", "meningococcal",
            # other combinations
            "fever and confusion", "confusion with high fever",
        ],
        "min_signals": 2,
        "condition": "possible meningitis — infection of the lining surrounding the brain",
        "mechanism": (
            "Meningitis is inflammation of the membranes surrounding the brain and spinal cord, "
            "usually caused by bacteria or viruses. Bacterial meningitis is one of the most "
            "time-critical emergencies in medicine — it can kill within 24 hours and cause "
            "permanent disability (deafness, brain damage, limb loss) even with treatment if delayed. "
            "The classic triad is severe headache, stiff neck, and fever. "
            "A non-blanching rash (purple or red spots that do not fade when pressed with a glass) "
            "means bacteria are in the bloodstream — this is the most dangerous sign."
        ),
        "tests": "Blood cultures, lumbar puncture, and CT scan — all in the ER urgently",
        "urgency": "GO TO THE EMERGENCY ROOM NOW — bacterial meningitis can kill within hours",
        "escalate_if": [
            "any purple or red spots that do NOT fade when a glass is pressed firmly against them",
            "fever with stiff neck and sensitivity to light together",
            "confusion, extreme drowsiness, or loss of consciousness",
            "seizures",
            "a rash that is spreading rapidly",
        ],
        "immediate_actions": [
            "Go to the emergency room immediately — do not wait for symptoms to worsen",
            "Do the glass test: press a clear glass firmly against any rash spots. If they do NOT fade, call 112 / 102 now",
            "If you cannot reach the ER quickly and a non-blanching rash is present, call emergency services immediately",
            "Keep the person in a quiet, dark room if light sensitivity is present while preparing to leave",
            "Do not give food or water — they may need immediate procedures",
            "If IV antibiotics are available and a doctor is present, they should be given before transfer",
        ],
    },

    {
        "id": "seizure",
        "is_life_threatening": True,
        "signals": [
            "seizure", "convulsions", "convulsing", "fits", "fitting",
            "having a fit", "having fits", "a fit suddenly",
            "tonic-clonic", "epileptic attack", "epilepsy attack",
            "shaking uncontrollably", "body shaking cannot stop", "body shaking won't stop",
            "fell down shaking", "collapsed and shaking",
            "jerking movements", "uncontrolled jerking", "limbs jerking",
            "foaming at mouth", "foaming from mouth",
            "blank stare not responding", "staring blankly unresponsive",
            "febrile seizure", "fever seizure", "seizure with fever",
            "loss of consciousness shaking",
        ],
        "min_signals": 1,
        "condition": "a seizure — sudden abnormal electrical activity in the brain",
        "mechanism": (
            "A seizure is caused by a sudden, abnormal burst of electrical activity in the brain. "
            "It can cause convulsions, loss of consciousness, blank staring, or muscle rigidity. "
            "Most seizures stop on their own within 1–3 minutes. "
            "A seizure lasting more than 5 minutes (status epilepticus) is a life-threatening emergency "
            "requiring immediate IV medication. A first-ever seizure always requires emergency evaluation "
            "to find the cause. In children, febrile seizures from high fever are common but still "
            "need assessment."
        ),
        "tests": "Blood sugar, electrolytes, EEG, and brain imaging (CT/MRI) in the ER",
        "urgency": "Call emergency services if: first-ever seizure, lasts over 5 minutes, or person does not regain consciousness",
        "escalate_if": [
            "seizure lasts more than 5 minutes without stopping",
            "the person does not regain consciousness after the shaking stops",
            "another seizure begins shortly after the first ends",
            "this is the person's first ever seizure",
            "seizure happens in water or causes a fall with head injury",
            "the person is pregnant",
        ],
        "immediate_actions": [
            "Stay calm and stay with the person — most seizures end within 1–3 minutes on their own",
            "Ease them to the floor and clear the area of hard or sharp objects",
            "Cushion their head with something soft — a folded jacket works",
            "Turn them gently onto their side to prevent choking on saliva or vomit",
            "Time the seizure from the moment it starts — this is critical information for paramedics",
            "DO NOT restrain them or hold their limbs down",
            "DO NOT put anything in their mouth — people cannot swallow their tongue",
            "After the seizure: keep them on their side, speak calmly, let them rest (post-ictal phase)",
            "Call emergency services if it lasts over 5 minutes or they don't regain consciousness",
        ],
    },

    {
        "id": "severe_breathing",
        "is_life_threatening": True,
        "signals": [
            "can't breathe", "cannot breathe", "struggling to breathe",
            "gasping for air", "gasping", "gasping and can't breathe",
            "severe shortness of breath", "very short of breath at rest",
            "lips turning blue", "fingernails turning blue", "face turning blue",
            "skin turning blue", "blue lips", "blue fingernails", "cyanosis",
            "choking", "something stuck in throat", "airway blocked",
            "severe asthma attack", "asthma inhaler not working", "inhaler not helping",
            "breathing very fast and distressed", "rapid breathing and panic",
            "wheezing and can't breathe", "no air moving", "air not going in",
            "suffocating", "feels like suffocating",
        ],
        "min_signals": 1,
        "condition": "severe breathing difficulty — a respiratory emergency",
        "mechanism": (
            "Severe breathing difficulty means the body is critically short of oxygen. "
            "Causes include severe asthma, anaphylaxis, a blocked airway, pulmonary embolism "
            "(blood clot in the lung), pneumonia with respiratory failure, or heart failure. "
            "Blue lips or fingernails (cyanosis) is a sign that oxygen levels are dangerously low — "
            "the brain begins to suffer within minutes. Any of these causes requires immediate "
            "emergency intervention."
        ),
        "tests": "Pulse oximetry, arterial blood gas, and chest X-ray — in the ER immediately",
        "urgency": "CALL EMERGENCY SERVICES (112 / 102 / 911) IMMEDIATELY",
        "escalate_if": [
            "lips, face, or fingernails are turning blue or grey — call now if not already done",
            "the person cannot speak in full sentences due to breathlessness",
            "an asthma inhaler is providing no relief after repeated puffs",
            "breathing is getting rapidly worse",
            "the person loses consciousness",
        ],
        "immediate_actions": [
            "Call emergency services (112 / 102 / 911) immediately",
            "Sit the person upright, leaning slightly forward — this opens the airway more than lying down",
            "If choking (something stuck): perform abdominal thrusts (Heimlich manoeuvre) for a conscious adult",
            "If severe asthma: use the blue reliever inhaler (salbutamol) — 1 puff every 30–60 seconds, up to 10 puffs while waiting for help",
            "Loosen any tight clothing around the neck and chest",
            "Do not leave the person alone",
            "If they lose consciousness and stop breathing, begin CPR",
        ],
    },

    {
        "id": "poisoning_overdose",
        "is_life_threatening": True,
        "signals": [
            "overdose", "took too many pills", "swallowed too much medication",
            "drug overdose", "medication overdose", "paracetamol overdose",
            "took pills to hurt myself", "took a lot of pills",
            "swallowed poison", "drank poison", "ingested chemicals",
            "swallowed bleach", "drank bleach", "swallowed cleaning product",
            "pesticide swallowed", "ingested pesticide", "rat poison swallowed",
            "carbon monoxide", "gas leak and feeling sick", "gas poisoning",
            "smoke inhalation severe",
            "poisoning", "toxic ingestion", "accidentally swallowed",
            "unconscious after taking pills", "not waking up after medication",
        ],
        "min_signals": 1,
        "condition": "suspected poisoning or drug overdose",
        "mechanism": (
            "Poisoning or overdose causes toxic levels of a substance to overwhelm the body's "
            "ability to process or eliminate it. Different substances cause different effects: "
            "opioids slow breathing to a stop, stimulants cause cardiac arrest, and paracetamol "
            "(acetaminophen) overdose causes fatal liver failure days later even after the person "
            "initially feels fine. Carbon monoxide poisoning is invisible and odourless. "
            "Many antidotes only work if given within a short window — time is critical."
        ),
        "tests": "Toxicology screen, paracetamol and salicylate levels, ECG, and liver function tests in the ER",
        "urgency": "CALL EMERGENCY SERVICES (112 / 102 / 911) OR POISON CONTROL NOW",
        "escalate_if": [
            "the person is unconscious or cannot be woken up",
            "breathing is very slow, shallow, or has stopped",
            "seizures are occurring",
            "lips or skin are turning blue or grey",
            "any chemical, pesticide, or carbon monoxide exposure — even if they seem fine right now",
        ],
        "immediate_actions": [
            "Call emergency services (112 / 102 / 911) immediately",
            "Tell them exactly: what substance was taken, how much, and when",
            "Do NOT induce vomiting unless specifically instructed by emergency services — for many substances it causes more damage",
            "Bring the medication bottle, packaging, or container to hospital — it helps doctors identify the exact substance",
            "If the person is unconscious but breathing, place them in the recovery position (on their side)",
            "If carbon monoxide or gas: move the person to fresh air immediately, then call emergency services",
            "For paracetamol overdose: go to the ER even if they feel completely fine — liver damage appears 2–3 days later",
            "Stay on the line with emergency services",
        ],
    },

    {
        "id": "infant_high_fever",
        "is_life_threatening": True,
        "signals": [
            "baby fever", "infant fever", "newborn fever", "toddler high fever",
            "3 month old fever", "6 month old fever", "2 month old fever",
            "baby temperature high", "infant temperature very high",
            "baby not responding", "baby very sleepy with fever", "baby hard to wake",
            "baby stiff with fever", "baby rash and fever",
            "baby won't eat and fever", "newborn not feeding and sick",
            "2 year old very high fever", "1 year old very high fever",
            "child fever 40", "baby 40 degrees", "infant 39 degrees",
            "febrile convulsion baby", "baby shaking with fever",
        ],
        "min_signals": 1,
        "condition": "high fever in an infant or young child — requires urgent assessment",
        "mechanism": (
            "In infants under 3 months, any fever above 38°C (100.4°F) is a medical emergency. "
            "Their immune systems cannot contain serious infections (sepsis, meningitis, urinary "
            "tract infections) the way older children can, and the source must be urgently identified. "
            "In children aged 3 months to 5 years, fever above 39°C lasting more than 48 hours — "
            "or any fever with a rash, stiff neck, breathing difficulty, a child who is unusually "
            "difficult to rouse, or a febrile seizure — requires the same urgency."
        ),
        "tests": "Blood count (CBC), urine test, blood culture if under 3 months — done by a doctor",
        "urgency": "Go to the ER NOW for infants under 3 months with any fever. Same day for children with fever above 39°C lasting 48+ hours or with worrying signs.",
        "escalate_if": [
            "infant is under 3 months old with any fever at all — no exceptions, go to ER now",
            "child has fever AND a rash, stiff neck, or sensitivity to light",
            "the child is unusually difficult to wake, limp, or not responding normally",
            "fever above 40°C (104°F) in any child",
            "the child is not urinating or has very dry mouth and lips (dehydration)",
            "a febrile seizure occurs",
        ],
        "immediate_actions": [
            "Infants under 3 months with any fever: go to the emergency room now — do not wait",
            "Undress the child to one light layer — do not bundle them up, it traps heat",
            "Give age-appropriate paracetamol (acetaminophen) at the correct weight-based dose",
            "Ibuprofen can be given to children over 6 months as an alternative",
            "Offer fluids frequently: breast milk, formula, or oral rehydration solution",
            "Do NOT use cold water baths or alcohol rubs — they cause shivering which raises temperature",
            "Do NOT give aspirin to any child — risk of Reye's syndrome",
            "Monitor temperature every 30 minutes and watch for any worsening signs",
        ],
    },

    {
        "id": "severe_head_injury",
        "is_life_threatening": True,
        "signals": [
            "hit head hard", "head injury", "head trauma", "fell and hit head",
            "head knocked", "knocked out", "loss of consciousness after fall",
            "unconscious after hitting head", "blacked out after hitting head",
            "concussion severe", "severe concussion",
            "vomiting after hitting head", "vomiting repeatedly after head injury",
            "confusion after hitting head", "can't remember after hitting head",
            "amnesia after fall", "memory loss after head injury",
            "seizure after hitting head",
            "clear fluid from ear", "clear fluid from nose after injury",
            "skull fracture", "depressed skull",
        ],
        "min_signals": 1,
        "condition": "possible serious head injury — requires urgent neurological assessment",
        "mechanism": (
            "A serious head injury can cause bleeding inside or around the brain (intracranial haemorrhage). "
            "An epidural haematoma has a classic 'lucid interval' — the person may seem fine for hours "
            "after the injury before suddenly deteriorating as blood pressure builds on the brain. "
            "Any loss of consciousness, repeated vomiting, confusion, or amnesia after a head injury "
            "is a red flag that requires immediate imaging. Clear fluid from the ear or nose after "
            "a head injury suggests a skull base fracture."
        ),
        "tests": "CT scan of the head — done in the ER urgently",
        "urgency": "GO TO THE EMERGENCY ROOM NOW if there was any loss of consciousness, repeated vomiting, confusion, or worsening headache",
        "escalate_if": [
            "the person lost consciousness at any point — even briefly",
            "repeated vomiting (more than once) after the head injury",
            "increasing confusion, drowsiness, or difficulty staying awake",
            "one pupil is larger than the other",
            "clear fluid draining from the ear or nose",
            "weakness or numbness in the arms or legs after the injury",
            "a seizure occurs after the head injury",
        ],
        "immediate_actions": [
            "Do not leave the person alone for at least 24 hours",
            "If they lost consciousness, call emergency services (112 / 102 / 911) immediately",
            "Keep the head and neck still if a spinal injury is possible — do not move them unless they are in danger",
            "Apply gentle pressure to any bleeding scalp wound with a clean cloth",
            "Do NOT give ibuprofen or aspirin for the pain — they increase bleeding risk. Paracetamol only.",
            "Do NOT give alcohol",
            "If they are deteriorating, go to the ER immediately — do not wait until morning",
        ],
    },

    # =========================================================================
    # MENTAL HEALTH CRISIS
    # =========================================================================

    {
        "id": "mental_health_crisis",
        "is_life_threatening": False,
        "is_mental_health": True,
        "signals": [
            "want to die", "want to kill myself", "thinking about suicide", "suicidal",
            "don't want to live", "end my life", "end it all", "not worth living",
            "better off dead", "nobody would care if i died", "planning to hurt myself",
            "self harm", "cutting myself", "hurting myself", "been cutting",
            "suicide attempt", "took pills to die", "took pills to hurt myself",
            "life is not worth living", "wish i was dead", "wish i wasn't here",
            "can't go on", "no reason to live", "want everything to stop",
            "thinking of ending it", "nothing to live for",
        ],
        "min_signals": 1,
        "condition": "you are going through something extremely painful right now — and you are not alone",
        "mechanism": (
            "Suicidal thoughts and self-harm are signs of overwhelming psychological pain — "
            "not weakness, not a character flaw. They are a symptom of something that is treatable. "
            "Depression, trauma, and extreme stress can make the mind convince itself that things "
            "will never get better. That is not the truth. People recover from this — and you deserve "
            "that chance."
        ),
        "tests": "A conversation with a crisis counselor is the most important first step right now",
        "urgency": "Please reach out to a crisis line right now — you deserve support",
        "escalate_if": [
            "you have already taken any pills or harmed yourself — call 112 / 102 now",
            "you have a specific plan and the means to carry it out",
            "you do not feel you can keep yourself safe right now",
        ],
        "immediate_actions": [
            "iCall India: 9152987821 (Mon–Sat, 8am–10pm)",
            "Vandrevala Foundation: 1860-2662-345 (24/7, free)",
            "iCall also offers free online counselling: icallhelpline.org",
            "International resources: findahelpline.com",
            "If you have already harmed yourself, call emergency services (112 / 102) right now",
        ],
    },

    # =========================================================================
    # SERIOUS CONDITIONS REQUIRING PROMPT MEDICAL EVALUATION
    # =========================================================================

    {
        "id": "metabolic_diabetes",
        "is_life_threatening": False,
        "signals": [
            "excessive thirst", "frequent urination", "urinating frequently",
            "polyuria", "polydipsia", "urination at night", "waking up to urinate",
            "nocturia", "blurry vision", "blurred vision", "slow healing",
            "slow-healing sores", "slow-healing wounds", "wounds not healing",
            "unexplained weight loss", "extreme hunger", "increased appetite",
        ],
        "min_signals": 3,
        "condition": "a blood sugar regulation condition — commonly associated with diabetes",
        "mechanism": (
            "High blood sugar causes the kidneys to filter excess glucose, pulling fluid "
            "with it and causing frequent urination. This leads to dehydration and extreme "
            "thirst — creating a cycle. Because cells can't properly absorb glucose for "
            "energy, the body signals extreme hunger and fatigue, and healing slows."
        ),
        "tests": "fasting blood glucose (FBG) and HbA1c — both are simple blood tests",
        "urgency": "within the next 24-48 hours",
        "escalate_if": [
            "you feel confused, extremely drowsy, or are breathing unusually fast or deeply",
            "you cannot keep fluids down or are vomiting",
            "a glucose meter reads above 400 mg/dL (22 mmol/L)",
            "you feel fruity breath, nausea, and extreme weakness together",
        ],
    },

    {
        "id": "dengue",
        "is_life_threatening": False,
        "signals": [
            "dengue", "pain behind eyes", "eye pain", "retro-orbital pain",
            "pain behind my eyes", "joint pain", "muscle pain", "bone pain",
            "breakbone", "bleeding gums", "platelet", "rash with fever",
            "rash after fever", "fever and rash", "high fever sudden",
            "sudden high fever", "mosquito bite fever",
        ],
        "min_signals": 2,
        "condition": "dengue fever",
        "mechanism": (
            "Dengue is a mosquito-borne viral infection. The classic presentation is sudden "
            "high fever, severe headache, pain behind the eyes, and intense joint and muscle "
            "pain (historically called 'breakbone fever'). A characteristic rash typically "
            "appears 2-5 days after fever onset. In severe cases, platelet counts drop, "
            "causing bleeding symptoms."
        ),
        "tests": (
            "dengue NS1 antigen test (most accurate in the first 5 days of fever) "
            "and a platelet count and full blood count"
        ),
        "urgency": "TODAY — dengue can progress to dengue hemorrhagic fever, which is life-threatening",
        "escalate_if": [
            "bleeding from gums, nose, or easy bruising",
            "blood in urine, vomit, or stool",
            "severe persistent abdominal pain",
            "vomiting more than 3 times in a 24-hour period",
            "extreme restlessness, confusion, or rapid breathing",
            "rapid drop in platelet count",
        ],
    },

    {
        "id": "malaria",
        "is_life_threatening": False,
        "signals": [
            "malaria", "cyclical fever",
            "fever and chills", "fever chills", "chills and fever",
            "shivering with fever", "shivering fever", "rigors",
            "high fever sweating", "fever sweating chills",
            "repeated fever", "fever comes and goes",
            "mosquito bite and fever", "mosquito bite fever",
        ],
        "min_signals": 2,
        "condition": "malaria",
        "mechanism": (
            "Malaria is caused by a Plasmodium parasite spread by Anopheles mosquitoes. "
            "The characteristic pattern of fever, chills, and sweating that cycles every "
            "24-72 hours corresponds to parasites bursting from red blood cells. "
            "Untreated, it can progress to cerebral malaria, which is life-threatening."
        ),
        "tests": "malaria rapid diagnostic test (RDT) or thick and thin blood smear",
        "urgency": "TODAY — malaria can deteriorate rapidly if untreated",
        "escalate_if": [
            "confusion, extreme drowsiness, or loss of consciousness",
            "seizures",
            "very dark, tea-coloured, or black urine",
            "severe difficulty breathing",
            "high fever above 39.5°C that does not respond to paracetamol",
        ],
    },

    {
        "id": "typhoid",
        "is_life_threatening": False,
        "signals": [
            "typhoid", "enteric fever", "sustained fever", "fever for many days",
            "fever for a week", "step-ladder fever", "rose spots",
            "contaminated water", "abdominal pain with fever week",
        ],
        "min_signals": 2,
        "condition": "typhoid fever (enteric fever)",
        "mechanism": (
            "Typhoid is caused by Salmonella Typhi bacteria from contaminated food or water. "
            "It causes a sustained fever that gradually increases over several days, accompanied "
            "by headache, abdominal discomfort, and sometimes a faint rash on the torso. "
            "It requires antibiotic treatment — it does not resolve on its own."
        ),
        "tests": (
            "Typhidot rapid test, Widal test, or blood culture "
            "(blood culture is the most definitive)"
        ),
        "urgency": "within 24 hours",
        "escalate_if": [
            "fever above 39°C lasting more than 3 days",
            "severe abdominal pain, especially with a rigid or board-like abdomen",
            "confusion or delirium",
            "bloody diarrhoea",
        ],
    },

    {
        "id": "thyroid",
        "is_life_threatening": False,
        "signals": [
            "unexplained weight gain", "unexplained weight loss", "gaining weight",
            "losing weight without reason", "hair loss", "hair thinning",
            "always cold", "always feeling cold", "cold intolerance",
            "always hot", "heat intolerance", "always feeling warm",
            "heart palpitations", "neck swelling", "swollen neck", "goitre",
            "dry skin", "brain fog", "constipation and fatigue",
        ],
        "min_signals": 3,
        "condition": "a thyroid condition (underactive or overactive thyroid gland)",
        "mechanism": (
            "The thyroid gland controls your metabolism, energy, and many body processes. "
            "An underactive thyroid (hypothyroidism) slows everything down: weight gain, "
            "fatigue, feeling cold, hair loss, and constipation. An overactive thyroid "
            "(hyperthyroidism) speeds everything up: weight loss, heat intolerance, "
            "palpitations, and anxiety."
        ),
        "tests": "thyroid panel: TSH (thyroid stimulating hormone), free T3, and free T4",
        "urgency": "within the next 1-2 weeks",
        "escalate_if": [
            "resting heart rate consistently above 120 beats per minute",
            "severe difficulty breathing or swallowing due to neck swelling",
            "sudden extreme weakness or muscle paralysis",
        ],
    },

    {
        "id": "anaemia",
        "is_life_threatening": False,
        "signals": [
            "pale skin", "pallor", "pale nails", "pale inner eyelids",
            "white gums", "fatigue and breathless", "breathless on walking",
            "breathless with mild activity", "rapid heartbeat", "pounding heart",
            "dizziness and lightheaded", "fainting", "cold hands and feet",
            "brittle nails", "spoon-shaped nails", "persistent tiredness",
        ],
        "min_signals": 3,
        "condition": "anaemia (low haemoglobin, possibly from iron or vitamin deficiency)",
        "mechanism": (
            "Anaemia means your red blood cells are not carrying enough oxygen to your "
            "tissues. This causes persistent fatigue, pale skin (especially in the inner "
            "eyelids and gums), and breathlessness even with mild activity. "
            "Iron deficiency is the most common cause, but B12 and folate deficiencies "
            "and chronic illness can also cause anaemia."
        ),
        "tests": (
            "complete blood count (CBC) with haemoglobin, ferritin, serum iron, "
            "and vitamin B12 and folate levels"
        ),
        "urgency": "within 48 hours",
        "escalate_if": [
            "chest pain or difficulty breathing at rest",
            "heart rate consistently over 120 bpm",
            "fainting or near-fainting episodes",
            "extreme pallor with confusion",
        ],
    },

    {
        "id": "liver",
        "is_life_threatening": False,
        "signals": [
            "yellow skin", "jaundice", "yellowing of skin", "yellow eyes",
            "whites of eyes yellow", "dark urine", "dark coloured urine",
            "pale stool", "clay stool", "clay-coloured stool", "grey stool",
            "right upper abdominal pain", "right upper abdomen", "liver pain",
            "hepatitis", "itchy skin with jaundice",
        ],
        "min_signals": 2,
        "condition": "a liver condition (such as hepatitis or liver inflammation)",
        "mechanism": (
            "Yellowing of the skin and eyes (jaundice) occurs when the liver cannot "
            "properly process bilirubin — a waste product from old red blood cells. "
            "Dark urine and pale stools occur because bilirubin is being excreted through "
            "urine instead of stool. This pattern strongly suggests the liver is under "
            "significant stress and needs immediate evaluation."
        ),
        "tests": "liver function tests (LFTs), total and direct bilirubin, and hepatitis panel (A, B, C)",
        "urgency": "TODAY",
        "escalate_if": [
            "severe abdominal pain or a rigid, board-like abdomen",
            "confusion, personality changes, or extreme drowsiness (possible hepatic encephalopathy)",
            "vomiting blood or very dark coffee-ground material",
            "high fever with jaundice",
        ],
    },

    {
        "id": "kidney",
        "is_life_threatening": False,
        "signals": [
            "swollen feet", "swollen ankles", "puffy face", "facial puffiness",
            "foamy urine", "frothy urine", "bubbly urine", "blood in urine",
            "dark urine with swelling", "reduced urination", "no urination",
            "lower back pain flank", "flank pain", "kidney pain",
        ],
        "min_signals": 2,
        "condition": "a kidney function issue",
        "mechanism": (
            "The kidneys filter your blood and regulate fluid balance. When they are "
            "under stress, fluid accumulates causing swelling in the feet, ankles, and "
            "face. Foamy urine indicates protein is leaking through the kidneys — "
            "a sign they are not filtering properly."
        ),
        "tests": (
            "kidney function tests (creatinine, eGFR, blood urea nitrogen) "
            "and urine analysis (urinalysis with microscopy)"
        ),
        "urgency": "within 24 hours",
        "escalate_if": [
            "no urination for more than 8 hours",
            "severe swelling accompanied by difficulty breathing",
            "confusion or extreme fatigue with swelling",
        ],
    },

    {
        "id": "appendicitis",
        "is_life_threatening": False,
        "signals": [
            "lower right abdominal pain", "right lower abdomen", "lower right pain",
            "pain started near navel", "pain moved to right side",
            "appendix pain", "rebound tenderness", "pain right side abdomen",
        ],
        "min_signals": 2,
        "condition": "possible appendicitis (inflammation of the appendix)",
        "mechanism": (
            "Appendicitis classically begins with pain near the navel that gradually "
            "shifts to the lower right abdomen over 12-24 hours, often accompanied by "
            "nausea, low-grade fever, and loss of appetite. If the appendix ruptures, "
            "the pain may briefly subside then return much more severely."
        ),
        "tests": (
            "physical examination by a doctor, followed by abdominal ultrasound "
            "or CT scan if needed"
        ),
        "urgency": "IMMEDIATELY — go to an emergency room now",
        "escalate_if": [
            "sudden severe worsening of pain after a period of improvement (possible rupture)",
            "fever above 38.5°C with abdominal pain",
            "abdomen feels rigid or board-like",
            "pain so severe you cannot stand upright or walk normally",
        ],
    },

    # =========================================================================
    # PCOS / PCOD
    # =========================================================================

    {
        "id": "pcos_pcod",
        "is_life_threatening": False,
        "signals": [
            "irregular periods", "irregular menstrual cycle", "missed periods",
            "periods stopped", "no period", "no periods", "periods not coming",
            "infrequent periods", "skipped period", "oligomenorrhea",
            "facial hair", "chin hair", "upper lip hair", "excess hair on face",
            "hirsutism", "unwanted hair growth", "hair on chest",
            "acne and irregular periods", "acne and weight gain",
            "pcos", "pcod", "polycystic ovary", "ovarian cysts",
            "hair thinning and irregular periods", "hair fall and irregular periods",
            "weight gain and irregular periods", "difficulty conceiving",
            "infertility and irregular periods",
        ],
        "min_signals": 2,
        "condition": "PCOS (Polycystic Ovary Syndrome) / PCOD",
        "mechanism": (
            "PCOS is a hormonal condition affecting roughly 1 in 5 women in India. "
            "Elevated androgens (male hormones) cause irregular or absent periods, "
            "excess hair growth, scalp hair thinning, and acne. Insulin resistance "
            "is common and contributes to weight gain and difficulty managing blood sugar. "
            "PCOS also affects fertility. It is manageable with lifestyle changes and, "
            "in some cases, medication — but it needs a proper diagnosis first."
        ),
        "tests": (
            "pelvic ultrasound (transvaginal or abdominal), hormonal panel "
            "(LH, FSH, testosterone, DHEA-S, AMH), fasting insulin, "
            "fasting blood glucose, and thyroid function (TSH)"
        ),
        "urgency": "within the next 1–2 weeks — this is not an emergency but should not be ignored",
        "escalate_if": [
            "you are trying to conceive and have not been successful after 12 months (or 6 months if over 35)",
            "you haven't had a period in over 3 months — uterine lining build-up needs assessment",
            "severe pelvic pain (could indicate ovarian cyst torsion — ER needed urgently)",
            "symptoms of diabetes: excessive thirst, frequent urination, blurred vision",
        ],
    },

    # =========================================================================
    # URINARY TRACT INFECTION (UTI)
    # =========================================================================

    {
        "id": "uti",
        "is_life_threatening": False,
        "signals": [
            "burning urination", "burning when urinating", "burning pee",
            "painful urination", "pain while peeing", "stinging when peeing",
            "frequent urination burning", "need to pee constantly",
            "urgency to urinate", "can't hold urine", "urine urgency",
            "cloudy urine", "smelly urine", "foul-smelling urine",
            "blood in urine", "pink urine", "red urine",
            "lower abdominal pain and urination", "pelvic pain and urination",
            "uti", "urinary infection", "bladder infection",
        ],
        "min_signals": 2,
        "condition": "a urinary tract infection (UTI)",
        "mechanism": (
            "A UTI occurs when bacteria (most commonly E. coli from the gut) enter the urethra "
            "and infect the bladder. Classic symptoms are burning pain on urination, urgency and "
            "frequency, cloudy or strong-smelling urine, and lower pelvic discomfort. "
            "UTIs are much more common in women due to a shorter urethra. "
            "Simple UTIs respond well to antibiotics but need a urine test to confirm the bacteria "
            "and ensure the right antibiotic is used. If untreated, bacteria can travel up to the "
            "kidneys (pyelonephritis) — a more serious infection."
        ),
        "tests": (
            "urine routine examination and culture/sensitivity (mid-stream urine sample) "
            "— this confirms the infection and identifies the right antibiotic"
        ),
        "urgency": "within 24 hours — most UTIs respond quickly to the right antibiotic",
        "escalate_if": [
            "fever above 38°C / 100.4°F with urinary symptoms (suggests kidney involvement)",
            "back or flank pain (upper back, sides) along with urinary symptoms",
            "shaking, chills, nausea, or vomiting with urinary symptoms",
            "symptoms in a pregnant woman — requires prompt treatment",
            "symptoms in a man or child — UTIs are less common and need prompt evaluation",
            "symptoms not improving after 2 days of antibiotic treatment",
        ],
    },

    # =========================================================================
    # KIDNEY STONE
    # =========================================================================

    {
        "id": "kidney_stone",
        "is_life_threatening": False,
        "signals": [
            "severe flank pain", "flank pain radiating", "pain in side radiating",
            "pain from back to groin", "pain radiating to groin",
            "loin to groin pain", "colicky pain", "waves of pain",
            "kidney stone", "renal calculi", "urinary stone", "stone in kidney",
            "blood in urine and back pain", "pink urine and pain",
            "pain urinating and blood", "burning and back pain",
            "nausea and severe back pain", "vomiting and flank pain",
            "can't find comfortable position", "writhing in pain",
        ],
        "min_signals": 2,
        "condition": "a possible kidney stone (renal calculus)",
        "mechanism": (
            "A kidney stone forms when minerals in urine crystallise into a hard mass. "
            "When a stone moves from the kidney into the ureter, it causes one of the most "
            "severe pains in medicine — a colicky, wave-like agony that typically radiates "
            "from the lower back or flank down to the groin and inner thigh. "
            "Nausea and vomiting often accompany the pain. Blood in the urine is common "
            "as the stone scratches the ureter wall. Most stones under 5mm pass on their own; "
            "larger stones may require medical intervention."
        ),
        "tests": (
            "urine analysis (dipstick and microscopy for blood), "
            "non-contrast CT urogram (gold standard) or renal ultrasound, "
            "serum creatinine and electrolytes"
        ),
        "urgency": "TODAY — severe kidney stone pain usually requires medical pain management",
        "escalate_if": [
            "fever above 38°C / 100.4°F with flank pain — infected stone is a urological emergency",
            "inability to pass urine at all (stone blocking both ureters or single kidney)",
            "pain so severe it cannot be managed at home",
            "vomiting so severe you cannot keep fluids down",
        ],
    },

    # =========================================================================
    # PNEUMONIA / LOWER RESPIRATORY INFECTION
    # =========================================================================

    {
        "id": "pneumonia",
        "is_life_threatening": False,
        "signals": [
            "cough with yellow phlegm", "cough with green phlegm", "cough with yellow mucus",
            "cough with green mucus", "productive cough and fever",
            "chest pain when breathing", "pleuritic chest pain", "sharp chest pain breathing",
            "chest hurts when i breathe", "cough and chest tightness and fever",
            "shortness of breath and cough and fever",
            "high fever and cough", "high temperature and cough",
            "pneumonia", "lung infection", "chest infection",
            "coughing up brown mucus", "rust coloured sputum",
            "breathing fast and cough", "rapid breathing and fever",
        ],
        "min_signals": 2,
        "condition": "a lower respiratory tract infection, possibly pneumonia",
        "mechanism": (
            "Pneumonia is an infection of the lung tissue, usually caused by bacteria, viruses, "
            "or fungi. The infection causes the air sacs (alveoli) to fill with fluid and pus, "
            "reducing the lung's ability to exchange oxygen. Classic signs are productive cough "
            "(yellow/green/rust-coloured phlegm), fever, and sharp chest pain that worsens with "
            "breathing or coughing. Shortness of breath varies from mild to severe. "
            "Pneumonia can range from manageable at home to life-threatening — it must be assessed "
            "by a doctor to determine severity and the right treatment."
        ),
        "tests": (
            "chest X-ray, full blood count (CBC), CRP, sputum culture if available, "
            "and pulse oximetry (oxygen saturation check)"
        ),
        "urgency": "TODAY — do not delay, especially if breathing is affected",
        "escalate_if": [
            "blood oxygen level (SpO2) below 94% on pulse oximeter",
            "breathing rate above 30 breaths per minute at rest",
            "lips or fingertips turning blue",
            "confusion or extreme drowsiness",
            "fever above 39.5°C not responding to paracetamol",
            "chest pain that is severe or constant",
        ],
    },

    # =========================================================================
    # COVID-19 / POST-COVID
    # =========================================================================

    {
        "id": "covid_respiratory",
        "is_life_threatening": False,
        "signals": [
            "loss of smell", "loss of taste", "can't smell anything", "can't taste anything",
            "no smell", "no taste", "smell gone", "taste gone",
            "anosmia", "ageusia", "parosmia",
            "covid", "coronavirus", "tested positive", "covid positive",
            "post covid", "long covid", "covid symptoms",
            "fever and dry cough", "dry cough and fatigue and fever",
            "shortness of breath and fever and fatigue",
            "fatigue after covid", "weakness after covid",
        ],
        "min_signals": 1,
        "condition": "a COVID-19 infection or COVID-related condition",
        "mechanism": (
            "COVID-19 (caused by SARS-CoV-2) most commonly presents with fever, dry cough, "
            "fatigue, and the distinctive loss of smell and/or taste (anosmia/ageusia). "
            "Most people recover with rest and supportive care at home. "
            "High-risk individuals (elderly, immunocompromised, or with heart/lung/diabetes conditions) "
            "can deteriorate rapidly and need monitoring. Post-COVID syndrome (long COVID) "
            "can persist for weeks to months with fatigue, brain fog, breathlessness, and other symptoms."
        ),
        "tests": (
            "COVID-19 RAT (rapid antigen test) at home or RT-PCR for confirmation; "
            "pulse oximetry daily if symptomatic; chest X-ray if breathless"
        ),
        "urgency": "isolate and rest; see a doctor within 24 hours if high-risk or worsening",
        "escalate_if": [
            "blood oxygen level (SpO2) below 94% or dropping",
            "breathlessness that prevents normal speech or activity",
            "persistent chest pain or pressure",
            "confusion, inability to stay awake, or blue lips",
            "fever above 39°C not responding to paracetamol after 3 days",
        ],
    },

    # =========================================================================
    # PANIC ATTACK (to distinguish from cardiac emergency)
    # =========================================================================

    {
        "id": "panic_attack",
        "is_life_threatening": False,
        "signals": [
            "panic attack", "anxiety attack",
            "racing heart and anxiety", "pounding heart and anxiety",
            "chest tightness and anxiety", "heart racing and scared",
            "heart pounding out of chest", "heart fluttering anxiety",
            "shortness of breath and anxiety", "can't breathe and anxious",
            "tingling hands and anxiety", "numbness hands anxiety",
            "feeling of doom", "feeling i'm going to die",
            "intense fear suddenly", "overwhelming fear suddenly",
            "shaking and anxious", "trembling anxiety",
            "dizzy and anxious", "lightheaded and very anxious",
            "sweating and anxious", "sweating and fear",
        ],
        "min_signals": 2,
        "condition": "a panic attack — an intense burst of anxiety with physical symptoms",
        "mechanism": (
            "A panic attack occurs when the body's fight-or-flight response triggers intensely "
            "without an actual physical threat. Adrenaline floods the body, causing a rapid or "
            "pounding heart, shortness of breath, chest tightness, tingling, dizziness, and "
            "an overwhelming sense of dread or feeling that you are dying. "
            "These physical sensations are real and frightening — but panic attacks are not "
            "dangerous and typically peak within 10 minutes. The challenge is that these symptoms "
            "overlap with heart attacks: a doctor visit is important to rule out a cardiac cause "
            "if this is a first episode or if you have any heart risk factors."
        ),
        "tests": (
            "ECG (electrocardiogram) and basic blood tests to rule out cardiac and thyroid causes "
            "— especially important for a first episode"
        ),
        "urgency": "see a doctor within 24–48 hours for a first episode; manage safely at home if previously diagnosed",
        "escalate_if": [
            "chest pain radiating to the left arm, jaw, or back (could be cardiac — go to ER)",
            "this is your first episode and you are over 40 or have cardiac risk factors",
            "symptoms are not improving after 20–30 minutes",
            "you faint or lose consciousness",
            "you have a history of heart disease",
        ],
    },

    # =========================================================================
    # DEPRESSION (non-crisis — needs support and professional care)
    # =========================================================================

    {
        "id": "depression",
        "is_life_threatening": False,
        "is_mental_health": True,
        "signals": [
            "feeling depressed", "depression", "persistent sadness",
            "sad all the time", "crying all the time", "can't stop crying",
            "no motivation", "lost interest in everything", "nothing feels good anymore",
            "can't enjoy anything", "anhedonia", "empty feeling",
            "sleeping too much", "can't sleep and sad", "insomnia and depressed",
            "feeling worthless", "feeling useless", "no purpose",
            "hopeless", "feeling hopeless", "future feels pointless",
            "withdrawing from people", "isolating myself", "don't want to leave house",
            "can't concentrate", "brain fog and sad", "memory problems and depression",
            "don't feel like eating", "no appetite and sad", "lost appetite",
        ],
        "min_signals": 2,
        "condition": "you may be experiencing depression — and what you're feeling is real and deserves support",
        "mechanism": (
            "Depression is not sadness or weakness — it is a real medical condition that affects "
            "how the brain regulates mood, energy, sleep, appetite, and the ability to experience "
            "pleasure. It is extremely common and very treatable. Reaching out is the hardest and "
            "most important first step."
        ),
        "tests": "a conversation with a doctor, psychologist, or counsellor is the most important step",
        "urgency": "when you're ready — but please don't carry this alone",
        "escalate_if": [
            "you are having any thoughts of harming yourself or ending your life — please call iCall (9152987821) or Vandrevala (1860-2662-345) right now",
            "you have stopped eating, drinking, or caring for yourself",
            "you are unable to function at work, school, or home",
        ],
    },

    # =========================================================================
    # DVT — DEEP VEIN THROMBOSIS
    # =========================================================================

    {
        "id": "dvt",
        "is_life_threatening": False,
        "signals": [
            "one leg swollen", "one leg swelling", "leg swelling one side",
            "one calf swollen", "calf swelling", "swollen calf",
            "leg painful and swollen", "leg red and swollen", "leg warm and swollen",
            "deep vein thrombosis", "dvt", "blood clot in leg",
            "leg pain after flight", "leg pain after long journey",
            "leg pain after surgery", "leg cramping one side",
            "swollen leg after flying", "swollen leg after bed rest",
        ],
        "min_signals": 2,
        "condition": "a possible deep vein thrombosis (DVT) — a blood clot in a deep vein",
        "mechanism": (
            "A DVT occurs when a blood clot forms in a deep vein, usually in the calf or thigh. "
            "Classic signs are swelling, redness, warmth, and pain in ONE leg. "
            "Risk factors include long flights or car journeys, recent surgery, prolonged bed rest, "
            "pregnancy, and certain medications. The critical danger is that the clot can break off "
            "and travel to the lungs (pulmonary embolism) — a life-threatening emergency. "
            "DVT must not be ignored or treated with home remedies — it requires anticoagulant treatment."
        ),
        "tests": (
            "D-dimer blood test and Doppler ultrasound of the affected leg — "
            "done urgently in a hospital or vascular clinic"
        ),
        "urgency": "TODAY — do not delay or self-treat",
        "escalate_if": [
            "sudden shortness of breath, chest pain, or rapid heart rate — call 112/911 immediately (possible pulmonary embolism)",
            "coughing up blood",
            "feeling faint or collapsing",
            "very rapid deterioration of symptoms in the leg",
        ],
    },
]


def detect_symptom_cluster(query: str) -> Optional[dict]:
    """
    Scan the user's query for known serious symptom cluster patterns.

    Conservative by design: requires multiple distinct signal matches to avoid
    false positives on common single symptoms like fatigue or headache alone.
    Life-threatening and mental health clusters with min_signals=1 are intentionally
    sensitive — a single mention of chest pain or suicidal ideation is enough.

    Returns the best-matching cluster dict, or None if no serious cluster is found.
    Life-threatening clusters are prioritised over non-life-threatening ones.
    """
    query_lower = query.lower()

    best_match = None
    best_score = 0
    best_is_critical = False

    for cluster in TIER2_CLUSTERS:
        matches = sum(1 for signal in cluster["signals"] if signal in query_lower)
        if matches < cluster["min_signals"]:
            continue

        is_critical = cluster.get("is_life_threatening", False) or cluster.get("is_mental_health", False)

        # Prioritise: critical clusters beat non-critical; within the same tier, more matches wins
        if is_critical and not best_is_critical:
            best_match = cluster
            best_score = matches
            best_is_critical = True
        elif is_critical == best_is_critical and matches > best_score:
            best_match = cluster
            best_score = matches

    if best_match:
        tier = (
            "LIFE-THREATENING" if best_match.get("is_life_threatening")
            else "MENTAL HEALTH CRISIS" if best_match.get("is_mental_health")
            else "SERIOUS"
        )
        logger.info(
            f"Symptom cluster detected [{tier}]: '{best_match['id']}' "
            f"({best_score} signals matched) | Query: {query[:80]}..."
        )
    return best_match


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE TEMPLATES
# Three templates for three distinct response modes.
# ─────────────────────────────────────────────────────────────────────────────

_EMERGENCY_PROMPT = """You are ServVia — a calm, clear-headed emergency response assistant. The person reading this may be in immediate danger. Every word counts.

The person described: {symptoms}

This is consistent with: {condition}

What is happening:
{mechanism}

Write an urgent, calm, and clear emergency response. Do NOT be verbose. Lead immediately with action.

Use this EXACT structure:

## This is a Medical Emergency

[One sentence: acknowledge what they described and why it needs immediate action right now.]

## Call Emergency Services Now
**Dial 112 (India ambulance) / 102 (EMRI) / 911 (international)**
Do not drive yourself. Get someone to call while you focus on the person.

## Do This Right Now — While You Wait:
{immediate_actions_list}

## Go to ER Immediately If Any of These Occur:
{escalate_list}

---
*ServVia is a clinical decision-support tool. In emergencies, always defer to emergency services personnel.*

Rules:
- Total response must be under 350 words
- Use clear bold markdown for the action items
- Tone: calm, direct, authoritative — like a paramedic on the phone
- No lengthy explanations, no hedging, no disclaimers beyond the footer
- Do NOT suggest home remedies
Respond now:"""


_MENTAL_HEALTH_PROMPT = """You are ServVia — a compassionate, non-judgmental presence. The person reading this is in significant emotional pain right now.

What they shared: {symptoms}

Your ONLY goal is to make them feel heard and to connect them to immediate human support. Nothing else.

Write a warm, genuine, human response using this structure:

[Open with 2–3 sentences of genuine acknowledgment. Do not use clinical language. Do not immediately jump to resources. Acknowledge the pain directly — what they said matters, and you hear them.]

[One sentence that gently normalises: suicidal thoughts and overwhelming pain are a symptom of how much they're suffering — not a character flaw, and not permanent.]

## You Are Not Alone — Please Reach Out Right Now

**You must include ALL of these crisis lines EXACTLY as written — do not rephrase or omit the numbers:**
{immediate_actions_list}

[One sentence encouraging them to make one call — just one. They don't have to figure anything else out right now.]

## If You Have Already Hurt Yourself
Call emergency services — **112 or 102** — right now. Please.

---
Rules:
- Warm, real, human — not clinical
- No jargon, no medical advice, no home remedies
- CRITICAL: The crisis line numbers (9152987821, 1860-2662-345) MUST appear verbatim in your response
- Under 220 words total
- The goal is one thing: get them to pick up the phone
Respond now:"""


_CLUSTER_PROMPT = """You are ServVia — a warm, deeply knowledgeable clinical triage assistant. You are a trusted family doctor giving honest, precise advice to someone you care about.

The user is experiencing: {symptoms}

This symptom pattern is consistent with: {condition}

Clinical mechanism (explain this in plain language):
{mechanism}

Chronobiology context (integrate naturally if relevant):
{bio_context}

Your task: Write a compassionate, clear, and genuinely helpful response — in this exact order:

1. **Opening** (1–2 sentences): Acknowledge how worrying and exhausting these symptoms must feel. Be personal, not generic.

2. **Honest pattern recognition**: Name the possible condition with appropriate clinical hedging:
   - "These symptoms together are consistent with..."
   - "This combination is commonly seen in..."
   - "Your body may be signalling..."
   Never say "You definitely have X." Always hedge. Then give a 2-3 sentence plain-language explanation of WHY this pattern points here (use the mechanism above).

3. **What to do**: See a doctor {urgency}. Be specific about urgency and why.

4. **Ask the doctor specifically for**: {tests}
   Name the actual tests, don't just say "get blood tests."

5. **What to document before your appointment**:
   - When each symptom started and its trajectory (improving / worsening / stable)
   - Any relevant family history
   - All current medications and supplements (with doses)
   - Recent travel, diet changes, sleep changes, or significant stress
   - Any previous similar episodes

6. **Safe comfort measures while you wait** (2–3 only — nothing that masks the diagnosis):
   - Adequate hydration (water, clear fluids, ORS if dehydrated)
   - Rest — reduce physical and mental exertion
   - Paracetamol (acetaminophen) at standard doses for fever or pain — safe for most people
   - Do NOT start herbal remedies or supplements until the cause is confirmed — some can interfere with test results or worsen the underlying condition

7. **Go to the ER or call emergency services immediately if**:
{escalate_list}

Format with clear markdown. Warm but direct. This is someone who trusts you with their health — give them the clear guidance a good doctor would give a family member.

Respond now:"""


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT RESPONSE GENERATOR (single LLM call — no RAG, no multi-agent)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_cluster_response(
    cluster: dict,
    user_symptoms: str,
    bio_context: str = "",
) -> str:
    """
    Generate a direct, targeted response for a detected symptom cluster.
    Single LLM call — no RAG retrieval, no multi-agent loop.

    Routes to the correct prompt template based on cluster type:
      - is_life_threatening=True  → _EMERGENCY_PROMPT
      - is_mental_health=True     → _MENTAL_HEALTH_PROMPT
      - neither                   → _CLUSTER_PROMPT
    """
    from legacy_agriculture.rag_service.openai_service import make_openai_request

    is_emergency = cluster.get("is_life_threatening", False)
    is_mental_health = cluster.get("is_mental_health", False)

    escalate_list = "\n".join(f"  - {e}" for e in cluster["escalate_if"])

    if is_emergency:
        immediate_actions = cluster.get("immediate_actions", [])
        immediate_actions_list = "\n".join(f"- **{a}**" for a in immediate_actions)
        prompt = _EMERGENCY_PROMPT.format(
            symptoms=user_symptoms,
            condition=cluster["condition"],
            mechanism=cluster["mechanism"],
            immediate_actions_list=immediate_actions_list,
            escalate_list=escalate_list,
        )
        temperature = 0.1

    elif is_mental_health:
        immediate_actions = cluster.get("immediate_actions", [])
        immediate_actions_list = "\n".join(f"- {a}" for a in immediate_actions)
        prompt = _MENTAL_HEALTH_PROMPT.format(
            symptoms=user_symptoms,
            immediate_actions_list=immediate_actions_list,
        )
        temperature = 0.4  # slightly warmer for genuine-feeling empathy

    else:
        prompt = _CLUSTER_PROMPT.format(
            symptoms=user_symptoms,
            condition=cluster["condition"],
            mechanism=cluster["mechanism"],
            urgency=cluster["urgency"],
            tests=cluster["tests"],
            escalate_list=escalate_list,
            bio_context=bio_context or "No chronobiology context available.",
        )
        temperature = 0.2

    response, exception, retries = await make_openai_request(prompt, temperature=temperature)

    if response and response.choices:
        return response.choices[0].message.content.strip()

    logger.error(f"Cluster response generation failed: {exception}")

    # ── Failsafe: deterministic fallback if LLM call fails ──────────────────
    if is_emergency:
        actions = cluster.get("immediate_actions", [])
        actions_str = "\n".join(f"- {a}" for a in actions[:4])
        return (
            f"## This is a Medical Emergency\n\n"
            f"The symptoms you are describing are consistent with **{cluster['condition']}**.\n\n"
            f"## Call Emergency Services Now\n"
            f"**Dial 112 / 102 / 911 immediately. Do not drive yourself.**\n\n"
            f"## Do This While You Wait:\n{actions_str}\n\n"
            f"*ServVia — clinical decision support only. Follow emergency services instructions.*"
        )

    if is_mental_health:
        return (
            f"What you're going through sounds incredibly painful, and I want you to know "
            f"you are not alone in this moment.\n\n"
            f"**Please reach out right now:**\n"
            f"- iCall India: **9152987821**\n"
            f"- Vandrevala Foundation: **1860-2662-345** (24/7, free)\n\n"
            f"If you have already hurt yourself, call **112 or 102** immediately.\n\n"
            f"You deserve support. Please make one call."
        )

    escalate_str = "; ".join(cluster["escalate_if"][:3])
    return (
        f"I'm concerned about the combination of symptoms you're describing. "
        f"They are consistent with {cluster['condition']} and need proper medical "
        f"evaluation {cluster['urgency']}.\n\n"
        f"Please ask your doctor about: **{cluster['tests']}**\n\n"
        f"**Go to the ER immediately if:** {escalate_str}."
    )
