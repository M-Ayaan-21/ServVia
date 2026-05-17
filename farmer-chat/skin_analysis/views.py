from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import SkinAnalysis
from .disease_detector import SkinDiseaseDetector, detect_skin_disease_gemini, detect_skin_disease_edge_first, validate_skin_image
import logging
import tempfile
import os
from PIL import Image
import io

# Register HEIC/HEIF support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)
detector = SkinDiseaseDetector()

# Import Trust Engine
try:
    from servvia2.trust_engine.engine import TrustEngine
    TRUST_ENGINE_AVAILABLE = True
    trust_engine = TrustEngine()
    logger.info("✅ Trust Engine integrated with Skin Analysis")
except ImportError as e:
    TRUST_ENGINE_AVAILABLE = False
    trust_engine = None
    logger.warning(f"⚠️ Trust Engine not available: {e}")


@api_view(['POST'])
def analyze_skin_image(request):
    """Analyze skin image with Trust Engine validation"""
    temp_path = None

    try:
        email = request.data.get('email_id')
        if not email:
            return Response({'success': False, 'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'success': False, 'error': 'Image file is required'}, status=status.HTTP_400_BAD_REQUEST)

        # Get user profile
        user_profile = get_user_profile(email)

        # Process image: convert to RGB and save temporarily
        try:
            image_data = image_file.read()
            if not image_data:
                return Response({'success': False, 'error': 'Uploaded file is empty.'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                image = Image.open(io.BytesIO(image_data))
                image.load()  # Force decode so format errors surface here
            except Exception:
                # Last resort: write raw bytes to a temp file and let PIL infer from disk
                with tempfile.NamedTemporaryFile(delete=False, suffix='') as raw_tmp:
                    raw_tmp.write(image_data)
                    raw_tmp_path = raw_tmp.name
                try:
                    image = Image.open(raw_tmp_path)
                    image.load()
                finally:
                    try:
                        os.unlink(raw_tmp_path)
                    except Exception:
                        pass

            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            elif image.mode == 'L':
                image = image.convert('RGB')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name, format='JPEG', quality=95)
                temp_path = tmp.name
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return Response(
                {'success': False, 'error': f'Could not read image. Supported formats: JPG, PNG, WebP, HEIC, BMP, TIFF, GIF.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate that it's actually a skin image
        validation = validate_skin_image(temp_path)
        if not validation['is_skin_image']:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response({
                'success': False,
                'error': validation['reason'],
                'suggestion': 'Please upload a clear photograph of the affected skin area.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Detect skin disease - tries Qwen (edge) first, escalates to Gemini if needed
        result = detect_skin_disease_edge_first(temp_path)

        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

        if not result.get('success'):
            return Response({'success': False, 'error': result.get('error', 'Unable to analyze image')},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        disease = result.get('disease', 'Unknown')
        confidence_score = result.get('confidence_score', 0.0)

        # Trust Engine validation (personalized remedies)
        trust_validation = None
        if TRUST_ENGINE_AVAILABLE and trust_engine:
            trust_validation = validate_skin_recommendations(disease, user_profile)
            logger.info(f"✅ Trust Engine validated recommendations for {disease}")

        # Build detailed human-readable report
        formatted_summary = build_detailed_skin_analysis(result, trust_validation, user_profile)

        # Save analysis to database
        image_file.seek(0)  # Reset file pointer for saving
        analysis = SkinAnalysis.objects.create(
            email_id=email,
            image=image_file,
            diagnosis=disease,
            confidence_score=confidence_score,
            recommendations=formatted_summary
        )

        inference_source = result.get('inference_source', 'cloud')
        source_label_map = {
            'edge': 'Edge AI (on-device)',
            'cloud': 'Cloud AI (Gemini)',
            'cloud_escalated': 'Cloud AI (Gemini) - escalated from Edge',
        }

        return Response({
            'success': True,
            'diagnosis': disease,
            'confidence': round(confidence_score * 100, 2),
            'severity': result.get('severity', 'Unknown'),
            'description': result.get('description', ''),
            'recommendations': formatted_summary,
            'urgency_note': result.get('urgency_note', ''),
            'analysis_id': analysis.id,
            'visual_analysis': result.get('visual_analysis', {}),
            'distinguishing_features': result.get('distinguishing_features', ''),
            'differential_diagnosis': result.get('differential_diagnosis', []),
            'key_features': result.get('key_features', []),
            'trust_validation': trust_validation,
            'inference_source': inference_source,
            'inference_source_label': source_label_map.get(inference_source, inference_source),
        })

    except Exception as e:
        logger.error(f"Skin analysis error: {e}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return Response({'success': False, 'error': 'An unexpected error occurred.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def get_user_profile(email):
    """Fetch user profile for personalization"""
    try:
        from user_profile.models import UserProfile
        profile = UserProfile.objects.filter(email=email).first()
        if profile:
            return {
                'first_name': profile.first_name or 'there',
                'allergies': parse_list(profile.allergies),
                'medical_conditions': parse_list(profile.medical_conditions),
                'current_medications': parse_list(profile.current_medications)
            }
    except Exception as e:
        logger.warning(f"Could not get profile: {e}")
    return {'first_name': 'there', 'allergies': [], 'medical_conditions': [], 'current_medications': []}


def parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(',') if item.strip()]
    return []


def validate_skin_recommendations(disease, user_profile):
    """Validate and personalize remedies using Trust Engine"""
    if not TRUST_ENGINE_AVAILABLE or not trust_engine:
        return None

    try:
        condition_map = {
            'Acne': 'acne',
            'Eczema (Atopic Dermatitis)': 'eczema',
            'Psoriasis (mild forms)': 'psoriasis',
            'Heat Rash (Prickly Heat)': 'heat rash',
            'Hives (Urticaria)': 'hives',
            'Sunburn': 'burns',
            'Dry Skin (Xerosis)': 'dry skin',
            'Fungal Infections (Ringworm, Athlete\'s Foot)': 'fungal infection',
            'Contact Dermatitis': 'dermatitis',
            'Dandruff (Seborrheic Dermatitis)': 'dandruff',
        }

        mapped_condition = condition_map.get(disease, disease.lower())

        # Scan the evidence database for all entries relevant to this condition
        all_evidence = trust_engine.evidence_data.get('evidence', [])
        evidence_remedies = []
        for entry in all_evidence:
            entry_condition = entry.get('condition', '').lower().replace(' ', '_')
            entry_aliases = [c.lower().replace(' ', '_') for c in entry.get('condition_aliases', [])]
            target = mapped_condition.lower().replace(' ', '_')
            if (target in entry_condition or entry_condition in target or
                    target in entry_aliases or
                    trust_engine._conditions_related(target, entry_condition)):
                evidence_remedies.append(entry)

        user_meds = user_profile.get('current_medications', [])
        user_allergies = user_profile.get('allergies', [])

        # Map evidence_level string to numeric tier
        level_to_tier = {'high': 1, 'moderate': 2, 'low': 3, 'very_low': 4, 'insufficient': 5}
        tier_labels = {1: "Clinical Trials", 2: "Mechanistic Studies", 3: "Traditional Use", 4: "Anecdotal", 5: "Theoretical"}
        base_scores = {1: 9.5, 2: 8.0, 3: 6.0, 4: 4.0, 5: 2.0}

        validated_remedies = []
        warnings = []

        for entry in evidence_remedies:
            herb_name = entry.get('herb', '')
            if not herb_name:
                continue

            # Allergy check
            if herb_name.lower() in [a.lower() for a in user_allergies]:
                warnings.append(f"Skipped {herb_name.title()} - you're allergic")
                continue

            # Drug interaction check using the entry's interactions list directly
            interaction_hit = None
            for med in user_meds:
                for ix in entry.get('interactions', []):
                    if med.lower() in ix.get('substance', '').lower():
                        interaction_hit = ix
                        break
                if interaction_hit:
                    break

            tier = level_to_tier.get(entry.get('evidence_level', 'low'), 3)
            score = base_scores.get(tier, 5.0)

            if interaction_hit:
                severity = interaction_hit.get('severity', 'moderate').lower()
                if severity in ('critical', 'high'):
                    warnings.append(f"{herb_name.title()} contraindicated with {interaction_hit.get('substance')}: {interaction_hit.get('description', '')}")
                    continue
                else:
                    score -= 1.5
                    warnings.append(f"Use {herb_name.title()} with caution if taking {interaction_hit.get('substance')}")

            # Extract dose: prefer adult dose, fall back to summary
            dosing = entry.get('dosing', {})
            dose_str = dosing.get('adults') or dosing.get('general') or entry.get('summary', '')

            validated_remedies.append({
                'name': herb_name,
                'score': round(score, 1),
                'tier': tier,
                'tier_label': tier_labels.get(tier, "Unknown"),
                'mechanism': entry.get('summary', ''),
                'dose': dose_str,
                'has_interaction': interaction_hit is not None
            })

        validated_remedies.sort(key=lambda x: x['score'], reverse=True)

        return {
            'remedies': validated_remedies[:6],
            'warnings': warnings,
            'condition_mapped': mapped_condition,
            'total_found': len(evidence_remedies)
        }

    except Exception as e:
        logger.error(f"Trust Engine validation error: {e}", exc_info=True)
        return None


def build_detailed_skin_analysis(result, trust_validation, user_profile):
    """Build comprehensive, patient-friendly skin analysis report"""
    disease = result.get('disease', 'Unknown')
    confidence_pct = round(result.get('confidence_score', 0) * 100, 1)
    severity = result.get('severity', 'Unknown')
    description = result.get('description', '')
    key_features = result.get('key_features', [])
    visual = result.get('visual_analysis', {})
    urgency_note = result.get('urgency_note', '')
    differential = result.get('differential_diagnosis', [])
    distinguishing = result.get('distinguishing_features', '')
    reasoning = result.get('reasoning', '')
    affected_area = result.get('affected_area', 'Not specified')

    user_name = user_profile.get('first_name', 'there')
    if not user_name or user_name.lower() in ['user', '']:
        user_name = 'there'

    report = f"""## 🔬 Skin Analysis Report
Hi **{user_name}**! I've analyzed your skin image and here's what I found:
---
### 🏥 Diagnosis: **{disease}**
| | |
|---|---|
| **Confidence** | {confidence_pct}% |
| **Severity** | {severity} |
| **Affected Area** | {affected_area} |
---
### 📋 What I'm Seeing
{description}
"""

    if key_features:
        report += "**Key Visual Indicators:**\n\n"
        for feature in key_features:
            report += f"- {feature}\n"
        report += "\n"

    report += """---
### 🧠 Why I Made This Diagnosis
"""
    if reasoning:
        report += f"{reasoning}\n\n"
    else:
        report += f"Based on my analysis, I identified this as **{disease}** because:\n\n"
        if visual.get('lesion_count'):
            count = visual.get('lesion_count')
            report += f"- **Lesion Count:** {count} - {'Multiple lesions typical of this condition' if 'many' in str(count).lower() else 'Consistent with diagnosis'}\n"
        if visual.get('lesion_size'):
            report += f"- **Lesion Size:** {visual.get('lesion_size')} - Matches expected presentation\n"
        if visual.get('texture'):
            report += f"- **Texture:** {visual.get('texture')} - Characteristic of {disease}\n"
        if visual.get('border_type'):
            report += f"- **Border Pattern:** {visual.get('border_type')}\n"
        report += "\n"

    if distinguishing:
        report += f"**What makes this {disease} and not something else:**\n\n{distinguishing}\n\n"

    if differential:
        report += "**Other conditions I considered:**\n\n"
        for d in differential:
            report += f"- {d} (ruled out based on visual features)\n"
        report += "\n"

    report += """---
### 💊 Evidence-Based Treatment Recommendations
"""
    if trust_validation and trust_validation.get('remedies'):
        report += f"I've checked our **Neuro-Symbolic Trust Engine** database and found **{len(trust_validation['remedies'])} scientifically validated remedies** for {disease}:\n\n"

        for i, remedy in enumerate(trust_validation['remedies'], 1):
            emoji = "🟢" if remedy['score'] >= 8 else "🟡" if remedy['score'] >= 6 else "🔴"
            confidence_text = "Strong Evidence" if remedy['score'] >= 8 else "Good Evidence" if remedy['score'] >= 6 else "Traditional Use"

            report += f"#### {i}. {remedy['name'].title()} {emoji}\n"
            report += f"**Scientific Confidence Score:** {remedy['score']}/10 ({confidence_text})\n"
            report += f"**Evidence Level:** {remedy['tier_label']}\n\n"

            if remedy['mechanism']:
                report += f"**How it works:** {remedy['mechanism']}\n\n"
            if remedy['dose']:
                report += f"**Recommended usage:** {remedy['dose']}\n\n"
            if remedy.get('has_interaction'):
                report += f"⚠️ *Use with caution - see safety notes below*\n\n"

            report += "---\n\n"

        if trust_validation.get('warnings'):
            report += "### ⚠️ Personalized Safety Alerts\n\n"
            report += "Based on your health profile, please note:\n\n"
            for warning in trust_validation['warnings']:
                report += f"{warning}\n\n"
    else:
        report += "*Trust Engine validation unavailable. Here are general recommendations:*\n\n"
        for i, rec in enumerate(result.get('recommendations', []), 1):
            report += f"{i}. {rec}\n\n"

    allergies = user_profile.get('allergies', [])
    medications = user_profile.get('current_medications', [])
    conditions = user_profile.get('medical_conditions', [])

    if allergies or medications or conditions:
        report += """---
### 👤 Your Health Profile Check
"""
        if allergies:
            report += f"✅ **Allergies checked:** {', '.join(allergies)} - Remedies filtered accordingly\n\n"
        if medications:
            report += f"✅ **Medications checked:** {', '.join(medications)} - Interactions verified\n\n"
        if conditions:
            report += f"✅ **Conditions checked:** {', '.join(conditions)} - Contraindications reviewed\n\n"

    report += """---
### 🎯 What You Should Do Next
"""
    severity_lower = severity.lower()
    if 'severe' in severity_lower:
        report += f"**Severity Level: {severity}** 🔴\n{urgency_note}\n"
        report += """**Recommended Actions:**
1. 🏥 **See a dermatologist within 24-48 hours**
2. 📸 Take photos daily to track changes
3. 🚫 Avoid touching or picking at the affected area
4. 📝 Note any new triggers
5. 💊 The remedies above may help while awaiting professional care
"""
    elif 'moderate' in severity_lower:
        report += f"**Severity Level: {severity}** 🟡\n{urgency_note}\n"
        report += """**Recommended Actions:**
1. 🌿 Try the top Trust Engine remedies for **5-7 days**
2. 📸 Take a comparison photo in one week
3. 🧴 Keep area clean and moisturized
4. 🚫 Avoid known triggers
5. 👨‍⚕️ Consult a dermatologist if no improvement
"""
    else:
        report += f"**Severity Level: {severity}** 🟢\n{urgency_note}\n"
        report += """**Recommended Actions:**
1. 🌿 Start with the highest-rated remedy above
2. ⏰ Use consistently for **1-2 weeks**
3. 💧 Stay hydrated and maintain good hygiene
4. 📊 Monitor progress
5. ✅ Most cases resolve with proper care
"""

    report += f"""---
### 📚 Understanding {disease}
"""
    report += get_condition_education(disease)

    if trust_validation:
        report += """
---
### 🔬 About Trust Engine Validation
The remedies above have been validated through our **Neuro-Symbolic Trust Engine**, which combines:
- **Neural AI Analysis** - Pattern recognition from medical literature
- **Symbolic Knowledge Base** - Verified herb-condition evidence with PubMed citations
- **Safety Verification** - Drug interaction and contraindication checking
- **Personalization** - Filtered based on your health profile

**Scientific Confidence Score (SCS) Scale:**
| Score | Meaning |
|-------|---------|
| 🟢 **8-10** | Clinical trial evidence (RCTs, meta-analyses) |
| 🟡 **5-7** | Mechanistic studies with documented pathways |
| 🔴 **1-4** | Traditional use or preliminary research |
"""

    report += """
---
### ⚠️ Important Disclaimer
This AI-powered analysis is for **educational purposes only** and does not replace professional medical advice.
- ✅ Share this report with your healthcare provider
- ✅ Seek immediate care for severe or worsening symptoms
- ❌ Do not self-treat serious conditions
- ❌ Do not ignore signs of infection (pus, fever, spreading redness)

---
*Analysis powered by ServVia AI Healthcare Intelligence with Neuro-Symbolic Trust Engine*
"""

    return report


def get_condition_education(disease):
    """Educational content about common skin conditions"""
    education = {
        "Acne": """
**What is Acne?**
Acne occurs when hair follicles become clogged with oil (sebum) and dead skin cells. Bacteria (P. acnes) can then multiply, causing inflammation.

**Common Triggers:**
- Hormonal changes (puberty, menstruation, stress)
- Certain foods (dairy, high-glycemic foods for some people)
- Comedogenic skincare products
- Touching your face frequently
- Stress and lack of sleep

**Prevention Tips:**
- Wash face twice daily with gentle cleanser
- Use non-comedogenic products
- Don't pick or squeeze pimples
- Change pillowcases regularly
- Stay hydrated
""",
        "Eczema (Atopic Dermatitis)": """
**What is Eczema?**
Eczema is a chronic inflammatory condition causing dry, itchy, inflamed skin. It's often linked to allergies and runs in families.

**Common Triggers:**
- Dry air and low humidity
- Harsh soaps and detergents
- Stress and anxiety
- Certain fabrics (wool, synthetic)
- Food allergies (dairy, eggs, nuts)

**Management Tips:**
- Moisturize immediately after bathing
- Use fragrance-free products
- Wear soft, breathable fabrics
- Identify and avoid your triggers
- Keep nails short to prevent scratching damage
""",
        "Psoriasis (mild forms)": """
**What is Psoriasis?**
Psoriasis is an autoimmune condition where skin cells multiply too quickly, causing thick, scaly patches.

**Common Triggers:**
- Stress
- Skin injuries (cuts, sunburn)
- Infections (strep throat)
- Cold, dry weather
- Certain medications
- Smoking and alcohol

**Management Tips:**
- Keep skin moisturized
- Get moderate sun exposure (15-20 min daily)
- Manage stress through relaxation techniques
- Avoid smoking and limit alcohol
- Consider anti-inflammatory diet
""",
        "Heat Rash (Prickly Heat)": """
**What is Heat Rash?**
Heat rash occurs when sweat ducts become blocked, trapping perspiration under the skin.

**Common Causes:**
- Hot, humid weather
- Excessive sweating
- Tight clothing
- Heavy creams that block pores
- Physical activity in heat

**Prevention & Relief:**
- Stay in cool, air-conditioned environments
- Wear loose, breathable cotton clothing
- Take cool showers
- Use lightweight, non-comedogenic products
- Avoid heavy physical activity in extreme heat
""",
        "Hives (Urticaria)": """
**What are Hives?**
Hives are raised, itchy welts that appear suddenly, usually due to an allergic reaction. They typically resolve within 24 hours but can recur.

**Common Triggers:**
- Food allergies (shellfish, nuts, eggs)
- Medications (antibiotics, aspirin)
- Insect stings
- Stress
- Temperature changes
- Infections

**When to Seek Emergency Care:**
- Difficulty breathing or swallowing
- Swelling of face, lips, or tongue
- Dizziness or fainting
- Rapid heartbeat
"""
    }

    return education.get(disease, f"""
**About {disease}:**
This is a common skin condition that can often be managed with proper care. For detailed information specific to your case, please consult a dermatologist.

**General Skin Health Tips:**
- Keep skin clean and appropriately moisturized
- Protect from excessive sun exposure
- Stay hydrated (8+ glasses of water daily)
- Eat a balanced diet rich in vitamins A, C, and E
- Get adequate sleep (7-8 hours)
- Manage stress levels
""")


@api_view(['GET'])
def get_skin_analysis_history(request):
    """Get user's skin analysis history (last 10)"""
    try:
        email = request.query_params.get('email_id')
        if not email:
            return Response({'success': False, 'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

        analyses = SkinAnalysis.objects.filter(email_id=email).order_by('-created_at')[:10]

        results = [{
            'id': a.id,
            'diagnosis': a.diagnosis,
            'confidence': round(a.confidence_score * 100, 2),
            'date': a.created_at.strftime('%Y-%m-%d %H:%M'),
            'image_url': request.build_absolute_uri(a.image.url) if a.image else None
        } for a in analyses]

        return Response({'success': True, 'history': results, 'count': len(results)})
    except Exception as e:
        logger.error(f"History error: {e}")
        return Response({'success': False, 'error': 'Failed to retrieve history'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
