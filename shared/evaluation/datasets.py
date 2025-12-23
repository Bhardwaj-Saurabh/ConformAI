"""
Evaluation Datasets for ConformAI

Pre-defined evaluation datasets for EU AI compliance questions.
"""

# EU AI Act Classification Questions
EU_AI_ACT_CLASSIFICATION = [
    {
        "input": "What are high-risk AI systems under the EU AI Act?",
        "expected_output": "High-risk AI systems are AI systems that pose significant risks to health, safety, or fundamental rights. They include: biometric identification systems, AI in critical infrastructure, educational/vocational training, employment, essential services, law enforcement, migration/asylum/border control, and administration of justice. These systems must meet strict requirements before deployment.",
        "metadata": {
            "category": "classification",
            "regulation": "EU AI Act",
            "article": "Article 6",
            "difficulty": "medium",
        },
    },
    {
        "input": "Are AI systems used in recruitment considered high-risk?",
        "expected_output": "Yes, AI systems used in employment, workers management, and access to self-employment are classified as high-risk under the EU AI Act. This includes AI used for recruitment, task allocation, monitoring, and evaluation of workers.",
        "metadata": {
            "category": "classification",
            "regulation": "EU AI Act",
            "article": "Annex III",
            "difficulty": "easy",
        },
    },
    {
        "input": "What are prohibited AI practices under the EU AI Act?",
        "expected_output": "Prohibited AI practices include: subliminal manipulation causing harm, exploitation of vulnerabilities of specific groups, social scoring by public authorities, real-time remote biometric identification in public spaces for law enforcement (with exceptions), emotion recognition in workplace and educational institutions, and biometric categorization to infer sensitive characteristics.",
        "metadata": {
            "category": "prohibition",
            "regulation": "EU AI Act",
            "article": "Article 5",
            "difficulty": "medium",
        },
    },
]

# GDPR and AI Training Data
GDPR_AI_DATA = [
    {
        "input": "Can I train an AI model on personal data under GDPR?",
        "expected_output": "Training AI models on personal data is permitted under GDPR if you have a valid legal basis (consent, legitimate interest, contractual necessity, etc.), comply with data minimization and purpose limitation principles, conduct a Data Protection Impact Assessment (DPIA) if required, and ensure appropriate technical and organizational measures for data security.",
        "metadata": {
            "category": "data_processing",
            "regulation": "GDPR",
            "article": "Articles 5, 6, 35",
            "difficulty": "hard",
        },
    },
    {
        "input": "What rights do individuals have regarding AI systems processing their data?",
        "expected_output": "Individuals have the right to: access their personal data, rectification of inaccurate data, erasure (right to be forgotten), restriction of processing, data portability, object to processing including profiling, and not be subject to solely automated decision-making with legal or significant effects without human intervention.",
        "metadata": {
            "category": "rights",
            "regulation": "GDPR",
            "article": "Articles 15-22",
            "difficulty": "medium",
        },
    },
]

# Compliance Obligations
COMPLIANCE_OBLIGATIONS = [
    {
        "input": "What documentation must high-risk AI systems maintain?",
        "expected_output": "High-risk AI systems must maintain: technical documentation including system design, development process, datasets used, training methodologies, validation/testing procedures, risk management documentation, conformity assessment reports, post-market monitoring plans, and automatically generated logs of system events.",
        "metadata": {
            "category": "obligations",
            "regulation": "EU AI Act",
            "article": "Articles 11, 12",
            "difficulty": "hard",
        },
    },
    {
        "input": "Do I need human oversight for high-risk AI systems?",
        "expected_output": "Yes, high-risk AI systems must be designed with human oversight measures. This includes: enabling individuals to understand system capabilities and limitations, remaining aware of automation bias, being able to correctly interpret system output, and being able to override or interrupt the system when necessary.",
        "metadata": {
            "category": "obligations",
            "regulation": "EU AI Act",
            "article": "Article 14",
            "difficulty": "medium",
        },
    },
]

# Biometric Systems
BIOMETRIC_SYSTEMS = [
    {
        "input": "Is facial recognition allowed under EU law?",
        "expected_output": "Facial recognition depends on the use case. Real-time remote biometric identification in public spaces for law enforcement is generally prohibited with limited exceptions (searching for victims, preventing imminent threats, prosecuting serious crimes). Post-event facial recognition requires judicial authorization. Private use must comply with GDPR, requiring valid legal basis and appropriate safeguards.",
        "metadata": {
            "category": "biometrics",
            "regulation": "EU AI Act, GDPR",
            "article": "Article 5, Article 9 GDPR",
            "difficulty": "hard",
        },
    },
    {
        "input": "Can employers use emotion recognition AI?",
        "expected_output": "The EU AI Act prohibits emotion recognition AI in workplace and educational institutions, with limited exceptions. This is considered a high-risk practice that could infringe on fundamental rights. Employers must not deploy emotion recognition systems for monitoring workers or making employment decisions.",
        "metadata": {
            "category": "biometrics",
            "regulation": "EU AI Act",
            "article": "Article 5(1)(f)",
            "difficulty": "easy",
        },
    },
]

# Generative AI
GENERATIVE_AI = [
    {
        "input": "What obligations apply to generative AI models?",
        "expected_output": "General-purpose AI models must: provide technical documentation, comply with copyright law, publish summaries of training data. High-impact general-purpose AI models (systemic risk) have additional requirements: model evaluations, adversarial testing, risk tracking, reporting serious incidents, and ensuring adequate cybersecurity.",
        "metadata": {
            "category": "generative_ai",
            "regulation": "EU AI Act",
            "article": "Articles 52, 53",
            "difficulty": "medium",
        },
    },
]

# All datasets combined
ALL_DATASETS = {
    "eu-ai-act-classification": {
        "description": "Questions about EU AI Act risk classification",
        "items": EU_AI_ACT_CLASSIFICATION,
    },
    "gdpr-ai-data": {
        "description": "Questions about GDPR and AI training data",
        "items": GDPR_AI_DATA,
    },
    "compliance-obligations": {
        "description": "Questions about compliance obligations for AI systems",
        "items": COMPLIANCE_OBLIGATIONS,
    },
    "biometric-systems": {
        "description": "Questions about biometric AI systems",
        "items": BIOMETRIC_SYSTEMS,
    },
    "generative-ai": {
        "description": "Questions about generative AI obligations",
        "items": GENERATIVE_AI,
    },
    "comprehensive-eval": {
        "description": "Comprehensive evaluation set covering all topics",
        "items": (
            EU_AI_ACT_CLASSIFICATION
            + GDPR_AI_DATA
            + COMPLIANCE_OBLIGATIONS
            + BIOMETRIC_SYSTEMS
            + GENERATIVE_AI
        ),
    },
}


def get_dataset(dataset_name: str) -> dict:
    """
    Get a pre-defined evaluation dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dict with 'description' and 'items'
    """
    return ALL_DATASETS.get(dataset_name, {"description": "", "items": []})


def list_datasets() -> list[str]:
    """List all available dataset names."""
    return list(ALL_DATASETS.keys())
