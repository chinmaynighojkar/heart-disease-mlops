"""Single source of truth for EU AI Act citations used across the compliance module.

All text is taken from **Regulation (EU) 2024/1689** (the EU AI Act), Official
Journal version of 13 June 2024. Annex III and Annex IV are reproduced verbatim
(machine-readable excerpts) so that every downstream artifact -- the
requirement->evidence map, the Annex III risk classifier, and the Annex IV
technical-documentation generator -- cites from one place and cannot drift.

Primary source:   https://eur-lex.europa.eu/eli/reg/2024/1689/oj
Readable mirror:   https://artificialintelligenceact.eu/

Nothing in this file is legal advice. It is a structured transcription of the
Regulation for use as a documentation/tooling aid.
"""

from __future__ import annotations

REGULATION = {
    "id": "Regulation (EU) 2024/1689",
    "short_name": "EU AI Act",
    "oj_version": "Official Journal version of 13 June 2024",
    "eli": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
}

# ---------------------------------------------------------------------------
# Chapter III, Section 2 -- Requirements for high-risk AI systems (Arts 9-15).
# Summaries are faithful paraphrases of the operative obligations; article and
# paragraph numbers are exact so a reader can trace each back to the OJ text.
# ---------------------------------------------------------------------------
ARTICLES = {
    "Article 9": {
        "title": "Risk management system",
        "obligation": (
            "Establish, implement, document and maintain a continuous, iterative "
            "risk-management system across the lifecycle: identify and analyse known "
            "and reasonably foreseeable risks to health, safety and fundamental "
            "rights; estimate and evaluate risks in intended use and foreseeable "
            "misuse; adopt targeted risk-management measures; and test to identify "
            "the most appropriate measures (Art. 9(1)-(8))."
        ),
        "key_paragraphs": ["9(2)", "9(5)", "9(6)", "9(9)"],
    },
    "Article 10": {
        "title": "Data and data governance",
        "obligation": (
            "Training, validation and testing data sets must be subject to "
            "appropriate data-governance practices: relevant, sufficiently "
            "representative, and to the best extent possible free of errors and "
            "complete; examination for biases likely to affect health, safety or "
            "fundamental rights or lead to prohibited discrimination; and handling "
            "of data gaps or shortcomings (Art. 10(2)-(5))."
        ),
        "key_paragraphs": ["10(2)", "10(3)", "10(4)"],
    },
    "Article 11": {
        "title": "Technical documentation",
        "obligation": (
            "Draw up technical documentation before the system is placed on the "
            "market and keep it up to date, demonstrating conformity with the "
            "Section 2 requirements. Minimum content is set out in Annex IV "
            "(Art. 11(1))."
        ),
        "key_paragraphs": ["11(1)"],
    },
    "Article 12": {
        "title": "Record-keeping",
        "obligation": (
            "Technically allow for the automatic recording of events (logs) over "
            "the lifetime of the system, appropriate to its intended purpose, to "
            "ensure a level of traceability of functioning (Art. 12(1)-(3))."
        ),
        "key_paragraphs": ["12(1)", "12(2)", "12(3)"],
    },
    "Article 13": {
        "title": "Transparency and provision of information to deployers",
        "obligation": (
            "Design and develop the system so its operation is sufficiently "
            "transparent to enable deployers to interpret output and use it "
            "appropriately; accompany it with instructions for use covering "
            "characteristics, capabilities and limitations of performance, "
            "including accuracy and known risks (Art. 13(1)-(3))."
        ),
        "key_paragraphs": ["13(1)", "13(3)(b)", "13(3)(d)"],
    },
    "Article 14": {
        "title": "Human oversight",
        "obligation": (
            "Design and develop the system so it can be effectively overseen by "
            "natural persons during use, including measures enabling the overseer "
            "to understand capabilities and limitations, remain aware of automation "
            "bias, correctly interpret output, and decide not to use or to override "
            "it (Art. 14(1)-(4))."
        ),
        "key_paragraphs": ["14(1)", "14(4)"],
    },
    "Article 15": {
        "title": "Accuracy, robustness and cybersecurity",
        "obligation": (
            "Design and develop the system to achieve an appropriate level of "
            "accuracy, robustness and cybersecurity, and to perform consistently in "
            "those respects throughout its lifecycle. Declared accuracy metrics go "
            "in the instructions for use; robustness includes resilience to errors, "
            "faults and inconsistencies, including feedback loops (Art. 15(1)-(5))."
        ),
        "key_paragraphs": ["15(1)", "15(3)", "15(4)"],
    },
    "Article 72": {
        "title": "Post-market monitoring",
        "obligation": (
            "Establish and document a post-market monitoring system proportionate "
            "to the risks, actively collecting and analysing data on performance "
            "over the system's lifetime, based on a post-market monitoring plan "
            "(Art. 72(1)-(3))."
        ),
        "key_paragraphs": ["72(1)", "72(3)"],
    },
}

# ---------------------------------------------------------------------------
# Annex III -- High-risk use-case areas referred to in Article 6(2).
# Verbatim excerpts of the areas most relevant to a clinical risk-scoring model.
# The full list has 8 areas; areas not relevant here are summarised by title.
# ---------------------------------------------------------------------------
ANNEX_III = {
    "reference": "Annex III, referred to in Article 6(2)",
    "areas": {
        "1": {"title": "Biometrics", "relevant": False,
              "text": "Remote biometric identification; biometric categorisation; emotion recognition."},
        "2": {"title": "Critical infrastructure", "relevant": False,
              "text": "AI used as safety components in critical digital infrastructure, road traffic, or supply of water, gas, heating or electricity."},
        "3": {"title": "Education and vocational training", "relevant": False,
              "text": "Admission/assignment, evaluation of learning outcomes, assessment of appropriate level of education, and exam-behaviour monitoring."},
        "4": {"title": "Employment, workers management and access to self-employment", "relevant": False,
              "text": "Recruitment/selection, and decisions on terms, promotion, termination, task allocation and performance monitoring."},
        "5": {
            "title": "Access to and enjoyment of essential private services and essential public services and benefits",
            "relevant": True,
            "points": {
                "5(a)": (
                    "AI systems intended to be used by public authorities or on behalf of public "
                    "authorities to evaluate the eligibility of natural persons for essential public "
                    "assistance benefits and services, including healthcare services, as well as to "
                    "grant, reduce, revoke, or reclaim such benefits and services."
                ),
                "5(b)": (
                    "AI systems intended to be used to evaluate the creditworthiness of natural persons "
                    "or establish their credit score, with the exception of AI systems used for the "
                    "purpose of detecting financial fraud."
                ),
                "5(c)": (
                    "AI systems intended to be used for risk assessment and pricing in relation to "
                    "natural persons in the case of life and health insurance."
                ),
                "5(d)": (
                    "AI systems intended to evaluate and classify emergency calls by natural persons or "
                    "to be used to dispatch, or to establish priority in the dispatching of, emergency "
                    "first response services, including by police, firefighters and medical aid, as well "
                    "as of emergency healthcare patient triage systems."
                ),
            },
        },
        "6": {"title": "Law enforcement", "relevant": False,
              "text": "Victim-risk assessment, polygraphs, evidence reliability, offending/re-offending risk, and profiling."},
        "7": {"title": "Migration, asylum and border control management", "relevant": False,
              "text": "Polygraphs; assessing security/irregular-migration/health risk of entrants; examining applications; detection/identification of persons."},
        "8": {"title": "Administration of justice and democratic processes", "relevant": False,
              "text": "Assisting a judicial authority in interpreting facts/law; influencing elections or referenda."},
    },
}

# Article 6(1): systems that are a safety component of a product, or are
# themselves a product, covered by the Union harmonisation legislation in
# Annex I (e.g. Regulation (EU) 2017/745 on medical devices) AND that require a
# third-party conformity assessment under that legislation, are high-risk.
ARTICLE_6_1 = {
    "reference": "Article 6(1)",
    "text": (
        "An AI system is high-risk where BOTH: (a) it is intended to be used as a "
        "safety component of a product, or is itself a product, covered by the Union "
        "harmonisation legislation listed in Annex I (which includes Regulation (EU) "
        "2017/745 on medical devices); AND (b) that product (or the AI system as a "
        "safety component) is required to undergo a third-party conformity assessment "
        "under that legislation."
    ),
}

# Article 6(3): derogation -- an Annex III system is NOT high-risk if it does not
# pose a significant risk of harm to health, safety or fundamental rights,
# including by not materially influencing the outcome of decision making. The
# derogation NEVER applies where the system performs profiling of natural persons.
ARTICLE_6_3 = {
    "reference": "Article 6(3)",
    "text": (
        "By derogation, an Annex III system is not high-risk where it does not pose a "
        "significant risk of harm to the health, safety or fundamental rights of "
        "natural persons, including by not materially influencing the outcome of "
        "decision-making. This applies where one or more conditions are met (narrow "
        "preparatory / procedural / confirmatory tasks). The derogation NEVER applies "
        "where the AI system performs profiling of natural persons."
    ),
}

# ---------------------------------------------------------------------------
# Annex IV -- Technical documentation content, referred to in Article 11(1).
# The 9 top-level sections, verbatim in substance. Used to structure the
# generated technical-documentation PDF.
# ---------------------------------------------------------------------------
ANNEX_IV = {
    "reference": "Annex IV, referred to in Article 11(1)",
    "sections": {
        "1": {
            "title": "General description of the AI system",
            "items": [
                "(a) intended purpose, provider name, and version and its relation to previous versions;",
                "(b) how the system interacts with hardware/software or other AI systems not part of it;",
                "(c) versions of relevant software/firmware and version-update requirements;",
                "(d) all forms in which the system is placed on the market or put into service (packages, downloads, APIs);",
                "(e) description of the hardware on which the system is intended to run;",
                "(f) where a component of products, photos/illustrations of external features, marking and internal layout;",
                "(g) a basic description of the user interface provided to the deployer;",
                "(h) instructions for use for the deployer.",
            ],
        },
        "2": {
            "title": "Detailed description of the elements of the AI system and of its development process",
            "items": [
                "(a) methods and steps performed for development, including any pre-trained/third-party tools;",
                "(b) design specifications: general logic, key design choices and rationale, what the system optimises for, "
                "expected output and output quality, and trade-offs made to comply with Chapter III Section 2;",
                "(c) system architecture and the computational resources used to develop, train, test and validate;",
                "(d) data requirements: datasheets on training methodologies, provenance, scope and characteristics of "
                "data sets, how data was obtained/selected, labelling procedures, and data-cleaning methodologies;",
                "(e) assessment of the human-oversight measures needed under Article 14, incl. technical measures under Art. 13(3)(d);",
                "(f) where applicable, pre-determined changes to the system and its performance and technical solutions for continuous compliance;",
                "(g) validation and testing procedures, incl. validation/testing data, metrics for accuracy, robustness and "
                "potentially discriminatory impacts, and dated/signed test logs and reports;",
                "(h) cybersecurity measures put in place.",
            ],
        },
        "3": {
            "title": "Monitoring, functioning and control of the AI system",
            "items": [
                "Capabilities and limitations in performance, including degrees of accuracy for specific persons/groups and "
                "overall expected accuracy; foreseeable unintended outcomes and sources of risk to health, safety, "
                "fundamental rights and discrimination; human-oversight measures under Article 14; and input-data specifications.",
            ],
        },
        "4": {
            "title": "Appropriateness of the performance metrics",
            "items": ["A description of the appropriateness of the chosen performance metrics for this specific AI system."],
        },
        "5": {
            "title": "Risk management system (Article 9)",
            "items": ["A detailed description of the risk-management system in accordance with Article 9."],
        },
        "6": {
            "title": "Lifecycle changes",
            "items": ["A description of relevant changes made by the provider to the system through its lifecycle."],
        },
        "7": {
            "title": "Harmonised standards applied",
            "items": [
                "A list of harmonised standards applied (fully/partly); where none apply, a detailed description of the "
                "solutions adopted to meet the Chapter III Section 2 requirements, and any other standards/specs applied.",
            ],
        },
        "8": {
            "title": "EU declaration of conformity",
            "items": ["A copy of the EU declaration of conformity referred to in Article 47."],
        },
        "9": {
            "title": "Post-market monitoring (Article 72)",
            "items": ["A detailed description of the post-market performance-evaluation system, including the Art. 72(3) plan."],
        },
    },
}


def article(name: str) -> dict:
    """Return the citation record for e.g. 'Article 15'."""
    return ARTICLES[name]


def annex_iv_sections() -> dict:
    return ANNEX_IV["sections"]
