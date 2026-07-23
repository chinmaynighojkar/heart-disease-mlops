"""Step 3a -- Pydantic schemas for Annex IV technical documentation.

One model per Annex IV top-level section (1-9). These give the generated
technical documentation a typed, validated structure: if a required field is
missing the build fails loudly rather than emitting a document with holes.

The schemas describe *structure*; `doc_generator.py` fills them from the real
model metadata in models/training_metrics.json.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Sec1GeneralDescription(BaseModel):
    """Annex IV(1) -- General description of the AI system."""
    intended_purpose: str
    provider: str
    version: str
    relation_to_previous_versions: str
    interactions: str                      # (b)
    software_versions: str                 # (c)
    forms_placed_on_market: str            # (d)
    target_hardware: str                   # (e)
    user_interface: str                    # (g)
    instructions_for_use: str              # (h)


class Sec2DevelopmentProcess(BaseModel):
    """Annex IV(2) -- Elements of the system and its development process."""
    development_methods: str               # (a)
    design_specifications: str             # (b) general logic, key choices, optimisation target
    optimisation_target: str               # (b)
    system_architecture: str               # (c)
    computational_resources: str           # (c)
    data_requirements: str                 # (d) datasheet: provenance, scope, characteristics
    data_cleaning: str                     # (d)
    human_oversight_assessment: str        # (e)
    predetermined_changes: str             # (f)
    validation_testing_procedure: str      # (g)
    accuracy_metrics: dict[str, float]     # (g)
    subgroup_metrics: dict[str, dict]      # (g) potentially discriminatory impact
    cybersecurity_measures: str            # (h)


class Sec3MonitoringControl(BaseModel):
    """Annex IV(3) -- Monitoring, functioning and control."""
    capabilities_and_limitations: str
    per_group_accuracy: str
    overall_expected_accuracy: str
    foreseeable_unintended_outcomes: str
    human_oversight_measures: str
    input_data_specifications: str


class Sec4MetricAppropriateness(BaseModel):
    """Annex IV(4) -- Appropriateness of the performance metrics."""
    rationale: str


class Sec5RiskManagement(BaseModel):
    """Annex IV(5) -- Risk-management system per Article 9."""
    description: str
    identified_risks: list[str]
    mitigations: list[str]
    residual_risks: list[str]


class Sec6LifecycleChanges(BaseModel):
    """Annex IV(6) -- Relevant changes through the lifecycle."""
    changes: str


class Sec7Standards(BaseModel):
    """Annex IV(7) -- Harmonised standards applied (or solutions adopted)."""
    harmonised_standards_applied: list[str]
    solutions_where_no_standard: str


class Sec8DeclarationOfConformity(BaseModel):
    """Annex IV(8) -- Copy of the EU declaration of conformity (Art. 47)."""
    status: str                            # e.g. "Not applicable -- demonstrator"
    explanation: str


class Sec9PostMarketMonitoring(BaseModel):
    """Annex IV(9) -- Post-market monitoring system and plan (Art. 72)."""
    system_description: str
    monitoring_plan: str
    known_limitations: str


class TechnicalDocumentation(BaseModel):
    """Full Annex IV technical-documentation record."""
    document_title: str = "Technical Documentation (EU AI Act Annex IV)"
    regulation: str = "Regulation (EU) 2024/1689"
    generated_utc: str
    doc_version: str
    content_hash: str = Field(default="", description="Set after rendering, for the audit trail")

    sec1: Sec1GeneralDescription
    sec2: Sec2DevelopmentProcess
    sec3: Sec3MonitoringControl
    sec4: Sec4MetricAppropriateness
    sec5: Sec5RiskManagement
    sec6: Sec6LifecycleChanges
    sec7: Sec7Standards
    sec8: Sec8DeclarationOfConformity
    sec9: Sec9PostMarketMonitoring
