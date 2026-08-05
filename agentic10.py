import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


from synthetic5 import (
#from synthetic6 import (
    sector_config,
    generate_businesses,
    generate_monthly_data,
    generate_gst_data,
    generate_upi_data,
    generate_aa_data,
    generate_epfo_data,
    engineer_features,
    compute_true_health_score,
)

from llm2 import ask_llm

MODEL_PATH = "Health_score2.pkl"
MODEL_COLUMNS_PATH = "Health_model2.json"

# ------------------------------------------------------------
# Agents the PlannerAgent is allowed to choose between when it
# still needs more evidence. Each maps 1:1 onto a data-collection
# Agent class below, which in turn pulls from synthetic5.py.
# ------------------------------------------------------------
AVAILABLE_AGENTS = [
    {"name": "GSTAgent", "description": "Collects and analyses GST filing data (revenue, taxable_turnover, gst_amount)."},
    {"name": "UPIAgent", "description": "Collects and analyses UPI transaction data (upi_value, upi_count) — a proxy for digital footprint."},
    {"name": "EPFOAgent", "description": "Collects EPFO employee/payroll data (employees, salary_paid, epfo_contribution)."},
    {"name": "AAAgent", "description": "Retrieves Account Aggregator bank data (bank_credits, bank_debits, bank_balance, cash_value)."},
]

SYSTEM_PROMPT = f"""
    You are the PlannerAgent of an MSME credit assessment system.
    Your job is NOT to solve tasks yourself.
    You must choose ONE agent from the list of remaining agents given to you.

    Full Agent Roster (for context):
    {AVAILABLE_AGENTS}

    Return ONLY the agent name, exactly as it appears in the remaining agents list.

    Example:
    GSTAgent
"""

@dataclass
class AssessmentContext:

    # business verification
    business_verified: bool = False
    verification_errors: List[str] = field(default_factory=list)

    gst_data: Optional[pd.DataFrame] = None
    upi_data: Optional[pd.DataFrame] = None
    aa_data: Optional[pd.DataFrame] = None
    epfo_data: Optional[pd.DataFrame] = None

    # prediction confidence
    confidence_reasons: List[str] = field(default_factory=list)

    #data agent
    business: Dict[str, Any] = field(default_factory = dict)
    monthly_df: Optional[pd.DataFrame] = None

    #feature agent
    features: Optional[pd.Series] = None

    #prediction agent
    health_score: Optional[float] = None
    confidence: float = 0.0

    #risk agent
    risk_band: Optional[str] = None

    #recommendation agent
    recommendations: List[Dict[str, Any]] = field(default_factory = list)
    primary_recommendation: Optional[str] = None

    #explanation agent
    explanation: Optional[Dict[str, Any]] = None

    #llm reasoning agent
    llm_summary: Optional[str] = None
    llm_summary_status: Optional[str] = None
    llm_prompt: Optional[str] = None

    available_sources: set[str] = field(default_factory = set)
    missing_sources: List[str] = field(default_factory = list)
    next_action: Optional[str] = None

    assumptions: List[str] = field(default_factory = list)
    current_agent: Optional[str] = None
    active_goals: List[str] = field(default_factory=list)
    goal_history: List[Dict[str, Any]] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    agent_status: Dict[str, str] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory = list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    
    completed_agents: set[str] = field(default_factory = set)
    log: List[str] = field(default_factory = list)

    def start_agent(self, agent: str, goal: str):
        self.current_agent = agent
        self.active_goals.append(goal)
        self.agent_status[agent] = "running"
        self.goal_history.append({
            "agent": agent,
            "goal": goal,
            "status": "running"
        })
        self.note(agent, f"Started: {goal}")

    def finish_agent(self, agent: str, goal: str, message: str, status: str = "completed"):
        self.completed_agents.add(agent)
        self.agent_status[agent] = status
        self.goal_history.append({"agent": agent, "goal": goal, "status": status, "message": message})
        if self.active_goals and self.active_goals[-1] == goal:
            self.active_goals.pop()

        self.current_agent = None

        self.note(agent, message)

    def note(self, agent_name: str, message: str) -> None:
        self.log.append(f"[{agent_name}] {message}")



class Agent:
    name = "Agent"
    goal = ""
    requires = ()
    produces = ()
    priority = 0

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        raise NotImplementedError

    def can_run(self, ctx):
        for req in self.requires:
            value = getattr(ctx, req, None)

            if value is None:
                return False

            if isinstance(value, bool):
                if not value:
                    return False
            elif isinstance(value, (list, dict, set, tuple)):
                if len(value) == 0:
                    return False

        return True
    

class EvidenceAgent(Agent):
    name = "EvidenceAgent"
    goal = "Identify available and missing financial data sources."
    requires = ("business_verified",)
    produces = ("available_sources", "missing_sources",)
    priority = 30

    def run(self, ctx: AssessmentContext) -> AssessmentContext:

        ctx.start_agent(self.name, self.goal)

        ctx.available_sources.clear()
        ctx.missing_sources.clear()

        sources = {

            "GST": ctx.gst_data,
            "UPI": ctx.upi_data,
            "AA": ctx.aa_data,
            "EPFO": ctx.epfo_data

        }

        for name, data in sources.items():
            if data is not None:
                ctx.available_sources.add(name)
            else:
                ctx.missing_sources.append(name)

        if ctx.missing_sources:
            ctx.observations.append({
                "agent": self.name,
                "type": "missing_sources",
                "message": f"Missing evidence sources: {ctx.missing_sources}"
            })

        ctx.finish_agent(
            self.name, 
            self.goal,
            f"Available: {sorted(ctx.available_sources)} | Missing: {ctx.missing_sources}"
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Evidence Collection",
            "reasoning": f"{len(ctx.available_sources)} of 4 sources available",
            "result": "Completed",
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "available_sources": sorted(ctx.available_sources),
                "missing_sources": list(ctx.missing_sources)
            }
        }

        return ctx

class BusinessVerificationAgent(Agent):

    name = "BusinessVerificationAgent"
    goal = "Verify that the applicant is a valid MSME business."
    requires = ("business",)
    produces = ("business_verified", "verification_errors",)
    priority = 20

    REQUIRED_FIELDS = [

        "business_id",
        "sector",
        "subcategory",
        
        "age_months",

        "gst_registered",
        "pan_available"

    ]

    def run(self, ctx):

        ctx.start_agent(self.name, self.goal)
        business = ctx.business
        errors = []

        # -------------------------
        # Required fields
        # -------------------------

        for field in self.REQUIRED_FIELDS:
            value = business.get(field)
            if value is None:
                errors.append(f"Missing {field}")

            if errors:
                ctx.verification_errors = errors
                ctx.business_verified = False

                for error in errors:
                    ctx.observations.append({
                        "agent": self.name,
                        "type": "verification_error",
                        "message": error
                    })

                ctx.finish_agent(
                    self.name,
                    self.goal,
                    f"Verification failed: {errors}",
                    status = "failed"
                )

                ctx.decision_history.append({
                    "agent": self.name,
                    "goal": self.goal,
                    "decision": "Verification",
                    "reasoning": errors,
                    "result": "Failed",
                    "timestamp": datetime.now()
                })

                ctx.agent_outputs[self.name] = {
                    "status": ctx.agent_status[self.name],
                    "outputs": {
                        "verified": False,
                        "errors": errors
                    }
                }

                return ctx

        # -------------------------
        # Sector validation
        # -------------------------

        sector = business.get("sector")
        if sector is not None:
            if business["sector"] not in sector_config:
                errors.append("Invalid sector")

            else:
                subcats = sector_config[
                    business["sector"]
                ]["subcategories"]

                if business.get("subcategory") not in subcats:
                    errors.append("Invalid subcategory")

        # -------------------------
        # GST
        # -------------------------

        if not business.get("gst_registered", False):
            errors.append(
                "Business is not GST registered."
            )

        # -------------------------
        # PAN
        # -------------------------

        if not business.get("pan_available", False):
            errors.append(
                "PAN not available."
            )

        if business.get("udyam_registered", False):
            ctx.observations.append({
                "agent": self.name,
                "type": "verification",
                "message": "Business is Udyam registered. MSME registration can be used as an additional credibility indicator."
            })
        else:
            ctx.observations.append({
                "agent": self.name,
                "type": "advisory",
                "message": "Business is not Udyam registered. Verification can continue, but some goverment support indicators may be unavailable."
            })

        # -------------------------
        # Final decision
        # -------------------------

        ctx.verification_errors = errors
        ctx.business_verified = len(errors) == 0

        if errors:
            for error in errors:
                ctx.observations.append({
                    "agent": self.name,
                    "type": "verification_error",
                    "message": error
                })
        else:
            ctx.observations.append({
                "agent": self.name,
                "type": "verification",
                "message": "Business passed verification."
            })

        if ctx.business_verified:
            ctx.finish_agent(
                self.name,
                self.goal,
                "Business profile successfully verified."
            )
            ctx.decision_history.append({
                "agent": self.name,
                "goal": self.goal,
                "decision": "Verification",
                "reasoning": "Mandatory business verification checks passed",
                "result": "Passed",
                "timestamp": datetime.now()
            })
        else:
            ctx.finish_agent(
                self.name,
                self.goal,
                f"Verification failed: {errors}",
                status = "failed"
            )
            ctx.decision_history.append({
                "agent": self.name,
                "goal": self.goal,
                "decision": "Verification",
                "reasoning": errors,
                "result": "Failed",
                "timestamp": datetime.now()
            })
            
        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "verified": ctx.business_verified,
                "errors": errors,
                "udyam_registered": business.get("udyam_registered", False)
            }
        }
        return ctx

class GSTAgent(Agent):
    name = "GSTAgent"
    goal = "Collect and prepare GST transaction data"
    requires = ("business_verified", )
    produces = ("gst_data", )
    priority = 40

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        gst_df = generate_gst_data(ctx.business)
        ctx.gst_data = gst_df
        ctx.observations.append({
            "agent": self.name,
            "type": "data_collection",
            "message": f"Collected {len(gst_df)} monthly GST records."
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            f"GST data collected ({len(gst_df)} records)."
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "GST Data Collection",
            "reasoning": "GST data successfully generated.",
            "result": "Completed",
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "records": len(gst_df),
                "columns": list(gst_df.columns)
            }
        }

        return ctx

class UPIAgent(Agent):
    name = "UPIAgent"
    goal = "Collect and prepare UPI transaction data."
    requires = ("business_verified",)
    produces = ("upi_data",)
    priority = 50

    def run(self, ctx: AssessmentContext) -> AssessmentContext:

        ctx.start_agent(self.name, self.goal)

        upi_df = generate_upi_data(ctx.business)

        ctx.upi_data = upi_df

        ctx.observations.append({
            "agent": self.name,
            "type": "data_collection",
            "message": f"Collected {len(upi_df)} UPI transaction records."
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            f"UPI data collected ({len(upi_df)} records)."
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "UPI Data Collection",
            "reasoning": "UPI data successfully generated.",
            "result": "Completed",
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "records": len(upi_df),
                "columns": list(upi_df.columns)
            }
        }

        return ctx

class EPFOAgent(Agent):

    name = "EPFOAgent"
    goal = "Collect and prepare EPFO employee contribution data."
    requires = ("business_verified",)
    produces = ("epfo_data",)
    priority = 60

    def run(self, ctx: AssessmentContext) -> AssessmentContext:

        ctx.start_agent(self.name, self.goal)

        epfo_df = generate_epfo_data(ctx.business)

        ctx.epfo_data = epfo_df

        ctx.observations.append({
            "agent": self.name,
            "type": "data_collection",
            "message": f"Collected {len(epfo_df)} EPFO records."
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            f"EPFO data collected ({len(epfo_df)} records)."
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "EPFO Data Collection",
            "reasoning": "EPFO data successfully generated.",
            "result": "Completed",
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "records": len(epfo_df),
                "columns": list(epfo_df.columns)
            }
        }

        return ctx

class AAAgent(Agent):

    name = "AAAgent"
    goal = "Collect and prepare Account Aggregator financial data."
    requires = ("business_verified",)
    produces = ("aa_data",)
    priority = 70

    def run(self, ctx: AssessmentContext) -> AssessmentContext:

        ctx.start_agent(self.name, self.goal)

        aa_df = generate_aa_data(ctx.business)

        ctx.aa_data = aa_df

        ctx.observations.append({
            "agent": self.name,
            "type": "data_collection",
            "message": f"Collected {len(aa_df)} AA records."
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            f"AA data collected ({len(aa_df)} records)."
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "AA Data Collection",
            "reasoning": "Account Aggregator data successfully generated.",
            "result": "Completed",
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "records": len(aa_df),
                "columns": list(aa_df.columns)
            }
        }

        return ctx
    
class FeatureEngineeringAgent(Agent):
    name = "FeatureEngineeringAgent"
    goal = "Generate model-ready features from all available evidence."
    requires = ("business",)
    produces = ("features",)
    priority = 80

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        source_frames = [
            df for df in (ctx.gst_data, ctx.upi_data, ctx.aa_data, ctx.epfo_data)
            if df is not None
        ]

        monthly_df = source_frames[0]
        for df in source_frames[1:]:
            monthly_df = monthly_df.merge(df, on="month", how="outer")
        monthly_df = monthly_df.sort_values("month").reset_index(drop=True)
        ctx.monthly_df = monthly_df

        features = engineer_features(monthly_df, ctx.business)
        features["is_NTB"] = int(not ctx.business.get("has_relationship_with_our_bank", False))
        features["is_NTC"] = int( ctx.business.get("credit_history") == "None")
        ctx.features = features

        used_sources = []

        if ctx.gst_data is not None:
            used_sources.append("GST")
        if ctx.upi_data is not None:
            used_sources.append("UPI")
        if ctx.aa_data is not None:
            used_sources.append("AA")
        if ctx.epfo_data is not None:
            used_sources.append("EPFO")

        if not used_sources:
            used_sources.append("business profile only")

        missing = ctx.missing_sources if ctx.missing_sources else ["None"]

        ctx.observations.append({
            "agent": self.name,
            "type": "feature_engineering",
            "message": (
                f"{len(features)} features engineered "
                f"from {', '.join(used_sources)}. "
                f"Missing: {', '.join(missing)}."
            )
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            "Engineered lending features."
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Feature Engineering",
            "reasoning": (
                    f"Generated features using {', '.join(used_sources)}. "
                    f"Missing sources: {', '.join(missing)}."
            ),
            "result": "Completed",
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "used_sources": used_sources,
                "feature_count": len(features),
                "feature_names": list(features.index),
            }
        }
        return ctx

class PredictionAgent(Agent):
    name = "PredictionAgent"
    goal = "Predict the financial health score from engineered features."
    requires = ("features", )
    produces = ("health_score", "risk_band", "confidence")
    priority = 90

    RISK_BANDS = [
        (300, 499, "High Risk"),
        (500, 649, "Moderate Risk"),
        (650, 799, "Low Risk"),
        (800, 900, "Excellent")
    ]

    def __init__(self, model, model_columns):
        self.model = model
        self.model_columns = model_columns

    def _band_for(self, score: float) -> str:
        for lo, hi, label in self.RISK_BANDS:
            if lo <= score <= hi:
                return label
        return "Unclassified"

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        features = ctx.features.copy()

        X_new = (
            pd.DataFrame([features])
            .reindex(columns=self.model_columns, fill_value=0)
            .astype(float)
        )
        score = float(self.model.predict(X_new)[0])
        score = float(np.clip(score, 300, 900))

        ctx.health_score = round(score)
        ctx.risk_band = self._band_for(score)
        model_name = type(self.model).__name__

        confidence = 1.0
        if ctx.missing_sources:
            confidence -= 0.1 * len(ctx.missing_sources)

        confidence = max(0.5, confidence)
        ctx.confidence = confidence
        ctx.confidence_reasons.clear()

        if ctx.missing_sources:
            ctx.confidence_reasons.append(
                f"Missing evidence: {', '.join(ctx.missing_sources)}"
            )
        else:
            ctx.confidence_reasons.append(
                "All available evidence used."
            )

        ctx.confidence_reasons.append(
            f"{len(features)} engineered features available for prediction."
        )
        ctx.confidence_reasons.append(
            f"Overall confidence: {confidence:.2f}"
        )

        ctx.observations.append({
            "agent": self.name,
            "type": "prediction",
            "score": ctx.health_score,
            "risk_band": ctx.risk_band,
            "confidence": confidence,
            "message": (
                f"Predicted health score {ctx.health_score} "
                f"({ctx.risk_band}) with confidence {confidence:.2f}."
            ),
            "model": model_name
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            f"Predicted score {ctx.health_score} ({ctx.risk_band})."
        )

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Prediction",
            "model": model_name,
            "reasoning": (
                f"Prediction made using {len(features)} engineered features. "
                f"Confidence: {confidence:.2f}. "
                f"Missing sources: {', '.join(ctx.missing_sources) if ctx.missing_sources else 'None'}."
            ),
            "result":  {
                "health_score": ctx.health_score,
                "risk_band": ctx.risk_band
            },
            "timestamp": datetime.now()
        })

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "model": model_name,
                "health_score": ctx.health_score,
                "risk_band": ctx.risk_band,
                "confidence": confidence,
                "confidence_reasons": list(ctx.confidence_reasons),
                "feature_count": len(features)
            }
        }

        return ctx
    
class RecommendationAgent(Agent):
    name = "RecommendationAgent"
    goal = "Generate lending recommendations based on the predicted financial health."
    requires = ("health_score", "risk_band", "features")
    produces = ("recommendations",)
    priority = 100

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        f = ctx.features if ctx.features is not None else {}
        recs: List[Dict[str, Any]] = []
        triggered_rules = []
        triggered_features = []
        timestamp = datetime.now()

        RECOMMENDATION_THRESHOLDS = {
                "revenue_cv": 0.50,
                "revenue_growth": 0.0,
                "bank_balance_cv": 0.60,
                "upi_penetration": 0.15,
                "gst_rate_max": 0.25,
                "high_confidence": 0.90,
                "medium_confidence": 0.75,
        }


        if ctx.risk_band == "Excellent":
            triggered_rules.append("Excellent_Risk")
            recs.append(
                {
                    "category": "Credit",
                    "code": "FAST_TRACK_LIMIT",
                    "priority": "Low",
                    "reason": "Low risk business with strong financial profile."
                }
            )
        elif ctx.risk_band == "Low Risk":
            triggered_rules.append("Low_Risk")
            recs.append(
                {
                    "category": "Credit",
                    "code": "STANDARD_WORKING_CAPITAL",
                    "priority": "Medium",
                    "reason": "Moderate stability with manageable risk."
                }
            )
        elif ctx.risk_band == "Moderate Risk":
            triggered_rules.append("Moderate_Risk")
            recs.append(
                {
                    "category": "Credit",
                    "code": "SMALL_SHORT_TERM_LOAN",
                    "priority": "High",
                    "reason": "Business should build repayment history before larger exposure."
                }
            )
        else:
            triggered_rules.append("High_Risk")
            recs.append(
                {
                    "category": "Credit",
                    "code": "SECURED_LOAN_ONLY",
                    "priority": "Critical",
                    "reason": "High risk profile."
                }
            )

        if f.get("revenue_cv", 0) > RECOMMENDATION_THRESHOLDS["revenue_cv"]:
            triggered_rules.append("High_Revenue_Variation")
            triggered_features.append("revenue_cv")
            recs.append({
                "category": "Cash Flow",
                "code": "FLEXIBLE_REPAYMENT",
                "priority": "High",
                "reason": "Revenue fluctuates significantly."
            })
        if f.get("revenue_growth", 0) < RECOMMENDATION_THRESHOLDS["revenue_growth"]:
            triggered_rules.append("Negative_Revenue_Growth")
            triggered_features.append("revenue_growth")
            recs.append({
                "category": "Monitoring",
                "code": "RM_REVIEW",
                "priority": "High",
                "reason": "Revenue trend is declining."
            })
        if f.get("upi_penetration", 1.0) < RECOMMENDATION_THRESHOLDS["upi_penetration"]:
            triggered_rules.append("Low_UPI_Penetration")
            triggered_features.append("upi_penetration")
            recs.append({
                "category": "Digital",
                "code": "UPI_ONBOARDING",
                "priority": "Medium",
                "reason": "Digital payment adoption is low."
            })
        if f.get("bank_balance_cv", 0) > RECOMMENDATION_THRESHOLDS["bank_balance_cv"]:
            triggered_rules.append("Unstable_Bank_Balance")
            triggered_features.append("bank_balance_cv")
            recs.append({
                "category": "Cash Flow",
                "code": "OVERDRAFT_FACILITY",
                "priority": "Medium",
                "reason": "Bank balance varies significantly."
            })
        if f.get("is_NTB", 0):
            triggered_rules.append("New_To_Bank")
            triggered_features.append("is_NTB")
            recs.append({
                "category": "Documentation",
                "code": "COMPLETE_KYC_AA",
                "priority": "Medium",
                "reason": "New-to-Bank customer."
            })
        if f.get("is_NTC", 0):
            triggered_rules.append("New_To_Credit")
            triggered_features.append("is_NTC")
            recs.append({
                "category": "Credit",
                "code": "STARTER_CREDIT",
                "priority": "High",
                "reason": "No formal credit history."
            })
        gst_rate = f.get("gst_effective_rate", 0.18)
        if gst_rate == 0 or gst_rate > RECOMMENDATION_THRESHOLDS["gst_rate_max"]:
            triggered_rules.append("GST_Compliance_Check")
            triggered_features.append("gst_effective_rate")
            recs.append({
                "category": "Compliance",
                "code": "VERIFY_GST",
                "priority": "Critical",
                "reason": "GST values appear inconsistent."
            })

        seen = set()
        unique_recs = []

        for r in recs:
            if r["code"] not in seen:
                unique_recs.append(r)
                seen.add(r["code"])

        recs = unique_recs
        priority_order = {
            "Critical": 0,
            "High": 1,
            "Medium": 2,
            "Low": 3,
        }
        recs.sort(key=lambda r: priority_order.get(r["priority"], 99))
        ctx.recommendations = recs

        triggered_rules = list(dict.fromkeys(triggered_rules))
        ctx.triggered_rules = triggered_rules

        triggered_features = list(dict.fromkeys(triggered_features))
        ctx.triggered_features = triggered_features

        primary_recommendation = recs[0]["code"] if recs else None
        ctx.primary_recommendation = primary_recommendation
        primary_reason = recs[0]["reason"] if recs else None
        ctx.primary_recommendation_reason = primary_reason

        recommendation_summary = (
            f"{primary_recommendation}: {primary_reason}"
            if primary_recommendation else
            "No recommendations generated."
        )

        priorities = sorted(
            {r["priority"] for r in recs},
            key=lambda x: priority_order.get(x, 99)
        )
        ctx.recommendation_priorities = priorities

        ctx.observations.append({
            "agent": self.name,
            "type": "recommendation",
            "health_score": ctx.health_score,
            "risk_band": ctx.risk_band,
            "recommendation_count": len(recs),
            "primary_recommendation": primary_recommendation,
            "triggered_rules": triggered_rules,
            "confidence": ctx.confidence,
            "message": (
                f"{primary_recommendation} selected. "
                f"{len(triggered_rules)} rule(s) triggered "
                f"{len(recs)} recommendation(s)."
            )
        })

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Recommendation",
            "reasoning": (
                f"Generated recommendations using health score "
                f"{ctx.health_score} ({ctx.risk_band}). "
                f"Confidence: {ctx.confidence:.2f}. "
                f"Missing sources: {', '.join(ctx.missing_sources) if ctx.missing_sources else 'None'}. "
                f"Triggered rules: {', '.join(triggered_rules)}. "
                f"Triggered features: "
                f"{', '.join(triggered_features) if triggered_features else 'None'}."
            ),
            "result": {
                "recommendation_count": len(recs),
                "primary": primary_recommendation,
                "confidence": ctx.confidence,
                "risk_band": ctx.risk_band,
                "health_score": ctx.health_score,
                "summary": recommendation_summary
            },
            "timestamp": timestamp
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            f"Generated {len(recs)} recommendation(s)."
        )

        categories = sorted({r["category"] for r in recs})
        ctx.recommendation_categories = categories

        if ctx.confidence >= RECOMMENDATION_THRESHOLDS["high_confidence"]:
            recommendation_confidence = "High"
        elif ctx.confidence >= RECOMMENDATION_THRESHOLDS["medium_confidence"]:
            recommendation_confidence = "Medium"
        else:
            recommendation_confidence = "Low"

        
        ctx.triggered_feature_count = len(triggered_features)
        ctx.recommendation_summary = recommendation_summary
        ctx.recommendation_confidence = recommendation_confidence

        ctx.recommendation_metadata = {
            "primary": primary_recommendation,
            "rule_count": len(triggered_rules),
            "feature_count": len(triggered_features),
            "confidence_level": recommendation_confidence,
            "confidence_score": ctx.confidence
        }

        ctx.recommendation_reasoning = {
            "risk_band": ctx.risk_band,
            "health_score": ctx.health_score,
            "confidence": ctx.confidence,
            "triggered_rules": triggered_rules,
            "triggered_features": triggered_features,
            "missing_sources": ctx.missing_sources,
        }

        ctx.recommendation_timestamp = timestamp
        ctx.recommendation_status = "Generated"
        ctx.recommendation_count = len(recs)

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "recommendation_count": len(recs),
                "primary_recommendation": primary_recommendation,
                "recommendations": recs,
                "primary_recommendation_reason": primary_reason,
                "categories": categories,
                "triggered_rules": triggered_rules,
                "priorities": priorities,
                "confidence": {
                    "score": ctx.confidence,
                    "level": recommendation_confidence
                },
                "triggered_rule_count": len(triggered_rules),
                "summary": recommendation_summary,
                "recommendation_confidence": recommendation_confidence,
                "triggered_features": triggered_features,
                "triggered_feature_count": len(triggered_features)
            }
        }

        return ctx
    
class ExplanationAgent(Agent):
    name = "ExplanationAgent"
    goal = "Generate a transparent explanation for the predicted financial health score."
    requires = ("health_score", "risk_band", "features")
    produces = ("explanation",)
    priority = 110

    def __init__(self, model: xgb.XGBRegressor, model_columns: List[str], reference_stats: Optional[pd.DataFrame] = None):
        self.model = model
        self.model_columns = model_columns
        self.importances = pd.Series(
            model.feature_importances_, index=model_columns
        ).sort_values(ascending=False)
        self.reference_stats = reference_stats  # population mean/std, optional

    def _direction(self, feature: str, value: float) -> str:
        if self.reference_stats is None or feature not in self.reference_stats.index:
            return ""
        mean = self.reference_stats.loc[feature, "mean"]
        if pd.isna(mean):
            return ""
        if value > mean:
            return "above average"
        elif value < mean:
            return "below average"
        return "about average"

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        timestamp = datetime.now()
        if ctx.features is None:
            ctx.finish_agent(
                self.name,
                self.goal,
                "No features available for explanation.",
                status="FAILED"
            )
            return ctx

        top_features = self.importances.head(5)
        lines = [
            f"Predicted score {ctx.health_score:.0f}/900 "
            f"({ctx.risk_band}). Key drivers for this model overall, and "
            f"how this business compares:"
        ]

        features = ctx.features

        if isinstance(features, dict):
            feature_names = features.keys()
        else:
            feature_names = features.index

        for feat, importance in top_features.items():
            if feat not in feature_names:
                continue
            value = features[feat]
            direction = self._direction(feat, value)
            direction_str = f", {direction} for the portfolio" if direction else ""
            lines.append(
                f"  - {feat} (model weight {importance:.1%}): "
                f"value={value:.3f}{direction_str}"
            )
        ctx.explanation = "\n".join(lines)

        ctx.explanation_metadata = {
            "top_features": list(top_features.index),
            "feature_count": len(top_features),
            "risk_band": ctx.risk_band,
            "health_score": ctx.health_score,
            "confidence": ctx.confidence,
            "summary": lines[0],
        }

        ctx.observations.append({
            "agent": self.name,
            "type": "explanation",
            "health_score": ctx.health_score,
            "risk_band": ctx.risk_band,
            "confidence": ctx.confidence,
            "top_features": list(top_features.index),
            "message": (
                f"Generated explanation using "
                f"{len(top_features)} important model features."
            )
        })

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Explanation",
            "reasoning": (
                f"Generated explanation from top "
                f"{len(top_features)} model features."
            ),
            "result": {
                "risk_band": ctx.risk_band,
                "health_score": ctx.health_score,
                "confidence": ctx.confidence
            },
            "timestamp": timestamp
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            "Generated explanation successfully."
        )

        ctx.explanation_summary = lines[0]
        ctx.explanation_line_count = len(lines)
        ctx.explanation_feature_count = len(top_features)
        ctx.explanation_timestamp = timestamp
        ctx.explanation_status = "Generated"

        feature_directions = {}
        for feat in top_features.index:
            if feat not in feature_names:
                continue

            feature_directions[feat] = self._direction(feat, features[feat])

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "summary": lines[0],
                "explanation": ctx.explanation,
                "top_features": list(top_features.index),
                "feature_importances": {
                    feat: float(importance)
                    for feat, importance in top_features.items()
                },
                "feature_directions": feature_directions,
                "risk_band": ctx.risk_band,
                "health_score": ctx.health_score,
                "confidence": ctx.confidence,
                "line_count": len(lines)
            }
        }

        return ctx
    

class LLMReasoningAgent(Agent):
    """
    Uses a local LLM (via llm.ask_llm, backed by Ollama) to turn the
    pipeline's own structured evidence, features, prediction, explanation
    and recommendations into a professional, natural-language credit
    assessment narrative.

    This agent is deliberately kept "read-only": it is given nothing but
    facts the pipeline has already computed, and the LLM's system prompt
    (in llm.py) instructs it to never invent facts and to only reason over
    what it is given. If the LLM call fails for any reason (e.g. the local
    Ollama server isn't running), the pipeline degrades gracefully — the
    rest of the assessment (score, risk band, explanation, recommendations)
    is unaffected, and a clear fallback message is recorded instead.
    """

    name = "LLMReasoningAgent"
    goal = "Generate a professional natural-language credit narrative using an LLM, grounded strictly in the pipeline's own evidence."
    requires = ("health_score", "risk_band", "recommendations")
    produces = ("llm_summary",)
    priority = 120

    def _build_prompt(self, ctx: AssessmentContext) -> str:
        business = ctx.business
        lines = [
            "Write a concise, professional MSME credit assessment narrative "
            "(4-6 sentences) for a credit committee memo, using ONLY the "
            "facts below. Do not invent any numbers or facts that are not "
            "listed here.",
            "",
            "BUSINESS PROFILE",
            f"- Sector / Subcategory: {business.get('sector')} / {business.get('subcategory')}",
            f"- Size: {business.get('business_size')}   Age: {business.get('age_months')} months",
            f"- Credit history: {business.get('credit_history')}",
            f"- GST registered: {business.get('gst_registered')}   "
            f"PAN available: {business.get('pan_available')}   "
            f"Udyam registered: {business.get('udyam_registered')}",
            f"- Existing relationship with our bank: {business.get('has_relationship_our_bank')}   "
            f"With another bank: {business.get('has_relationship_other_bank')}",
            "",
            "EVIDENCE",
            f"- Data sources available: {sorted(ctx.available_sources) or 'None'}",
            f"- Data sources missing: {ctx.missing_sources or 'None'}",
            "",
            "MODEL OUTPUT",
            f"- Health score: {ctx.health_score} / 900",
            f"- Risk band: {ctx.risk_band}",
            f"- Prediction confidence: {ctx.confidence:.0%}",
        ]

        explanation_meta = getattr(ctx, "explanation_metadata", None)
        if explanation_meta and explanation_meta.get("top_features"):
            lines.append(f"- Key model drivers: {', '.join(explanation_meta['top_features'])}")

        lines.append("")
        lines.append("RECOMMENDATIONS")
        if ctx.recommendations:
            for r in ctx.recommendations:
                lines.append(
                    f"- [{r['priority']}] {r['code']} ({r['category']}): {r['reason']}"
                )
        else:
            lines.append("- None generated.")

        return "\n".join(lines)

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        timestamp = datetime.now()
        prompt = self._build_prompt(ctx)

        try:
            print("ask_llm object:", ask_llm)
            summary = ask_llm(prompt)
            status = "Generated"
            message = "LLM narrative generated successfully."
        except Exception as exc:
            summary = (
                "LLM narrative unavailable (local LLM could not be reached: "
                f"{exc}). The deterministic explanation and recommendations "
                "above remain valid and were not affected."
            )
            status = "Failed"
            message = f"LLM call failed: {exc}"

        ctx.llm_summary = summary
        ctx.llm_summary_status = status
        ctx.llm_prompt = prompt

        ctx.observations.append({
            "agent": self.name,
            "type": "llm_reasoning",
            "status": status,
            "message": message,
        })

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "LLM Narrative Generation",
            "reasoning": (
                "Sent the pipeline's structured evidence, prediction and "
                "recommendations to a local LLM to produce a natural-"
                "language credit assessment narrative."
            ),
            "result": status,
            "timestamp": timestamp
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            message,
            status="completed" if status == "Generated" else "failed"
        )

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "llm_summary": ctx.llm_summary,
                "llm_summary_status": status,
                "prompt_char_count": len(prompt),
            }
        }

        return ctx


class PlannerAgent(Agent):
    name = "PlannerAgent"
    goal = "Determine the next agent required to complete the assessment."
    requires = ()
    produces = ("next_action",)
    priority = 10
    MIN_CONFIDENCE = 0.80

    def _remaining_data_agents(self, ctx: AssessmentContext) -> List[Dict[str, str]]:
        """Agents from AVAILABLE_AGENTS that haven't produced their data yet."""
        collected = {
            "GSTAgent": ctx.gst_data is not None,
            "UPIAgent": ctx.upi_data is not None,
            "EPFOAgent": ctx.epfo_data is not None,
            "AAAgent": ctx.aa_data is not None,
        }
        return [a for a in AVAILABLE_AGENTS if not collected.get(a["name"], False)]

    def _choose_data_agent(self, ctx: AssessmentContext, remaining: List[Dict[str, str]]) -> str:
        """
        Ask the LLM which of the remaining data agents should run next,
        using the same step-by-step reasoning as the standalone planner draft.
        Falls back to the first remaining agent if the LLM is unreachable
        or returns something unusable.
        """
        if len(remaining) == 1:
            return remaining[0]["name"]

        business = ctx.business
        user_input = (
            f"Business ID: {business.get('business_id')} | "
            f"Sector: {business.get('sector')} / {business.get('subcategory')} | "
            f"Size: {business.get('business_size')} | Age: {business.get('age_months')} months"
        )

        evidence_collected = [
            d["agent"] for d in ctx.decision_history
            if d["agent"] in {a["name"] for a in AVAILABLE_AGENTS}
        ]

        prompt = f"""
            {SYSTEM_PROMPT}

            User Input: {user_input}
            Goal: Gather enough evidence to confidently assess whether this business should receive a MSME loan.
            Current Confidence: {ctx.confidence:.2f}
            Evidence Collected: {evidence_collected}
            Available Agents: {remaining}

            Think step by step and ask yourself:
            1. Do I already have enough evidence?
            2. If not, what is the biggest uncertainty?
            3. Which available agent can reduce that uncertainty the most?
            4. Has the agent already been used? If yes, then search for other unused available agents.

            With this, which agent should be executed next?
            Return ONLY the agent name.
        """

        remaining_names = [a["name"] for a in remaining]

        try:
            raw_choice = ask_llm(prompt).strip()
        except Exception as exc:
            ctx.note(self.name, f"LLM agent selection failed ({exc}); defaulting to {remaining_names[0]}.")
            return remaining_names[0]

        for name in remaining_names:
            if name.lower() in raw_choice.lower():
                return name

        ctx.note(self.name, f"LLM returned an unrecognised agent '{raw_choice}'; defaulting to {remaining_names[0]}.")
        return remaining_names[0]

    def run(self, ctx: AssessmentContext) -> AssessmentContext:
        ctx.start_agent(self.name, self.goal)
        timestamp = datetime.now()
        reason = ""

        remaining_data_agents = self._remaining_data_agents(ctx)

        if remaining_data_agents:
            chosen = self._choose_data_agent(ctx, remaining_data_agents)
            ctx.next_action = chosen
            reason = (
                f"Planner selected {chosen} to reduce the biggest remaining "
                f"evidence gap ({[a['name'] for a in remaining_data_agents]})."
            )

        elif ctx.features is None:
            ctx.next_action = "feature_engineering"
            reason = "Features have not been engineered."

        elif ctx.health_score is None:
            ctx.next_action = "prediction"
            reason = "Financial health score has not been predicted."

        elif ctx.confidence < self.MIN_CONFIDENCE:
            ctx.next_action = "collect_more"
            reason = "Prediction confidence is below the required threshold and no further agents remain."

        elif not ctx.recommendations:
            ctx.next_action = "recommend"
            reason = "Recommendations have not been generated."

        elif ctx.explanation is None:
            ctx.next_action = "explain"
            reason = "Explanation has not been generated."

        elif ctx.llm_summary is None:
            ctx.next_action = "llm_summary"
            reason = "LLM credit narrative has not been generated."

        else:
            ctx.next_action = None
            reason = "Assessment pipeline completed successfully."

        ctx.observations.append({
            "agent": self.name,
            "type": "planning",
            "next_action": ctx.next_action,
            "reason": reason,
            "completed_agents": list(ctx.completed_agents),
            "confidence": ctx.confidence,
            "health_score": ctx.health_score,
            "risk_band": ctx.risk_band,
        })

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Planning",
            "reasoning": reason,
            "result": {
                "next_action": ctx.next_action,
                "confidence": ctx.confidence,
                "health_score": ctx.health_score,
                "risk_band": ctx.risk_band,
            },
            "timestamp": timestamp
        })

        ctx.planner_reason = reason
        ctx.planner_timestamp = timestamp
        ctx.planner_status = "Completed"

        ctx.finish_agent(
            self.name,
            self.goal,
            f"Next action: {ctx.next_action}"
        )

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "next_action": ctx.next_action,
                "reason": reason,
                "completed_agents": list(ctx.completed_agents),
                "confidence": ctx.confidence,
                "health_score": ctx.health_score,
                "risk_band": ctx.risk_band,
                "timestamp": timestamp,
            }
        }

        return ctx
    
class MissingDataAgent(Agent):

    name = "MissingDataAgent"
    goal = "Identify missing evidence and determine what additional information is required."
    requires = ()
    produces = ("confidence_reasons", "assumptions", "next_action")
    priority = 90

    FEATURE_SOURCE_MAP = {
        "upi_penetration": "UPI",
        "bank_balance_cv": "AA",
    }

    def run(self, ctx: AssessmentContext) -> AssessmentContext:

        ctx.start_agent(self.name, self.goal)
        timestamp = datetime.now()

        missing_sources = []
        missing_features = []
        confidence_reasons = []
        assumptions = []

        features = ctx.features

        if features is None:
            missing_features.extend(self.FEATURE_SOURCE_MAP.keys())
        else:
            feature_names = (
                features.keys()
                if isinstance(features, dict)
                else features.index
            )

            for feature, source in self.FEATURE_SOURCE_MAP.items():
                if feature not in feature_names:
                    missing_features.append(feature)

        for feature in missing_features:
            source = self.FEATURE_SOURCE_MAP[feature]

            if source not in missing_sources:
                missing_sources.append(source)

        for source in ctx.missing_sources:
            if source not in missing_sources:
                missing_sources.append(source)

        if ctx.confidence < PlannerAgent.MIN_CONFIDENCE:
            confidence_reasons.append(
                f"Prediction confidence ({ctx.confidence:.2f}) is below the required threshold "
                f"({PlannerAgent.MIN_CONFIDENCE:.2f})."
            )

        if missing_sources:
            confidence_reasons.append(
                f"Missing evidence sources: {', '.join(missing_sources)}."
            )

        if missing_features:
            confidence_reasons.append(
                f"Unavailable engineered features: {', '.join(missing_features)}."
            )

        for source in missing_sources:
            assumptions.append(
                f"{source} data unavailable. Model prediction uses partial evidence."
            )

        ctx.confidence_reasons = confidence_reasons
        ctx.assumptions = assumptions
        ctx.next_action = (
            "collect_more"
            if missing_sources or ctx.confidence < PlannerAgent.MIN_CONFIDENCE
            else None
        )

        ctx.observations.append({
            "agent": self.name,
            "type": "missing_data",
            "missing_sources": missing_sources,
            "missing_features": missing_features,
            "confidence": ctx.confidence,
            "message": (
                f"Detected {len(missing_sources)} missing source(s) "
                f"and {len(missing_features)} missing feature(s)."
            )
        })

        ctx.decision_history.append({
            "agent": self.name,
            "goal": self.goal,
            "decision": "Missing Data Analysis",
            "reasoning": (
                "Confidence and evidence availability were evaluated "
                "to determine additional data requirements."
            ),
            "result": {
                "missing_sources": missing_sources,
                "missing_features": missing_features,
                "next_action": ctx.next_action,
                "confidence": ctx.confidence
            },
            "timestamp": timestamp
        })

        ctx.finish_agent(
            self.name,
            self.goal,
            (
                "Additional evidence required."
                if ctx.next_action == "collect_more"
                else "No additional evidence required."
            )
        )

        ctx.agent_outputs[self.name] = {
            "status": ctx.agent_status[self.name],
            "outputs": {
                "missing_sources": missing_sources,
                "missing_features": missing_features,
                "confidence_reasons": confidence_reasons,
                "assumptions": assumptions,
                "next_action": ctx.next_action,
                "confidence": ctx.confidence
            }
        }

        return ctx

class AssessmentPipeline:
    def __init__(
        self,
        model: xgb.XGBRegressor,
        model_columns: List[str],
        reference_stats: Optional[pd.DataFrame] = None,
    ):
        self.verifier = BusinessVerificationAgent()
        self.gst_agent = GSTAgent()
        self.upi_agent = UPIAgent()
        self.epfo_agent = EPFOAgent()
        self.aa_agent = AAAgent()
        self.evidence_agent = EvidenceAgent()
        self.feature_agent = FeatureEngineeringAgent()
        self.prediction_agent = PredictionAgent(model, model_columns)
        self.planner_agent = PlannerAgent()
        self.missing_data_agent = MissingDataAgent()
        self.recommendation_agent = RecommendationAgent()
        self.explanation_agent = ExplanationAgent(model, model_columns, reference_stats)
        self.llm_agent = LLMReasoningAgent()

        # Lets the planner loop dispatch straight to the agent it picked by name.
        self.data_agents = {
            "GSTAgent": self.gst_agent,
            "UPIAgent": self.upi_agent,
            "EPFOAgent": self.epfo_agent,
            "AAAgent": self.aa_agent,
        }

    def assess(self, business: Dict[str, Any]) -> AssessmentContext:

        ctx = AssessmentContext(business=business)
        ctx.note(
            "AssessmentPipeline",
            "Assessment pipeline started."
        )
        ctx.observations.append({
            "agent":"AssessmentPipeline",
            "type":"pipeline",
            "message":"Assessment pipeline started."
        })
        pipeline_timestamp = datetime.now()
        # STEP 1 : Business Verification

        if self.verifier.can_run(ctx):
            ctx = self.verifier.run(ctx)

        if not ctx.business_verified:
            ctx.note(
                "AssessmentPipeline",
                "Assessment terminated because business verification failed."
            )
            ctx.agent_outputs["AssessmentPipeline"] = {
                "status": "failed",
                "reason": "Business verification failed"
            }
            return ctx
        # STEP 2 onward: everything — data collection, feature engineering,
        # prediction, recommendations, explanation and the LLM narrative —
        # is driven by the PlannerAgent loop below. The planner decides,
        # one step at a time, which data agent (GST/UPI/EPFO/AA) closes the
        # biggest evidence gap, exactly like the standalone planner draft.

        while True:
            ctx = self.planner_agent.run(ctx)
            action = ctx.next_action

            if action is None:
                ctx.next_action = None
                break

            elif action == "collect_more":
                ctx = self.missing_data_agent.run(ctx)
                ctx.note(
                    "AssessmentPipeline",
                    "Assessment completed with limited confidence because additional evidence is unavailable."
                )
                ctx.next_action = None
                # Can't generate more synthetic evidence.
                # Exit after documenting missing information.
                break

            elif action == "recommend":
                if self.recommendation_agent.can_run(ctx):
                    ctx = self.recommendation_agent.run(ctx)
                else:
                    break

            elif action == "explain":
                if self.explanation_agent.can_run(ctx):
                    ctx = self.explanation_agent.run(ctx)
                else:
                    break

            elif action == "llm_summary":
                if self.llm_agent.can_run(ctx):
                    ctx = self.llm_agent.run(ctx)
                else:
                    break

            elif action == "prediction":
                if self.prediction_agent.can_run(ctx):
                    ctx = self.prediction_agent.run(ctx)
                else:
                    break

            elif action == "feature_engineering":
                if self.evidence_agent.can_run(ctx):
                    ctx = self.evidence_agent.run(ctx)
                if self.feature_agent.can_run(ctx):
                    ctx = self.feature_agent.run(ctx)
                else:
                    break
                if self.prediction_agent.can_run(ctx):
                    ctx = self.prediction_agent.run(ctx)

            elif action in self.data_agents:
                # The planner picked exactly one evidence source to fetch.
                ctx = self.data_agents[action].run(ctx)

                if self.evidence_agent.can_run(ctx):
                    ctx = self.evidence_agent.run(ctx)

                # If a prediction already exists, this is a "collect more"
                # refinement pass — refresh features + prediction with the
                # newly gathered evidence before the planner reassesses.
                if ctx.health_score is not None:
                    if self.feature_agent.can_run(ctx):
                        ctx = self.feature_agent.run(ctx)

                if self.prediction_agent.can_run(ctx):
                    ctx = self.prediction_agent.run(ctx)

            else:

                ctx.observations.append({
                    "agent":"AssessmentPipeline",
                    "type":"planner_error",
                    "message":f"Unknown planner action {action}"
                })
                ctx.note(
                    "AssessmentPipeline",
                    f"Unknown planner action: {action}"
                )
                break

        ctx.observations.append({
            "agent": "AssessmentPipeline",
            "type": "pipeline",
            "message": "Assessment pipeline completed.",
            "completed_agents": sorted(ctx.completed_agents),
            "health_score": ctx.health_score,
            "risk_band": ctx.risk_band,
            "confidence": ctx.confidence
        })

        ctx.decision_history.append({
            "agent": "AssessmentPipeline",
            "goal": "Complete financial health assessment",
            "decision": "Pipeline Execution",
            "reasoning": "Executed the complete assessment workflow.",
            "result": {
                "completed": True,
                "health_score": ctx.health_score,
                "risk_band": ctx.risk_band,
                "confidence": ctx.confidence
            },
            "timestamp": pipeline_timestamp
        })

        ctx.completed_agents.add("AssessmentPipeline")

        ctx.agent_outputs["AssessmentPipeline"] = {

            "status": "completed",
            "timestamp": pipeline_timestamp,
            "outputs": {
                "business_verified": ctx.business_verified,
                "health_score": ctx.health_score,
                "risk_band": ctx.risk_band,
                "confidence": ctx.confidence,
                "recommendation_count": len(ctx.recommendations),
                "completed_agents": sorted(ctx.completed_agents),
                "missing_sources": ctx.missing_sources,
                "next_action": ctx.next_action,
                "available_sources": sorted(ctx.available_sources),
                "verification_errors": ctx.verification_errors,
                "planner_reason": getattr(ctx, "planner_reason", None),
                "primary_recommendation": ctx.primary_recommendation,
                "observation_count": len(ctx.observations),
                "decision_count": len(ctx.decision_history),
                "log_entries": len(ctx.log),
            }
        }

        ctx.note(
            "AssessmentPipeline",
            "Assessment pipeline completed."
        )
        return ctx
        
    def print_report(self, ctx: AssessmentContext) -> None:
        print("\n" + "=" * 60)
        print(f"MSME FINANCIAL HEALTH REPORT — business_id={ctx.business['business_id']}")
        print("=" * 60)
        print(f"Sector: {ctx.business['sector']} / {ctx.business['subcategory']}")
        print(f"Size: {ctx.business['business_size']}   Age: {ctx.business['age_months']} months")
        print(f"\nHealth Score: {ctx.health_score:.0f} / 900")
        print(f"Risk Band: {ctx.risk_band}")
        print(f"Confidence: {ctx.confidence:.2%}")
        print("\nExplanation:")
        if ctx.explanation:
            print(ctx.explanation)
        else:
            print("No explanation generated.")
        print("\nRecommendations:")
        if ctx.recommendations:
            for i, r in enumerate(ctx.recommendations, 1):
                print(f"{i}. {r}")
        else:
            print("No recommendations.")
        print("\nAI Credit Officer Narrative (LLM):")
        if ctx.llm_summary:
            print(f"[{ctx.llm_summary_status}] {ctx.llm_summary}")
        else:
            print("No LLM narrative generated.")
        print("\nAgent trace:")
        for entry in ctx.log:
            print(f"  {entry}")
        print("=" * 60 + "\n")
        print("\nEvidence")
        print(f"Available : {sorted(ctx.available_sources)}")
        print(f"Missing   : {ctx.missing_sources}")
        print("\nPlanner")
        print(f"Next Action : {ctx.next_action}")
        print(f"Reason      : {getattr(ctx, 'planner_reason', 'N/A')}")
        print("\nCompleted Agents")
        for agent in sorted(ctx.completed_agents):
            print(f"  ✓ {agent}")
        print("\nVerification")
        print(f"Status    : {'PASSED' if ctx.business_verified else 'FAILED'}")

        if ctx.verification_errors:
            print("Errors:")
            for error in ctx.verification_errors:
                print(f"  • {error}")
        if ctx.primary_recommendation:
            print(f"\nPrimary Recommendation: {ctx.primary_recommendation}")
        print("\nStatistics")
        print(f"Observations : {len(ctx.observations)}")
        print(f"Decisions    : {len(ctx.decision_history)}")
        print(f"Log Entries  : {len(ctx.log)}")



def train_or_load_model(n_train: int = 4000, force_retrain: bool = False, seed: int = 24):
    
    if (not force_retrain and os.path.exists(MODEL_PATH)
            and os.path.exists(MODEL_COLUMNS_PATH)):
        print(f"Loading cached model from {MODEL_PATH} ...")
        model = joblib.load(MODEL_PATH)
        with open(MODEL_COLUMNS_PATH) as fh:
            model_columns = json.load(fh)
        return model, model_columns, None

    print(f"No cached model found — training a fresh model on {n_train} "
          f"synthetic businesses...")
    biz_df = generate_businesses(n=n_train, seed=seed)

    feature_rows, targets = [], []
    for idx, row in biz_df.iterrows():
        monthly = generate_monthly_data(row, end_date=datetime(2026, 7, 15))
        feats = engineer_features(monthly, row)
        score = compute_true_health_score(feats)
        if pd.isna(score) or np.isinf(score):
            continue
        feature_rows.append(feats)
        targets.append(score)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets, name="health_score")

    X_model = X.drop(columns=["business_id", "sector", "subcategory", "size",
                               "credit_history", "has_relationship_our_bank",
                               "has_relationship_other_bank"])
    X_model["is_NTB"] = (~X["has_relationship_our_bank"]).astype(int)
    X_model["is_NTC"] = (X["credit_history"] == "None").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Training complete. R²={r2_score(y_test, y_pred):.4f}  "
          f"MAE={mean_absolute_error(y_test, y_pred):.2f}")

    joblib.dump(model, MODEL_PATH)
    with open(MODEL_COLUMNS_PATH, "w") as fh:
        json.dump(list(X_model.columns), fh)

    reference_stats = X_model.agg(["mean", "std"]).T
    return model, list(X_model.columns), reference_stats


DEMO_BUSINESS = {
    "business_id": 999001,
    "sector": "Textiles and Apparel",
    "subcategory": "Garment manufacturing units",
    "business_size": "Small",

    "age_months": 30,

    "gst_registered": True,
    "pan_available": True,
    "udyam_registered": True,

    "credit_history": "Limited",

    "has_relationship_our_bank": False,
    "has_relationship_other_bank": True,
}


def main():
    parser = argparse.ArgumentParser(description="Agentic MSME health-score pipeline")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a JSON file describing the business to assess. "
                              "If omitted, a built-in demo business is used.")
    parser.add_argument("--retrain", action="store_true",
                         help="Force retraining the model instead of using the cache.")
    parser.add_argument("--train-size", type=int, default=4000,
                         help="Number of synthetic businesses to train on.")
    args = parser.parse_args()

    model, model_columns, reference_stats = train_or_load_model(
        n_train=args.train_size, force_retrain=args.retrain
    )

    pipeline = AssessmentPipeline(model, model_columns, reference_stats)

    if args.input:
        with open(args.input) as fh:
            business = json.load(fh)
    else:
        business = DEMO_BUSINESS
        print("\nNo --input file given — running the built-in demo business.")

    print(f"Assessing Business {business['business_id']}...")
    ctx = pipeline.assess(business)
    print("\nAssessment completed.\n")
    pipeline.print_report(ctx)


if __name__ == "__main__":
    main()