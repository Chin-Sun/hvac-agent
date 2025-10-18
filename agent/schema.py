"""
HVAC Booking Agent - Data Schema

Pydantic data models for HVAC booking information.

Author: Qian Sun
Date: 2025-10-17
Version: 1.0.0
License: MIT License

Copyright (c) 2025 Qian Sun. Licensed under the MIT License.
"""

from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional, Literal

Severity = Literal["critical", "high", "medium", "low"]


class BookingIntent(BaseModel):
    service_type: Literal[
        "ac_repair",
        "furnace_maintenance",
        "installation",
        "cleaning",
        "ventilation_maintenance",
        "other",
    ]
    equipment_brand: Optional[str] = None
    problem_summary: Optional[str] = None
    severity: Optional[Severity] = None
    property_type: Optional[
        Literal["apartment", "detached_house", "townhouse", "commercial", "other"]
    ] = None
    address: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postal_code: Optional[str] = None
    preferred_timeslots: List[str] = Field(default_factory=list)
    access_notes: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class AgentOutput(BaseModel):
    summary: str
    booking: BookingIntent
