#!/usr/bin/env python3
"""
HVAC Booking Agent CLI

A structured booking process CLI for HVAC services using OpenAI.

Author: Qian Sun
Date: 2025-10-17
Version: 1.0.0
License: MIT License

Usage:
    python cli.py [OPTIONS]

Options:
    --api-key TEXT     OpenAI API key
    --model TEXT       OpenAI model to use (default: gpt-4)
    --verbose          Enable verbose output

Examples:
    python cli.py
    python cli.py --model gpt-4-turbo
    python cli.py --api-key sk-xxx --verbose
    python cli.py --api-key sxxxxx --model gpt-4-turbo --verbose
"""

import json
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from agent.llm_client import create_llm_client
from agent.schema import AgentOutput
from agent.prompt import (
    get_llm_system_prompt,
    get_extraction_prompt,
    get_validation_prompt,
    get_followup_prompt,
)

# Initialize Rich console
console = Console()


@click.command()
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--model", default="gpt-4", help="OpenAI model to use")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(api_key: str, model: str, verbose: bool):
    """HVAC Booking Agent - Structured Booking Process"""

    # Check API key
    if not api_key:
        console.print("[red]Error: OpenAI API key is required[/red]")
        console.print("Set OPENAI_API_KEY environment variable or use --api-key option")
        sys.exit(1)

    # Start booking process
    start_booking_process(api_key, model, verbose)


def start_booking_process(api_key: str, model: str, verbose: bool):
    """Start prompt chain booking process"""

    # Display welcome message
    welcome_message = (
        "[bold blue]HVAC Booking Service System[/bold blue]\n"
        "I'll help you book HVAC services. Please tell me what you need and I'll guide you through the process."
    )
    console.print(Panel.fit(welcome_message, title="Welcome"))

    try:
        # Create LLM client
        client = create_llm_client(api_key=api_key, model=model)

        # Test connection
        with console.status("[bold green]Testing API connection..."):
            if not client.test_connection():
                console.print("[red]❌ API connection failed![/red]")
                return

        console.print("[green]✅ Connected to OpenAI API[/green]")

        # Start prompt chain process
        booking_data = run_prompt_chain(client)

        if not booking_data:
            console.print("[yellow]Booking cancelled[/yellow]")
            return

        # Confirm information
        if confirm_booking_information(booking_data):
            # Generate booking summary
            generate_booking_summary(client, booking_data)
        else:
            console.print("[yellow]Booking cancelled[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def run_prompt_chain(client) -> Optional[Dict[str, Any]]:
    """Run the prompt chain process to collect booking information"""

    console.print("\n[bold cyan]Let's start with your HVAC service request[/bold cyan]")
    console.print("=" * 50)

    # Step 1: Get initial user request
    user_request = Prompt.ask(
        "\n[bold blue]Please tell me what HVAC service you need[/bold blue]"
    )

    # Initialize conversation history
    conversation_history = [user_request]
    current_booking_data = {}

    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        console.print(f"\n[dim]Processing your request... (Step {iteration})[/dim]")

        try:
            # Step 2: Extract information from current conversation
            extracted_data = extract_booking_information(client, conversation_history)

            if not extracted_data:
                console.print("[red]Failed to extract booking information[/red]")
                return None

            # Update current booking data with extracted information
            current_booking_data.update(extracted_data)

            # Step 3: Validate against schema and check completeness
            validation_result = validate_booking_information(
                client, current_booking_data
            )

            if validation_result.get("is_complete", False):
                console.print("[green]✅ All required information collected![/green]")
                break

            # Step 4: Generate follow-up question
            followup_question = generate_followup_question(
                client, current_booking_data, validation_result
            )

            if not followup_question:
                console.print("[yellow]Unable to generate follow-up question[/yellow]")
                break

            # Ask the follow-up question
            console.print(f"\n[bold blue]Question:[/bold blue] {followup_question}")
            user_response = Prompt.ask("Your answer")

            if user_response.lower() in ["quit", "exit", "cancel"]:
                console.print("[yellow]Booking cancelled by user[/yellow]")
                return None

            # Add to conversation history
            conversation_history.append(f"Q: {followup_question}\nA: {user_response}")

        except Exception as e:
            console.print(f"[red]Error in prompt chain: {str(e)}[/red]")
            return None

    if iteration >= max_iterations:
        console.print(
            "[yellow]Maximum iterations reached. Using collected information.[/yellow]"
        )

    return current_booking_data


def extract_booking_information(
    client, conversation_history: List[str]
) -> Optional[Dict[str, Any]]:
    """Extract booking information from conversation using LLM"""

    try:
        # Build conversation content
        conversation_content = "\n".join(
            [f"Turn {i + 1}: {turn}" for i, turn in enumerate(conversation_history)]
        )

        # Process with LLM
        messages = [
            {"role": "system", "content": get_extraction_prompt()},
            {"role": "user", "content": conversation_content},
        ]

        response_content = client._chat_completion(messages, temperature=0.1)

        # Parse JSON response
        import json

        response_data = json.loads(response_content)

        # Extract booking data from response
        booking_data = response_data.get("booking", {})

        # Convert to simple dict format for easier processing
        result = {}
        for key, value in booking_data.items():
            if value is not None and value != "" and value != []:
                result[key] = value

        return result

    except Exception as e:
        console.print(f"[red]Error extracting information: {str(e)}[/red]")
        return None


def validate_booking_information(
    client, booking_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate booking information against schema requirements"""

    try:
        # Create a summary of current booking data
        booking_summary = f"Current booking information: {booking_data}"

        messages = [
            {"role": "system", "content": get_validation_prompt()},
            {"role": "user", "content": booking_summary},
        ]

        response_content = client._chat_completion(messages, temperature=0.1)

        # Parse JSON response
        import json

        validation_result = json.loads(response_content)

        return validation_result

    except Exception as e:
        console.print(f"[red]Error validating information: {str(e)}[/red]")
        return {"is_complete": False, "missing_fields": [], "questions": []}


def generate_followup_question(
    client, booking_data: Dict[str, Any], validation_result: Dict[str, Any]
) -> Optional[str]:
    """Generate a follow-up question based on missing information"""

    try:
        # Create context for follow-up question generation
        context = f"""
Current booking data: {booking_data}
Missing fields: {validation_result.get("missing_fields", [])}
Previous questions: {validation_result.get("questions", [])}
"""

        messages = [
            {"role": "system", "content": get_followup_prompt()},
            {"role": "user", "content": context},
        ]

        followup_question = client._chat_completion(messages, temperature=0.3)

        return followup_question.strip()

    except Exception as e:
        console.print(f"[red]Error generating follow-up question: {str(e)}[/red]")
        return None


def confirm_booking_information(booking_data: Dict) -> bool:
    """Confirm booking information"""

    console.print("\n[bold cyan]Please confirm your booking information[/bold cyan]")
    console.print("=" * 50)

    # Create information table
    table = Table(title="Booking Information Confirmation")
    table.add_column("Item", style="cyan")
    table.add_column("Information", style="green")

    # Add service type
    service_type_map = {
        "ac_repair": "AC Repair",
        "furnace_maintenance": "Furnace Maintenance",
        "installation": "Equipment Installation",
        "cleaning": "Cleaning Service",
        "ventilation_maintenance": "Ventilation System Maintenance",
        "other": "Other Service",
    }
    table.add_row(
        "Service Type",
        service_type_map.get(booking_data.get("service_type", ""), "Unknown"),
    )

    # Add problem description
    if booking_data.get("problem_description"):
        table.add_row(
            "Problem Description",
            (
                booking_data["problem_description"][:50] + "..."
                if len(booking_data["problem_description"]) > 50
                else booking_data["problem_description"]
            ),
        )

    # Add severity level
    severity_map = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    table.add_row(
        "Severity Level", severity_map.get(booking_data.get("severity", ""), "Unknown")
    )

    # Add property type
    property_type_map = {
        "apartment": "Apartment",
        "detached_house": "Detached House",
        "townhouse": "Townhouse",
        "commercial": "Commercial Building",
        "other": "Other",
    }
    table.add_row(
        "Property Type",
        property_type_map.get(booking_data.get("property_type", ""), "Unknown"),
    )

    # Add address information
    address_parts = []
    if booking_data.get("address"):
        address_parts.append(booking_data["address"])
    if booking_data.get("city"):
        address_parts.append(booking_data["city"])
    if booking_data.get("province"):
        address_parts.append(booking_data["province"])
    if booking_data.get("postal_code"):
        address_parts.append(booking_data["postal_code"])

    if address_parts:
        table.add_row("Address", ", ".join(address_parts))

    # Add contact information
    if booking_data.get("contact_name"):
        table.add_row("Contact Name", booking_data["contact_name"])
    if booking_data.get("contact_phone"):
        table.add_row("Phone", booking_data["contact_phone"])
    if booking_data.get("contact_email"):
        table.add_row("Email", booking_data["contact_email"])

    # Add time preference
    if booking_data.get("time_preference"):
        table.add_row("Time Preference", booking_data["time_preference"])

    # Add access notes
    if booking_data.get("access_notes"):
        table.add_row(
            "Access Notes",
            (
                booking_data["access_notes"][:50] + "..."
                if len(booking_data["access_notes"]) > 50
                else booking_data["access_notes"]
            ),
        )

    # Add special requirements
    if booking_data.get("constraints"):
        table.add_row(
            "Special Requirements",
            (
                booking_data["constraints"][:50] + "..."
                if len(booking_data["constraints"]) > 50
                else booking_data["constraints"]
            ),
        )

    console.print(table)

    # Confirm using interaction prompts
    return Confirm.ask("\nPlease confirm if the above information is correct?")


def generate_booking_summary(client, booking_data: Dict):
    """Generate booking summary"""

    console.print("\n[bold cyan]Generating booking summary[/bold cyan]")

    try:
        # Build conversation content
        conversation_turns = []

        # Add service type and problem description
        service_type_map = {
            "ac_repair": "AC Repair",
            "furnace_maintenance": "Furnace Maintenance",
            "installation": "Equipment Installation",
            "cleaning": "Cleaning Service",
            "ventilation_maintenance": "Ventilation System Maintenance",
            "other": "Other Service",
        }

        service_type = service_type_map.get(
            booking_data.get("service_type", ""), "Unknown Service"
        )
        problem_desc = booking_data.get("problem_description", "")

        conversation_turns.append(
            f"I need {service_type} service. Problem description: {problem_desc}"
        )

        # Add address information
        address_parts = []
        if booking_data.get("address"):
            address_parts.append(booking_data["address"])
        if booking_data.get("city"):
            address_parts.append(booking_data["city"])
        if booking_data.get("province"):
            address_parts.append(booking_data["province"])

        if address_parts:
            conversation_turns.append(f"Address: {', '.join(address_parts)}")

        # Add time preference
        if booking_data.get("time_preference"):
            conversation_turns.append(
                f"Preferred service time: {booking_data['time_preference']}"
            )

        # Add contact information
        if booking_data.get("contact_name"):
            conversation_turns.append(f"Contact person: {booking_data['contact_name']}")
        if booking_data.get("contact_phone"):
            conversation_turns.append(f"Contact phone: {booking_data['contact_phone']}")

        # Add other information
        if booking_data.get("access_notes"):
            conversation_turns.append(
                f"Access instructions: {booking_data['access_notes']}"
            )
        if booking_data.get("constraints"):
            conversation_turns.append(
                f"Special requirements: {booking_data['constraints']}"
            )

        # Process with LLM using structured prompt
        with console.status("[bold green]Processing booking information..."):
            result = client.process_conversation(
                conversation_turns=conversation_turns,
                system_prompt=get_llm_system_prompt(),
            )

        # Display booking summary
        display_booking_summary(result)

        # Save booking record
        save_booking_record(booking_data, result)

    except Exception as e:
        console.print(f"[red]Failed to generate summary: {str(e)}[/red]")


def display_booking_summary(result: AgentOutput):
    """Display booking summary"""

    console.print(
        "\n[bold green]✅ Booking summary generated successfully[/bold green]"
    )

    # Create summary panel
    summary_panel = Panel(result.summary, title="Booking Summary", border_style="green")
    console.print(summary_panel)

    # Create detailed information table
    table = Table(title="Detailed Booking Information")
    table.add_column("Item", style="cyan")
    table.add_column("Information", style="green")

    booking = result.booking

    table.add_row("Service Type", booking.service_type or "Not specified")
    table.add_row("Equipment Brand", booking.equipment_brand or "Not specified")
    table.add_row("Problem Summary", booking.problem_summary or "Not specified")
    table.add_row("Severity Level", booking.severity or "Not specified")
    table.add_row("Property Type", booking.property_type or "Not specified")
    table.add_row("Address", booking.address or "Not specified")
    table.add_row("City", booking.city or "Not specified")
    table.add_row("Province", booking.province or "Not specified")
    table.add_row(
        "Time Preference",
        (
            ", ".join(booking.preferred_timeslots)
            if booking.preferred_timeslots
            else "Not specified"
        ),
    )
    table.add_row("Contact Name", booking.contact_name or "Not specified")
    table.add_row("Phone", booking.contact_phone or "Not specified")
    table.add_row("Email", booking.contact_email or "Not specified")
    table.add_row(
        "Constraints", ", ".join(booking.constraints) if booking.constraints else "None"
    )
    table.add_row("Confidence", f"{booking.confidence:.2f}")

    console.print(table)

    console.print("\n[bold blue]Booking process completed![/bold blue]")
    console.print(
        "Our customer service team will contact you within 24 hours to confirm specific arrangements."
    )


def save_booking_record(booking_data: Dict, result: AgentOutput):
    """Save booking record"""

    try:
        # Build complete record using original booking_data and AI summary
        record = {
            "booking_data": booking_data,  # Collected data
            "ai_summary": result.summary,  # AI-generated summary text
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save to file
        filename = "booking_records.jsonl"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(f"[dim]Booking record saved to {filename}[/dim]")

    except Exception as e:
        console.print(f"[yellow]Failed to save record: {str(e)}[/yellow]")


if __name__ == "__main__":
    main()
