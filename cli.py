#!/usr/bin/env python3
"""
HVAC Booking Agent CLI

A structured booking process CLI for HVAC services using OpenAI.

Author: Qian Sun
Date: 2025-10-18
Version: 3.0.0
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
from agent.prompt import (
    get_guidance_prompt,
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
            # Save booking record directly
            save_booking_record_simple(booking_data)
        else:
            console.print("[yellow]Booking cancelled[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def run_prompt_chain(client) -> Optional[Dict[str, Any]]:
    """Run the prompt chain process to collect booking information"""

    console.print("\n[bold cyan]Let's start with your HVAC service request[/bold cyan]")
    console.print("=" * 50)

    # Initialize conversation state
    conversation_history = []
    current_booking_data = {}
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        console.print(f"\n[dim]Processing your request... (Step {iteration})[/dim]")

        try:
            # Step 1: Get conversation guidance strategy
            guidance_result = get_conversation_guidance(
                client, current_booking_data, conversation_history
            )

            if not guidance_result:
                console.print("[red]Failed to get conversation guidance[/red]")
                return None

            # Step 2: Use guidance to interact with user
            if guidance_result.get("recommended_strategy") == "A" and iteration == 1:
                # Initial greeting and first question
                console.print(
                    f"\n[bold green]{guidance_result.get('conversation_starter', 'Hi! How can I help you with HVAC services?')}[/bold green]"
                )
            else:
                # Follow-up questions
                console.print(
                    f"\n[bold blue]{guidance_result.get('conversation_starter', 'Could you provide more information?')}[/bold blue]"
                )

            # Get user response
            user_response = Prompt.ask("Your answer")

            if user_response.lower() in ["quit", "exit", "cancel"]:
                console.print("[yellow]Booking cancelled by user[/yellow]")
                return None

            # Add to conversation history
            conversation_history.append(user_response)

            # Step 3: Extract information from current conversation
            extracted_data = extract_booking_information(client, conversation_history)

            if not extracted_data:
                console.print("[red]Failed to extract booking information[/red]")
                return None

            # Update current booking data with extracted information
            current_booking_data.update(extracted_data)

            # Step 4: Check if we have enough information
            if guidance_result.get("recommended_strategy") == "D":
                console.print("[green]✅ All required information collected![/green]")
                break

        except Exception as e:
            console.print(f"[red]Error in prompt chain: {str(e)}[/red]")
            return None

    if iteration >= max_iterations:
        console.print(
            "[yellow]Maximum iterations reached. Using collected information.[/yellow]"
        )

    return current_booking_data


def get_conversation_guidance(
    client, current_booking_data: Dict[str, Any], conversation_history: List[str]
) -> Optional[Dict[str, Any]]:
    """Get conversation guidance strategy using the new guidance prompt"""

    try:
        # Build context for guidance analysis
        context = f"""
Current extracted information: {current_booking_data}
Missing critical information: {get_missing_critical_info(current_booking_data)}
Conversation stage: {len(conversation_history)} turns
"""

        messages = [
            {"role": "system", "content": get_guidance_prompt()},
            {"role": "user", "content": context},
        ]

        response_content = client._chat_completion(messages, temperature=1)

        # Parse JSON response
        guidance_result = json.loads(response_content)

        return guidance_result

    except Exception as e:
        console.print(f"[red]Error getting conversation guidance: {str(e)}[/red]")
        return None


def get_missing_critical_info(booking_data: Dict[str, Any]) -> List[str]:
    """Determine what critical information is missing"""
    missing = []

    # Critical information checklist
    if not booking_data.get("service_type"):
        missing.append("service_type")
    if not booking_data.get("problem_summary"):
        missing.append("problem_summary")
    if not booking_data.get("contact_name"):
        missing.append("contact_name")
    if not booking_data.get("contact_phone"):
        missing.append("contact_phone")

    return missing


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


def save_booking_record_simple(booking_data: Dict):
    """Save booking record with original data only"""

    try:
        # Build record using only original booking_data
        record = {
            "booking_data": booking_data,  # Original collected data
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save to file
        filename = "booking_records.jsonl"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(f"[dim]Booking record saved to {filename}[/dim]")
        console.print("\n[bold blue]Booking process completed![/bold blue]")
        console.print(
            "Our customer service team will contact you within 24 hours to confirm specific arrangements."
        )

    except Exception as e:
        console.print(f"[yellow]Failed to save record: {str(e)}[/yellow]")


if __name__ == "__main__":
    main()
