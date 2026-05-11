import argparse
import requests
import json
import os
import sys
import subprocess
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8000")
console = Console()

def check_health():
    with console.status("[bold green]Checking API health...") as status:
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                console.print(Panel("[bold green][DONE] API is healthy and reachable![/bold green]", title="Health Check", border_style="green"))
                return True
            else:
                console.print(Panel(f"[bold red][FAIL] API returned status code {response.status_code}[/bold red]", title="Health Check", border_style="red"))
        except requests.exceptions.ConnectionError:
            console.print(Panel(f"[bold red][ERROR] Could not connect to API at {API_URL}. Is the server running?[/bold red]\n\nTry running: [bold cyan]python cli.py serve[/bold cyan]", title="Health Check", border_style="red"))
        except Exception as e:
            console.print(Panel(f"[bold red][ERROR] Error: {e}[/bold red]", title="Health Check", border_style="red"))
    return False

def run_ingestion():
    console.print(Panel("[bold blue]Starting Ingestion Pipeline[/bold blue]\nThis will process the catalog, build indices, and sync with Pinecone.", border_style="blue"))
    
    try:
        # Import main from ingestion.py
        from scripts.ingestion import main as start_ingest
        start_ingest()
        console.print("\n[bold green][DONE] Ingestion pipeline completed successfully![/bold green]")
    except ImportError:
        console.print("[bold red][ERROR] Error: Could not find ingestion.py or its dependencies.[/bold red]")
    except Exception as e:
        console.print(f"[bold red][ERROR] Ingestion failed: {e}[/bold red]")

def chat():
    if not check_health():
        return

    messages = []
    console.print(Panel("[bold magenta]SHL Assessment Recommender Chat[/bold magenta]\nType [bold cyan]'exit'[/bold cyan] to quit. Conversation state is maintained locally.", border_style="magenta"))
    
    while True:
        try:
            user_input = console.input("[bold yellow]You:[/bold yellow] ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            with console.status("[bold cyan]Agent is thinking...") as status:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"messages": messages},
                    timeout=60
                )
                
            if response.status_code == 200:
                data = response.json()
                reply = data.get("reply", "")
                recommendations = data.get("recommendations", [])
                
                # Display AI Response
                console.print(Panel(Markdown(reply), title="[bold blue]AI Recommender[/bold blue]", border_style="blue"))
                
                if recommendations:
                    table = Table(title="Recommended Assessments", show_header=True, header_style="bold magenta")
                    table.add_column("#", style="dim", width=2)
                    table.add_column("Name", style="bold white")
                    table.add_column("Type", style="cyan")
                    table.add_column("Duration", style="green")
                    
                    for i, rec in enumerate(recommendations, 1):
                        table.add_row(str(i), rec['name'], rec['test_type'], rec['duration'])
                    
                    console.print(table)
                    
                    # Detailed rationales
                    for i, rec in enumerate(recommendations, 1):
                        console.print(Panel(f"[dim]{rec['rationale']}[/dim]\n[bold cyan]Link:[/bold cyan] {rec['url']}", title=f"Rationale for {rec['name']}", border_style="dim"))
                
                messages.append({"role": "assistant", "content": reply})
                
                if data.get("end_of_conversation"):
                    console.print("\n[bold yellow]! Conversation marked as complete by AI.[/bold yellow]")
                    # Keep chatting unless they exit? Let's stay in the loop.
            else:
                console.print(f"[bold red]Error {response.status_code}:[/bold red] {response.text}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

def serve():
    console.print(Panel("[bold green]Starting FastAPI Server...[/bold green]\nUvicorn will run on http://localhost:8000", border_style="green"))
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], check=True)
    except FileNotFoundError:
        console.print("[bold red][ERROR] Error: 'uvicorn' not found. Please install requirements: pip install -r requirements.txt[/bold red]")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Server stopped.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error starting server: {e}[/bold red]")

def main():
    parser = argparse.ArgumentParser(description="SHL Assessment Recommender CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("health", help="Check if the API server is running")
    subparsers.add_parser("ingest", help="Run the data ingestion pipeline")
    subparsers.add_parser("chat", help="Start an interactive chat session")
    subparsers.add_parser("serve", help="Start the FastAPI server")
    
    args = parser.parse_args()
    
    if args.command == "health":
        check_health()
    elif args.command == "ingest":
        run_ingestion()
    elif args.command == "chat":
        chat()
    elif args.command == "serve":
        serve()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
