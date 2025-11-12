#!/usr/bin/env python3
"""
Complete Gumroad Setup Automation Script
This script handles the entire Gumroad integration setup process
"""

import os
import sys
import webbrowser
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load existing environment
load_dotenv()
console = Console()

class GumroadSetup:
    def __init__(self):
        self.console = console
        self.env_file = Path('.env')

    def step_header(self, step_num: int, title: str, description: str = ""):
        """Display a step header."""
        header = f"[bold blue]Step {step_num}: {title}[/bold blue]"
        if description:
            header += f"\n[dim]{description}[/dim]"

        panel = Panel.fit(header, border_style="blue")
        self.console.print(panel)
        self.console.print()

    def check_existing_setup(self):
        """Check if Gumroad is already configured."""
        token = os.getenv('GUMROAD_ACCESS_TOKEN')
        if token:
            self.console.print("[green]✓ Gumroad access token already configured![/green]")
            if Confirm.ask("Do you want to test the existing configuration?"):
                return self.test_api_connection()
            return True
        return False

    def open_gumroad_applications(self):
        """Open Gumroad applications page in browser."""
        # Use the correct URL for Gumroad applications
        url = "https://app.gumroad.com/settings/applications"
        self.console.print(f"[blue]🌐 Opening: {url}[/blue]")

        try:
            webbrowser.open(url)
            self.console.print("[green]✓ Browser opened to Gumroad applications page[/green]")
            self.console.print("[yellow]� If this page doesn't load, try: python troubleshoot_gumroad.py[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not open browser automatically: {e}[/yellow]")
            self.console.print(f"[blue]Please manually visit: {url}[/blue]")
            self.console.print("[blue]Or run: python troubleshoot_gumroad.py[/blue]")

    def guide_application_creation(self):
        """Guide user through creating the Gumroad application."""
        instructions = """
[bold]Create New Application:[/bold]
1. Click the [bold blue]"Create Application"[/bold blue] button
2. Fill in the details:
   • [bold]Name:[/bold] Chronogrid Publisher
   • [bold]Description:[/bold] Automated publishing for Chronogrid releases
   • [bold]Website URL:[/bold] https://github.com/yavru421/chronogrid-video-processor
   • [bold]Redirect URI:[/bold] (leave blank)
3. Click [bold blue]"Create Application"[/bold blue]

[bold]Generate Access Token:[/bold]
4. On the application page, find [bold blue]"Access Token"[/bold blue] section
5. Click [bold blue]"Generate Token"[/bold blue]
6. [bold red]IMPORTANT:[/bold red] Copy the token immediately (shown only once!)
"""

        panel = Panel.fit(instructions, title="📝 Gumroad Application Setup Instructions")
        self.console.print(panel)

        if Confirm.ask("Have you completed the application creation and copied the access token?"):
            return True
        else:
            self.console.print("[yellow]Take your time - this script will wait.[/yellow]")
            return False

    def get_access_token(self):
        """Prompt user for their access token."""
        self.console.print()
        self.console.print("[bold]🔑 Enter Your Gumroad Access Token[/bold]")
        self.console.print("[dim]This was shown only once when you generated it.[/dim]")
        self.console.print("[dim]If you lost it, you'll need to generate a new one.[/dim]")
        self.console.print()

        while True:
            token = Prompt.ask("Access Token", password=True)

            if not token or len(token.strip()) < 10:
                self.console.print("[red]❌ Token appears to be invalid (too short)[/red]")
                continue

            # Basic validation - should be a reasonable length
            if len(token) < 20:
                self.console.print("[yellow]⚠️  Token seems short. Gumroad tokens are usually longer.[/yellow]")
                if not Confirm.ask("Are you sure this is correct?"):
                    continue

            return token.strip()

    def save_token_to_env(self, token: str):
        """Save the token to .env file."""
        # Read existing .env content
        env_content = ""
        if self.env_file.exists():
            env_content = self.env_file.read_text()

        # Update or add GUMROAD_ACCESS_TOKEN
        lines = env_content.split('\n')
        token_line = f"GUMROAD_ACCESS_TOKEN={token}"
        token_found = False

        for i, line in enumerate(lines):
            if line.startswith('GUMROAD_ACCESS_TOKEN='):
                lines[i] = token_line
                token_found = True
                break

        if not token_found:
            lines.append(token_line)

        # Write back to file
        self.env_file.write_text('\n'.join(lines))

        # Reload environment
        load_dotenv()

        self.console.print("[green]✓ Access token saved to .env file[/green]")

    def test_api_connection(self):
        """Test the API connection with the token."""
        token = os.getenv('GUMROAD_ACCESS_TOKEN')

        if not token:
            self.console.print("[red]❌ No access token found[/red]")
            return False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Testing Gumroad API connection...", total=None)

            try:
                # Test token by getting user info
                url = "https://api.gumroad.com/v2/user"
                params = {'access_token': token}

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    progress.update(task, description="[green]✓ API connection successful!")
                    time.sleep(1)  # Show success briefly

                    user_data = response.json()
                    user_info = Panel.fit(
                        f"[bold]Name:[/bold] {user_data['user']['name']}\n"
                        f"[bold]Email:[/bold] {user_data['user']['email']}\n"
                        f"[bold]Bio:[/bold] {user_data['user'].get('bio', 'N/A')}",
                        title="👤 Gumroad Account Verified"
                    )
                    self.console.print(user_info)
                    return True

                elif response.status_code == 401:
                    progress.update(task, description="[red]❌ Invalid access token")
                    time.sleep(1)
                    self.console.print("[red]The access token is invalid. Please check and try again.[/red]")
                    return False

                else:
                    progress.update(task, description=f"[red]❌ API error: {response.status_code}")
                    time.sleep(1)
                    self.console.print(f"[red]Gumroad API error: {response.text}[/red]")
                    return False

            except requests.exceptions.RequestException as e:
                progress.update(task, description=f"[red]❌ Network error: {str(e)}")
                time.sleep(1)
                self.console.print(f"[red]Network error: {e}[/red]")
                return False

    def create_test_product(self):
        """Optionally create a test product to verify write permissions."""
        if not Confirm.ask("Would you like to create a test product to verify write permissions?"):
            return

        token = os.getenv('GUMROAD_ACCESS_TOKEN')
        if not token:
            return

        self.console.print("[blue]📤 Creating test product...[/blue]")

        try:
            url = "https://api.gumroad.com/v2/products"
            data = {
                'access_token': token,
                'name': 'Chronogrid Test Product - DELETE ME',
                'description': 'This is a test product created by the setup script. Please delete it from your Gumroad dashboard.',
                'price': 100,  # $1.00 in cents
                'tags': 'test,chronogrid,setup'
            }

            response = requests.post(url, data=data, timeout=15)

            if response.status_code == 200:
                product_data = response.json()
                product = product_data['product']

                self.console.print("[green]✅ Test product created successfully![/green]")

                product_info = Panel.fit(
                    f"[bold]Name:[/bold] {product['name']}\n"
                    f"[bold]ID:[/bold] {product['id']}\n"
                    f"[bold]URL:[/bold] https://gumroad.com/l/{product['permalink']}\n"
                    f"[bold]Price:[/bold] ${product['price']/100:.2f}",
                    title="🧪 Test Product Created"
                )
                self.console.print(product_info)

                self.console.print("[yellow]⚠️  Remember to delete this test product from your Gumroad dashboard![/yellow]")

            else:
                self.console.print(f"[red]❌ Failed to create test product: {response.status_code}[/red]")
                self.console.print(f"Response: {response.text}")

        except requests.exceptions.RequestException as e:
            self.console.print(f"[red]❌ Network error: {e}[/red]")

    def run_full_setup(self):
        """Run the complete setup process."""
        self.console.print("[bold green]🚀 Complete Gumroad Setup Automation[/bold green]\n")

        # Check if already configured
        if self.check_existing_setup():
            self.console.print("[green]🎉 Gumroad is already configured and working![/green]")
            return

        # Step 1: Open Gumroad applications page
        self.step_header(
            1,
            "Access Gumroad Applications",
            "We'll open your browser to the Gumroad applications page"
        )
        self.open_gumroad_applications()

        # Step 2: Guide through application creation
        self.step_header(
            2,
            "Create Gumroad Application",
            "Follow the detailed instructions below"
        )
        while not self.guide_application_creation():
            time.sleep(1)  # Brief pause

        # Step 3: Get access token
        self.step_header(
            3,
            "Enter Access Token",
            "Paste the token you generated in Step 2"
        )
        token = self.get_access_token()

        # Step 4: Save token
        self.step_header(
            4,
            "Save Configuration",
            "Saving your access token securely"
        )
        self.save_token_to_env(token)

        # Step 5: Test connection
        self.step_header(
            5,
            "Test API Connection",
            "Verifying your token works with Gumroad API"
        )
        if self.test_api_connection():
            # Step 6: Optional test product
            self.step_header(
                6,
                "Optional Test Product",
                "Create a test product to verify write permissions"
            )
            self.create_test_product()

            # Success message
            self.console.print("\n[bold green]🎉 Gumroad Setup Complete![/bold green]")
            self.console.print("You can now use: [bold]python apt_publish.py chronogrid.yml[/bold]")
        else:
            self.console.print("\n[red]❌ Setup failed. Please check your token and try again.[/red]")
            sys.exit(1)

def main():
    setup = GumroadSetup()
    setup.run_full_setup()

if __name__ == "__main__":
    main()