#!/usr/bin/env python3
"""
Gumroad Product Creator - Automated Product Setup
Creates Chronogrid product on Gumroad and sets up license delivery
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()
console = Console()

def get_api_token():
    """Get Gumroad API token from environment or prompt user."""
    token = os.getenv('GUMROAD_ACCESS_TOKEN')

    if not token:
        console.print("[yellow]⚠️  Gumroad API token not found in environment[/yellow]")
        console.print("Please enter your Gumroad access token:")
        console.print("[dim](Get this from: https://app.gumroad.com/settings/applications)[/dim]")
        token = Prompt.ask("Access Token", password=True)

        if not token:
            console.print("[red]❌ No token provided[/red]")
            return None

        # Save to .env for future use
        env_file = Path('.env')
        env_content = env_file.read_text() if env_file.exists() else ""
        if 'GUMROAD_ACCESS_TOKEN=' not in env_content:
            with open(env_file, 'a') as f:
                f.write(f"\nGUMROAD_ACCESS_TOKEN={token}\n")
            console.print("[green]✓ Token saved to .env file[/green]")

    return token

def test_api_connection(token):
    """Test API connection and return user info."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Testing Gumroad API connection...", total=None)

        try:
            response = requests.get(
                "https://api.gumroad.com/v2/user",
                params={'access_token': token},
                timeout=10
            )

            if response.status_code == 200:
                progress.update(task, description="[green]✓ API connection successful!")
                return response.json()['user']
            else:
                progress.update(task, description=f"[red]❌ API error: {response.status_code}")
                console.print(f"[red]Response: {response.text}[/red]")
                return None

        except Exception as e:
            progress.update(task, description=f"[red]❌ Connection error: {str(e)}")
            return None

def create_chronogrid_product(token):
    """Create the Chronogrid Pro License product on Gumroad."""
    console.print("[blue]📦 Creating Chronogrid Pro License product...[/blue]")

    product_data = {
        'access_token': token,
        'name': 'Chronogrid Pro License - $49',
        'description': '''Unlock unlimited AI analysis in Chronogrid - Professional Video Forensic Suite!

🎯 What you get:
• Unlimited AI-powered video analysis
• Advanced forensic video processing
• Professional-grade chronogrid generation
• Lifetime license - one-time payment
• Instant download after purchase

🔬 Perfect for:
• Law enforcement & investigators
• Video forensics professionals
• Content creators & analysts
• Security & surveillance teams

💡 Features:
• AI-powered scene analysis
• Temporal video processing
• Evidence extraction tools
• Multi-format support
• Professional reporting

Your license key will be delivered instantly after purchase.''',
        'price': 4900,  # $49.00 in cents
        'tags': 'AI,Video,Forensics,Analysis,Professional,License',
        'max_purchase_count': 1,  # One per customer
        'custom_receipt': 'Thank you for purchasing Chronogrid Pro! Your license key is attached.',
        'custom_permalink': 'chronogrid-pro-license'
    }

    try:
        response = requests.post(
            "https://api.gumroad.com/v2/products",
            data=product_data,
            timeout=15
        )

        if response.status_code == 200:
            product = response.json()['product']
            console.print("[green]✅ Product created successfully![/green]")

            product_info = Panel.fit(
                f"[bold]Name:[/bold] {product['name']}\n"
                f"[bold]ID:[/bold] {product['id']}\n"
                f"[bold]URL:[/bold] https://gumroad.com/l/{product['permalink']}\n"
                f"[bold]Price:[/bold] ${product['price']/100:.2f}\n"
                f"[bold]Status:[/bold] {product['published']}",
                title="🛍️ Product Created"
            )
            console.print(product_info)

            return product

        elif response.status_code == 422:
            # Product might already exist, try to find it
            console.print("[yellow]⚠️  Product might already exist, checking...[/yellow]")
            existing_product = find_existing_product(token)
            if existing_product:
                return existing_product
            else:
                console.print(f"[red]❌ Failed to create product: {response.text}[/red]")
                return None
        else:
            console.print(f"[red]❌ Failed to create product: {response.status_code}[/red]")
            console.print(f"Response: {response.text}")
            return None

    except Exception as e:
        console.print(f"[red]❌ Error creating product: {e}[/red]")
        return None

def find_existing_product(token):
    """Find existing Chronogrid product."""
    try:
        response = requests.get(
            "https://api.gumroad.com/v2/products",
            params={'access_token': token},
            timeout=10
        )

        if response.status_code == 200:
            products = response.json()['products']
            for product in products:
                if 'chronogrid' in product['name'].lower():
                    console.print(f"[green]✓ Found existing product: {product['name']}[/green]")
                    return product

        console.print("[yellow]⚠️  No existing Chronogrid product found[/yellow]")
        return None

    except Exception as e:
        console.print(f"[red]❌ Error searching products: {e}[/red]")
        return None

def generate_license_files(count=100):
    """Generate license files for upload."""
    console.print(f"[blue]🔑 Generating {count} license files...[/blue]")

    # Run the license generation script
    import subprocess
    result = subprocess.run([
        sys.executable, 'scripts/generate_license.py', '--gumroad', str(count)
    ], capture_output=True, text=True, cwd=Path(__file__).parent)

    if result.returncode == 0:
        console.print("[green]✅ License files generated successfully![/green]")
        license_dir = Path('gumroad_licenses')
        if license_dir.exists():
            files = list(license_dir.glob('*.txt'))
            console.print(f"📁 Created {len(files)} license files in '{license_dir}' folder")
        return True
    else:
        console.print("[red]❌ Failed to generate license files[/red]")
        console.print(f"Error: {result.stderr}")
        return False

def upload_license_files(token, product_id):
    """Upload license files to Gumroad product."""
    license_dir = Path('gumroad_licenses')
    if not license_dir.exists():
        console.print("[red]❌ License files directory not found[/red]")
        return False

    license_files = list(license_dir.glob('*.txt'))
    if not license_files:
        console.print("[red]❌ No license files found[/red]")
        return False

    console.print(f"[blue]📤 Uploading {len(license_files)} license files to Gumroad...[/blue]")

    # Note: Gumroad API doesn't directly support uploading multiple files via API
    # The files need to be uploaded manually through the web interface
    # But we can provide instructions

    console.print("[yellow]⚠️  Gumroad API doesn't support file uploads directly[/yellow]")
    console.print("[blue]Please upload the license files manually:[/blue]")
    console.print(f"1. Go to your product: https://app.gumroad.com/products/{product_id}/edit")
    console.print("2. Go to 'Digital Content' section")
    console.print(f"3. Upload all {len(license_files)} .txt files from the '{license_dir}' folder")
    console.print("4. Each file contains one unique license key")

    return True

def main():
    console.print("[bold green]🚀 Gumroad Product Creator[/bold green]\n")

    # Get API token
    token = get_api_token()
    if not token:
        sys.exit(1)

    # Test connection
    user = test_api_connection(token)
    if not user:
        console.print("[red]❌ API test failed. Please check your token.[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Connected as: {user['name']} ({user['email']})[/green]\n")

    # Check for existing product
    existing_product = find_existing_product(token)
    if existing_product:
        console.print(f"[blue]📦 Found existing product: {existing_product['name']}[/blue]")
        if Confirm.ask("Do you want to update this product instead of creating a new one?"):
            product = existing_product
        else:
            product = create_chronogrid_product(token)
    else:
        product = create_chronogrid_product(token)

    if not product:
        console.print("[red]❌ Failed to set up product[/red]")
        sys.exit(1)

    # Generate license files
    if not generate_license_files():
        console.print("[red]❌ Failed to generate license files[/red]")
        sys.exit(1)

    # Upload instructions
    upload_license_files(token, product['id'])

    # Success summary
    console.print("\n[bold green]🎉 Setup Complete![/bold green]")

    summary = Panel.fit(
        f"[bold]Product URL:[/bold] https://gumroad.com/l/{product['permalink']}\n"
        f"[bold]Product ID:[/bold] {product['id']}\n"
        f"[bold]Price:[/bold] ${product['price']/100:.2f}\n"
        f"[bold]License Files:[/bold] {len(list(Path('gumroad_licenses').glob('*.txt')))} generated\n"
        f"[bold]Next Step:[/bold] Upload license files to product in Gumroad dashboard",
        title="📋 Summary"
    )
    console.print(summary)

    console.print("\n[blue]🔗 Share this sales link: [bold]https://gumroad.com/l/{product['permalink']}[/bold][/blue]")
    console.print("[green]✅ Customers will get instant license key downloads![/green]")

if __name__ == "__main__":
    main()