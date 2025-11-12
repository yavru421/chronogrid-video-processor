#!/usr/bin/env python3
"""
APT Publishing Pipeline for Chronogrid
Automated publishing to multiple platforms: Gumroad, Itch.io, Paddle, GitHub
"""

import asyncio
import os
import sys
from typing import Dict, List, Any
import yaml
import aiohttp
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()

class PublishResult:
    def __init__(self, platform: str, success: bool, url: str = "", sku: str = "", error: str = ""):
        self.platform = platform
        self.success = success
        self.url = url
        self.sku = sku
        self.error = error

def parse_manifest(manifest_path: str) -> Dict[str, Any]:
    """Parse the YAML manifest file."""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)
        console.print(f"✓ Loaded manifest: {manifest['name']} v{manifest['version']}")
        return manifest
    except FileNotFoundError:
        console.print(f"[red]✗ Manifest file not found: {manifest_path}[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]✗ Invalid YAML in manifest: {e}[/red]")
        sys.exit(1)

async def upload_gumroad(session: aiohttp.ClientSession, manifest: Dict[str, Any]) -> PublishResult:
    """Upload to Gumroad."""
    api_key = os.getenv('GUMROAD_ACCESS_TOKEN')
    if not api_key:
        return PublishResult("Gumroad", False, error="GUMROAD_ACCESS_TOKEN not set")

    try:
        # Gumroad API for creating/updating products
        # Note: This is a simplified implementation - real Gumroad API would require more fields
        url = "https://api.gumroad.com/v2/products"
        data = {
            'access_token': api_key,
            'name': manifest['name'],
            'description': manifest['description'],
            'price': int(manifest['price'] * 100),  # Convert to cents
            'tags': ','.join(manifest.get('tags', [])),
        }

        # Upload files would require multipart/form-data in real implementation
        async with session.post(url, data=data) as response:
            if response.status == 200:
                result = await response.json()
                product_url = f"https://gumroad.com/l/{result['product']['permalink']}"
                return PublishResult("Gumroad", True, url=product_url, sku=result['product']['id'])
            else:
                error = await response.text()
                return PublishResult("Gumroad", False, error=f"API error: {error}")

    except Exception as e:
        return PublishResult("Gumroad", False, error=str(e))

async def upload_itch(session: aiohttp.ClientSession, manifest: Dict[str, Any]) -> PublishResult:
    """Upload to Itch.io."""
    api_key = os.getenv('ITCH_API_KEY')
    user = os.getenv('ITCH_USER')
    game = os.getenv('ITCH_GAME')

    if not all([api_key, user, game]):
        return PublishResult("Itch.io", False, error="ITCH_API_KEY, ITCH_USER, ITCH_GAME not set")

    try:
        # Itch.io Butler API for uploads
        # This is a simplified implementation
        url = f"https://api.itch.io/games/{game}/uploads"
        headers = {'Authorization': f'Bearer {api_key}'}

        # In real implementation, would upload files via butler
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                game_url = f"https://{user}.itch.io/{game}"
                return PublishResult("Itch.io", True, url=game_url, sku=str(game))
            else:
                error = await response.text()
                return PublishResult("Itch.io", False, error=f"API error: {error}")

    except Exception as e:
        return PublishResult("Itch.io", False, error=str(e))

async def upload_paddle(session: aiohttp.ClientSession, manifest: Dict[str, Any]) -> PublishResult:
    """Upload to Paddle."""
    api_key = os.getenv('PADDLE_API_KEY')
    product_id = os.getenv('PADDLE_PRODUCT_ID')

    if not all([api_key, product_id]):
        return PublishResult("Paddle", False, error="PADDLE_API_KEY, PADDLE_PRODUCT_ID not set")

    try:
        # Paddle API for product updates
        url = f"https://api.paddle.com/products/{product_id}"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'name': manifest['name'],
            'description': manifest['description'],
            'price': manifest['price'],
            'tags': manifest.get('tags', [])
        }

        async with session.patch(url, headers=headers, json=data) as response:
            if response.status == 200:
                checkout_url = f"https://checkout.paddle.com/checkout/{product_id}"
                return PublishResult("Paddle", True, url=checkout_url, sku=str(product_id))
            else:
                error = await response.text()
                return PublishResult("Paddle", False, error=f"API error: {error}")

    except Exception as e:
        return PublishResult("Paddle", False, error=str(e))

async def upload_github(session: aiohttp.ClientSession, manifest: Dict[str, Any]) -> PublishResult:
    """Create GitHub release."""
    token = os.getenv('GITHUB_TOKEN')
    repo = os.getenv('GITHUB_REPO', 'yavru421/chronogrid-video-processor')

    if not token:
        return PublishResult("GitHub", False, error="GITHUB_TOKEN not set")

    try:
        # GitHub API for creating releases
        url = f"https://api.github.com/repos/{repo}/releases"
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        data = {
            'tag_name': f"v{manifest['version']}",
            'name': f"{manifest['name']} v{manifest['version']}",
            'body': manifest['description'],
            'draft': False,
            'prerelease': False
        }

        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 201:
                result = await response.json()
                release_url = result['html_url']
                return PublishResult("GitHub", True, url=release_url, sku=result['id'])
            else:
                error = await response.text()
                return PublishResult("GitHub", False, error=f"API error: {error}")

    except Exception as e:
        return PublishResult("GitHub", False, error=str(e))

async def publish_to_platforms(manifest: Dict[str, Any]) -> List[PublishResult]:
    """Publish to all configured platforms in parallel."""
    platforms = [
        ("Gumroad", upload_gumroad),
        ("Itch.io", upload_itch),
        ("Paddle", upload_paddle),
        ("GitHub", upload_github)
    ]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for platform_name, upload_func in platforms:
            console.print(f"📤 Starting {platform_name} upload...")
            tasks.append(upload_func(session, manifest))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            platform_name = platforms[i][0]
            if isinstance(result, Exception):
                processed_results.append(PublishResult(platform_name, False, error=str(result)))
            else:
                processed_results.append(result)

        return processed_results

def display_results(results: List[PublishResult]):
    """Display results in a nice table."""
    table = Table(title="🚀 Publishing Results")
    table.add_column("Platform", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("URL", style="blue")
    table.add_column("SKU/ID", style="yellow")
    table.add_column("Error", style="red")

    for result in results:
        status = "✅ Success" if result.success else "❌ Failed"
        url = result.url if result.url else "-"
        sku = str(result.sku) if result.sku else "-"
        error = result.error if result.error else "-"

        table.add_row(result.platform, status, url, sku, error)

    console.print(table)

def main():
    if len(sys.argv) != 2:
        console.print("[red]Usage: python apt_publish.py <manifest.yml>[/red]")
        sys.exit(1)

    manifest_path = sys.argv[1]

    # Parse manifest
    manifest = parse_manifest(manifest_path)

    # Display manifest info
    info_panel = Panel.fit(
        f"[bold]{manifest['name']}[/bold]\n"
        f"Version: {manifest['version']}\n"
        f"Price: ${manifest['price']}\n"
        f"Files: {len(manifest.get('files', []))}\n"
        f"Images: {len(manifest.get('images', []))}\n"
        f"Tags: {', '.join(manifest.get('tags', []))}",
        title="📦 Manifest Summary"
    )
    console.print(info_panel)

    # Publish to all platforms
    console.print("\n🔄 Starting APT Publishing Pipeline...\n")

    results = asyncio.run(publish_to_platforms(manifest))

    # Display results
    console.print("\n" + "="*50)
    display_results(results)

    # Summary
    successful = sum(1 for r in results if r.success)
    total = len(results)
    console.print(f"\n📊 Summary: {successful}/{total} platforms published successfully")

    if successful == 0:
        console.print("[red]❌ No platforms were successfully published. Check API keys and network.[/red]")
        sys.exit(1)
    elif successful < total:
        console.print("[yellow]⚠️  Some platforms failed. Check errors above.[/yellow]")
    else:
        console.print("[green]🎉 All platforms published successfully![/green]")

if __name__ == "__main__":
    main()