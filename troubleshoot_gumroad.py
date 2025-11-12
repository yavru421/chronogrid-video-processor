#!/usr/bin/env python3
"""
Gumroad URL Troubleshooter
Help find the correct Gumroad applications page URL
"""

import webbrowser
from rich.console import Console
from rich.panel import Panel

console = Console()

def try_url(url: str, description: str) -> bool:
    """Try to open a URL and report success/failure."""
    console.print(f"[blue]🌐 Trying: {description}[/blue]")
    console.print(f"[dim]{url}[/dim]")

    try:
        webbrowser.open(url)
        console.print("[green]✓ Browser opened successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]❌ Failed: {e}[/red]")
        return False

def main():
    console.print("[bold blue]🔍 Gumroad URL Troubleshooter[/bold blue]\n")

    urls = [
        ("https://app.gumroad.com/settings/applications", "Gumroad App - Applications Page"),
        ("https://gumroad.com/settings/applications", "Gumroad Main - Applications Page"),
        ("https://app.gumroad.com/settings", "Gumroad App - Settings"),
        ("https://gumroad.com/settings", "Gumroad Main - Settings"),
        ("https://gumroad.com/dashboard", "Gumroad Dashboard"),
        ("https://gumroad.com/help/article/280-create-application-api", "API Documentation"),
    ]

    console.print("Let's try different Gumroad URLs to find the applications page:\n")

    successful_urls = []

    for url, description in urls:
        if try_url(url, description):
            successful_urls.append((url, description))
        console.print()

    if successful_urls:
        console.print("[green]✅ Successfully opened URLs:[/green]")
        for url, desc in successful_urls:
            console.print(f"  • {desc}: {url}")
    else:
        console.print("[red]❌ No URLs could be opened automatically[/red]")

    console.print("\n" + "="*50)

    # Manual navigation guide
    manual_guide = """
[bold]Manual Navigation Guide:[/bold]

1. Go to https://gumroad.com and log in
2. Click your profile picture/avatar (top right)
3. Look for "Settings" or "Account Settings"
4. Look for "Applications", "API", or "Developer" section
5. If you can't find it, search for "API" in Gumroad's help center

[bold]Alternative Access:[/bold]
- Try logging out and back in
- Clear browser cache and cookies
- Use a different browser or incognito mode
- Check if your Gumroad account has API access enabled
"""

    panel = Panel.fit(manual_guide, title="🗺️ Manual Access Instructions")
    console.print(panel)

    console.print("\n[blue]Once you find the applications page, create an application called 'Chronogrid Publisher'[/blue]")
    console.print("[blue]Then run: [bold]python setup_gumroad.py[/bold] again and paste your access token[/blue]")

if __name__ == "__main__":
    main()