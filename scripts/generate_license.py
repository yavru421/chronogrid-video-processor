#!/usr/bin/env python3
"""
License Key Generator for Chronogrid Premium

Run this script after a successful Stripe payment to generate a license key for a customer.

Usage:
    python generate_license.py customer@example.com

The generated key should be emailed to the customer.
"""

import sys
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
import csv

# Secure secret key (keep this private!)
LICENSE_SECRET = base64.b64decode("jIMw+tRbQKw36ZsENwsNR6nGvDJs/gXxBkXZR0XCNFE=")

def generate_license_key(email: str, expiry_days: int = 365*10) -> str:
    """
    Generate a license key for the given email.

    Format: base64(hmac_sha256(secret, email + expiry_timestamp))
    """
    expiry = datetime.now() + timedelta(days=expiry_days)
    expiry_timestamp = int(expiry.timestamp())

    # Create payload: email|expiry_timestamp
    payload = f"{email}|{expiry_timestamp}"

    # Generate HMAC-SHA256 signature
    signature = hmac.new(LICENSE_SECRET, payload.encode(), hashlib.sha256).digest()

    # Combine payload and signature, base64 encode
    license_data = f"{payload}|{base64.b64encode(signature).decode()}"
    license_key = base64.b64encode(license_data.encode()).decode()

    return license_key

def validate_license_key(license_key: str, email: str) -> bool:
    """
    Validate a license key for the given email.
    """
    try:
        # Decode license key
        license_data = base64.b64decode(license_key).decode()
        payload, signature_b64 = license_data.rsplit('|', 1)

        email_from_key, expiry_timestamp = payload.split('|', 1)
        expiry_timestamp = int(expiry_timestamp)

        # Verify email matches
        if email_from_key != email:
            return False

        # Verify expiry
        if datetime.now().timestamp() > expiry_timestamp:
            return False

        # Verify signature
        expected_signature = hmac.new(LICENSE_SECRET, payload.encode(), hashlib.sha256).digest()
        provided_signature = base64.b64decode(signature_b64)

        return hmac.compare_digest(expected_signature, provided_signature)

    except Exception:
        return False

def process_gumroad_batch(count: int) -> None:
    """Generate license files for Gumroad upload."""
    import os

    # Create output directory
    output_dir = "gumroad_licenses"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {count} license files for Gumroad...")

    for i in range(count):
        # Generate license with dummy email (not used for validation in Gumroad context)
        license_key = generate_license_key(f"gumroad_{i}@chronogrid.local")

        # Save as individual file
        filename = f"chronogrid_license_{i+1:04d}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(license_key)

    print(f"✓ Generated {count} license files in '{output_dir}/' folder")
    print("Upload all .txt files to your Gumroad product for digital delivery.")
    print("Each purchase will randomly receive one license file.")

def process_batch_orders(csv_file: str) -> None:
    """Process a batch of orders from CSV file."""
    licenses = []
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                email = row.get('email', '').strip()
                order_id = row.get('order_id', '').strip()
                if not email or not order_id:
                    print(f"Skipping invalid row: {row}")
                    continue

                license_key = generate_license_key(email)
                licenses.append({
                    'email': email,
                    'order_id': order_id,
                    'license_key': license_key
                })

        # Output results
        print("Batch license generation complete!")
        print(f"Generated {len(licenses)} licenses\n")

        # Save to CSV
        output_file = csv_file.replace('.csv', '_licenses.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['email', 'order_id', 'license_key'])
            writer.writeheader()
            writer.writerows(licenses)

        print(f"Results saved to: {output_file}")
        print("\nSample email template:")
        print("=" * 50)
        for license in licenses[:3]:  # Show first 3
            print(f"""
Subject: Your Chronogrid Premium License

Dear {license['email'].split('@')[0]},

Thank you for purchasing Chronogrid Premium!

Your license key: {license['license_key']}

To activate:
1. Open Chronogrid
2. Go to License > Enter License Key
3. Paste the key above

Enjoy unlimited AI analysis!

Best,
Chronogrid Team
""")

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found")
    except Exception as e:
        print(f"Error processing batch: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single: python generate_license.py customer@example.com")
        print("  Batch:  python generate_license.py --batch orders.csv")
        print("  Gumroad: python generate_license.py --gumroad 100")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) != 3:
            print("Usage: python generate_license.py --batch orders.csv")
            sys.exit(1)
        process_batch_orders(sys.argv[2])
    elif sys.argv[1] == "--gumroad":
        if len(sys.argv) != 3:
            print("Usage: python generate_license.py --gumroad <count>")
            sys.exit(1)
        try:
            count = int(sys.argv[2])
            if count <= 0:
                print("Count must be a positive integer")
                sys.exit(1)
            process_gumroad_batch(count)
        except ValueError:
            print("Count must be a valid integer")
            sys.exit(1)
    else:
        email = sys.argv[1].strip()
        if not email or '@' not in email:
            print("Invalid email address")
            sys.exit(1)

        license_key = generate_license_key(email)
        print(f"License key for {email}:")
        print(license_key)
        print("\nEmail this key to the customer.")

        # Test validation
        if validate_license_key(license_key, email):
            print("✓ License key validation successful")
        else:
            print("✗ License key validation failed")