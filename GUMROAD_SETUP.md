# Gumroad Application Setup Guide

## Step-by-Step: Create Your Gumroad API Application

### 1. Access Application Settings
Try these URLs in order (one should work):
- https://app.gumroad.com/settings/applications (recommended)
- https://gumroad.com/settings/applications
- Or: Go to gumroad.com → Login → Click your profile → Settings → Applications

If the page doesn't load, try:
- Clear your browser cache
- Use an incognito/private window
- Try a different browser
- Make sure you're logged into Gumroad

### 2. Create New Application
- Click the **"Create Application"** button
- Fill in the application details:
  - **Name**: `Chronogrid Publisher` (or any name you prefer)
  - **Description**: `Automated publishing system for Chronogrid releases`
  - **Website URL**: `https://github.com/yavru421/chronogrid-video-processor` (or your website)
  - **Redirect URI**: Leave blank (not needed for our use case)

### 3. Generate Access Token
- After creating the application, you'll see the application details page
- Look for the **"Access Token"** section
- Click **"Generate Token"**
- **IMPORTANT**: Copy the token immediately - it will only be shown once!

### 4. Configure Your Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your token
GUMROAD_ACCESS_TOKEN=your_actual_token_here
```

### 5. Test the Connection
```bash
python test_gumroad_api.py
```

This script will:
- ✅ Verify your token is valid
- ✅ Show your Gumroad account information
- 🔄 Optionally create a test product to verify write permissions

### 6. Expected Output
```
🔍 Testing Gumroad API token: abc123...
✅ Gumroad API token is valid!

👤 Gumroad Account Info
Name: Your Name
Email: your@email.com
Bio: Your bio here
```

### 7. Ready for Publishing
Once the test passes, you can run:
```bash
python apt_publish.py chronogrid.yml
```

## Troubleshooting

### "Invalid API token" Error
- Double-check you copied the token correctly
- Make sure there are no extra spaces
- Try regenerating the token in Gumroad settings

### "Application not found" Error
- Verify you're using the correct application
- Check that the application wasn't deleted

### Network Errors
- Ensure you have internet connection
- Gumroad API might be temporarily down

## Security Notes
- Keep your access token secure and never commit it to version control
- The token provides full access to your Gumroad account
- Regenerate tokens periodically for security
- Use environment variables or .env files (never hardcode tokens)

## API Permissions
Your application will have these permissions:
- ✅ Read user information
- ✅ Create and manage products
- ✅ Access sales data
- ✅ Manage digital content

The APT publishing system only uses product creation/management features.