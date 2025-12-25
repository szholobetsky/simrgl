# 🔒 Security Configuration Guide

## Overview

The Gradio UI can expose your code search system, which contains sensitive information about your codebase. This guide explains security options.

## ⚙️ Configuration Options

Edit `config.py` to change security settings:

### 1. **Localhost Only** (Default - Most Secure) ✅

```python
GRADIO_ACCESS_MODE = 'localhost'
```

**Access:**
- ✅ Only from the same computer (127.0.0.1)
- ❌ NOT from other devices on network
- ❌ NOT from internet

**Use When:**
- Working on your development machine
- Testing locally
- Handling sensitive code

**URL:** `http://127.0.0.1:7860`

---

### 2. **Network Access** (Less Secure) ⚠️

```python
GRADIO_ACCESS_MODE = 'network'
```

**Access:**
- ✅ From any device on your local network
- ❌ NOT from internet (unless ports forwarded)

**Use When:**
- Sharing with team members on same network
- Accessing from multiple devices (laptop, desktop, tablet)
- Internal company network

**Security Note:** Anyone on your network can access!

**Recommended:** Add authentication (see below)

**URL:** `http://your-ip-address:7860`

---

### 3. **Public Access** (Least Secure) 🚨

```python
GRADIO_ACCESS_MODE = 'public'
GRADIO_SHARE = True
```

**Access:**
- ✅ From anywhere on the internet via Gradio share link

**⚠️ WARNING:**
- Your code data is exposed to the internet!
- Anyone with the link can search your codebase
- Share link is public and temporary

**Use When:**
- Demo purposes only
- Non-sensitive data
- Short-term sharing

**NEVER use for production code!**

---

## 🔐 Adding Authentication

Protect your UI with username/password.

### Enable Authentication:

Edit `config.py`:

```python
# Uncomment and set your credentials:
GRADIO_USERNAME = 'admin'
GRADIO_PASSWORD = 'your_strong_password_here'
```

**Best Practices:**
- Use a strong password (12+ characters)
- Don't commit passwords to git (add config.py to .gitignore)
- Change default passwords
- Use different passwords for different deployments

### Multiple Users:

For multiple users, you can use a list:

```python
GRADIO_AUTH = [
    ('admin', 'admin_password'),
    ('developer', 'dev_password'),
    ('viewer', 'viewer_password')
]
```

Then modify `gradio_ui.py`:
```python
auth = config.GRADIO_AUTH  # Instead of tuple
```

---

## 🛡️ Security Recommendations

### For Development (Your PC):
```python
GRADIO_ACCESS_MODE = 'localhost'  # Localhost only
# No authentication needed
```

### For Team Use (Company Network):
```python
GRADIO_ACCESS_MODE = 'network'  # Network access
GRADIO_USERNAME = 'team'
GRADIO_PASSWORD = 'strong_password_123!'
```

### For Demos (Temporary):
```python
GRADIO_ACCESS_MODE = 'public'  # Public share link
GRADIO_USERNAME = 'demo'
GRADIO_PASSWORD = 'demo_pass'
GRADIO_SHARE = True

# ⚠️ Use ONLY for non-sensitive demo data!
```

---

## 🔍 Security Checklist

Before running the UI:

- [ ] Checked `GRADIO_ACCESS_MODE` in config.py
- [ ] Verified it matches your use case
- [ ] Added authentication if using 'network' or 'public'
- [ ] Confirmed no sensitive credentials in code
- [ ] Tested access from expected locations only
- [ ] Considered firewall rules if needed

---

## 🚪 Port Configuration

Default port: **7860**

To change:
```python
GRADIO_PORT = 8080  # Or any available port
```

**Firewall:**
- Localhost mode: No firewall rules needed
- Network mode: Allow port on local firewall
- Public mode: Gradio handles tunneling

---

## 🔒 Additional Security Measures

### 1. Use HTTPS (Production)

For production deployments, use a reverse proxy with SSL:

```nginx
# nginx example
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:7860;
    }
}
```

### 2. Rate Limiting

Add to `gradio_ui.py`:
```python
from gradio.themes import Default

demo = gr.Blocks(
    max_threads=10,  # Limit concurrent requests
    ...
)
```

### 3. IP Whitelisting

In production, use firewall rules or nginx:
```nginx
# Allow only specific IPs
allow 192.168.1.0/24;  # Your network
deny all;
```

---

## 🧪 Testing Security

### Test Localhost Mode:
```bash
# From same computer - should work
curl http://127.0.0.1:7860

# From another computer - should fail
curl http://your-ip:7860
```

### Test Network Mode:
```bash
# From another computer on network - should work
curl http://your-ip:7860
```

### Test Authentication:
```bash
# Without auth - should redirect to login
curl http://localhost:7860

# With auth - should work
curl -u admin:password http://localhost:7860
```

---

## ❓ FAQ

**Q: Can I access the UI from my phone on the same Wi-Fi?**

A: Yes, use `GRADIO_ACCESS_MODE = 'network'` and navigate to `http://your-pc-ip:7860`

**Q: Is the data encrypted?**

A: Not by default. Use HTTPS with reverse proxy for encryption.

**Q: What if I forget my password?**

A: Edit `config.py` to change `GRADIO_PASSWORD`

**Q: Can I disable authentication temporarily?**

A: Yes, comment out `GRADIO_USERNAME` and `GRADIO_PASSWORD` in config.py

**Q: Is localhost mode safe for sensitive code?**

A: Yes, it's only accessible from your computer. But anyone with access to your computer can use it.

---

## 📝 Summary

**Recommended Configuration:**

```python
# config.py

# For solo development
GRADIO_ACCESS_MODE = 'localhost'  # Most secure

# For team use
GRADIO_ACCESS_MODE = 'network'
GRADIO_USERNAME = 'team'
GRADIO_PASSWORD = 'secure_password_123'

# Never for production code
# GRADIO_ACCESS_MODE = 'public'  # ❌ Don't use!
```

**Default is SAFE:** The system defaults to `localhost` mode with no authentication.

---

**Last Updated:** 2025-12-21
**Version:** 1.0
