# UX Design: Headless Mini-PC Image Setup

> Concept document for a pre-configured Linux image for the SIMARGL/PANTEON tool ecosystem.
> Target: headless mini-PC (no monitor, no keyboard) accessible over LAN.

---

## Vision

A user buys a mini-PC, flashes a pre-built image, inserts a USB key, powers on —
and within 60 seconds every tool is reachable in a browser on any device connected
to the same LAN cable or home WiFi. No Linux knowledge required.

Hostname: **panteon.local**

---

## Part 1 — Disk Encryption with USB Key (LUKS)

### Goal
Data at rest is encrypted. The USB key is the "physical password".
Without the key the machine will not boot (or falls back to passphrase prompt).

### Setup steps

```bash
# 1. During OS installation — choose "Encrypt disk with LUKS"
#    (Ubuntu installer: "Advanced features" → "Use LVM with LUKS encryption")
#    Set a recovery passphrase and store it safely offline.

# 2. After installation — generate a random key file on the USB drive
sudo dd if=/dev/urandom bs=512 count=8 of=/media/usb/panteon.key
sudo chmod 400 /media/usb/panteon.key

# 3. Add the key to LUKS (replace /dev/sda3 with your encrypted partition)
sudo cryptsetup luksAddKey /dev/sda3 /media/usb/panteon.key

# 4. Find partition UUID
sudo blkid /dev/sda3
# → UUID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

# 5. Find USB drive by-id or by-label (label your USB "PANTEON_KEY")
sudo e2label /dev/sdb1 PANTEON_KEY

# 6. Edit /etc/crypttab
# sda3_crypt  UUID=<partition-uuid>  /dev/disk/by-label/PANTEON_KEY:/panteon.key  luks

# 7. Regenerate initramfs
sudo update-initramfs -u
```

### Behaviour matrix

| USB present at boot | Result |
|---|---|
| Yes | Boots automatically, no prompt |
| No | Falls back to passphrase prompt |
| Stolen PC, no USB | Data inaccessible |
| Stolen PC + USB | Data accessible — keep them physically separate |

### Recovery passphrase
Store the LUKS recovery passphrase in a password manager or printed document in a safe location.
If both USB and passphrase are lost, data is unrecoverable.

---

## Part 2 — Network: panteon.local and Auto-Start Services

### 2.1 Static hostname (mDNS via Avahi)

```bash
# Set hostname
sudo hostnamectl set-hostname panteon

# Install Avahi mDNS daemon
sudo apt install avahi-daemon

# Enable and start
sudo systemctl enable avahi-daemon
sudo systemctl start avahi-daemon
```

After this, any device on the same LAN (Linux, macOS, Windows 10+) can reach
the machine at `panteon.local` — no router configuration, no static IP needed.

Works over:
- Direct LAN cable (with or without router)
- Home WiFi router
- Any combination of the above simultaneously

For direct cable without a router, also install a DHCP server so the client
laptop gets an IP address automatically:

```bash
sudo apt install dnsmasq

# /etc/dnsmasq.conf
# interface=eth0
# dhcp-range=192.168.88.100,192.168.88.200,12h
```

The mini-PC itself keeps a static IP on the LAN interface:

```yaml
# /etc/netplan/01-lan.yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: false
      addresses: [192.168.88.1/24]
```

### 2.2 Systemd service units

Each tool gets a systemd unit so it starts on boot and restarts on failure.

**Template** (`/etc/systemd/system/<tool>.service`):

```ini
[Unit]
Description=<Tool name>
After=network.target ollama.service

[Service]
User=panteon
WorkingDirectory=/home/panteon
ExecStart=/home/panteon/.venv/bin/<command>
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Service table**

| Service | Command | Default port |
|---|---|---|
| ollama | `ollama serve` | 11434 |
| vyrii | `vyrii ui --host 0.0.0.0 --port 4896` | 4896 |
| simargl | `simargl ui --host 0.0.0.0 --port 7861` | 7861 |
| svitovyd | `svitovyd ui --host 0.0.0.0 --port 7860` | 7860 |
| ttyd | see Part 3 | 7681 |
| recoll | see Part 3 | 8080 |

Enable all at once:

```bash
for svc in ollama vyrii simargl svitovyd ttyd recoll; do
    sudo systemctl enable $svc
    sudo systemctl start $svc
done
```

### 2.3 Access URLs (after setup)

```
http://panteon.local:4896    — Vyrii (LLM chat + tools)
http://panteon.local:7861    — Simargl (code retrieval)
http://panteon.local:7860    — Svitovyd (code map)
http://panteon.local:7681    — ttyd (web terminal)
http://panteon.local:8080    — Recoll (full-text search)
```

---

## Part 3 — ttyd and Recoll

### 3.1 ttyd — Web Terminal

ttyd exposes a terminal session in the browser over WebSocket. Useful for
running CLI commands on the headless mini-PC from any device without SSH client.

```bash
# Install
sudo apt install ttyd

# Or build from source (newer version):
# https://github.com/tsl0922/ttyd

# Manual test
ttyd --port 7681 bash
# → open http://panteon.local:7681
```

Systemd unit (`/etc/systemd/system/ttyd.service`):

```ini
[Unit]
Description=ttyd Web Terminal
After=network.target

[Service]
User=panteon
ExecStart=/usr/bin/ttyd --port 7681 --interface 0.0.0.0 bash
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Security note: ttyd exposes a shell to anyone on the LAN.
For a home/private network this is acceptable.
For a shared network, add basic auth: `ttyd --credential user:password`.

### 3.2 Recoll — Full-Text Search Server

Recoll indexes local files (PDFs, docs, text, email) and provides a web UI.

```bash
# Install Recoll + web interface
sudo apt install recoll python3-recoll

# Install recoll web UI (recollwebui)
pip install recollwebui
# or clone: https://github.com/koniu/recoll-webui

# Index your documents
recollindex -c ~/.recoll

# Run web server
python3 /path/to/recollwebui/webui-standalone.py \
    -p 8080 -a 0.0.0.0
```

Systemd unit (`/etc/systemd/system/recoll.service`):

```ini
[Unit]
Description=Recoll Web UI
After=network.target

[Service]
User=panteon
ExecStart=/home/panteon/.venv/bin/python3 /home/panteon/recollwebui/webui-standalone.py -p 8080 -a 0.0.0.0
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Schedule nightly re-index via cron:

```bash
# crontab -e
0 3 * * * recollindex -c ~/.recoll
```

---

## Part 4 — Setup Questionnaire

> Minimal 10-question intake form. Purpose: determine required services,
> suitable LLM models, and minimum hardware specifications for the user's use case.
> This is a template — the real questionnaire will be expanded.

---

### Q0. Choose a unique name for your device

This name becomes your device's network address: `<name>.local`

Rules:
- Lowercase letters, digits, hyphens only
- Must be unique across all PANTEON devices on your network
- Can be changed later via Syryn Bluetooth interface

Examples: `panteon-john`, `panteon-office`, `panteon-01`, `panteon-studio`

Your device name: _______________

*Sets: mDNS hostname, used in all service URLs and Bluetooth identity.*

---

### Q1. Primary use case (choose all that apply)

- [ ] Chat with local LLM (general assistant)
- [ ] Code navigation and search (simargl, svitovyd)
- [ ] Document indexing and full-text search (recoll)
- [ ] Web research and crawling (vyrii WebAsk/WebCrawl)
- [ ] Automated agent tasks (vyrii DeepAgent)
- [ ] Running scheduled background tasks
- [ ] Other: _______________

*Determines: which services to install and enable.*

---

### Q2. What languages do your documents and codebases use?

- [ ] English only
- [ ] Ukrainian / Russian
- [ ] Mixed European languages
- [ ] Code only (no natural language documents)
- [ ] Other: _______________

*Determines: LLM model selection (multilingual vs English-only), embedding model.*

---

### Q3. Approximate total size of files you want to index

- [ ] Under 1 GB (personal notes, small project)
- [ ] 1–10 GB (mid-size codebase or document collection)
- [ ] 10–50 GB (large codebase or research archive)
- [ ] Over 50 GB

*Determines: disk size requirements and indexing time expectations.*

---

### Q4. Do you need to process images or PDFs with text extraction?

- [ ] No — plain text and code only
- [ ] Yes — PDFs (text-based)
- [ ] Yes — scanned documents (need OCR)
- [ ] Yes — images with text (screenshots, diagrams)

*Determines: whether to install Tesseract OCR, poppler, additional recoll plugins.*

---

### Q5. How do you plan to connect to the mini-PC?

- [ ] Direct LAN cable from my laptop
- [ ] Home WiFi router (same network)
- [ ] Both (cable sometimes, WiFi other times)
- [ ] Over the internet (VPN / port forwarding)

*Determines: network setup — dnsmasq, avahi, VPN configuration.*

---

### Q6. How many people will use this machine simultaneously?

- [ ] Only me
- [ ] 2–3 people (family / small team)
- [ ] More than 3

*Determines: whether to configure vyrii multi-user / basic auth, resource limits per service.*

---

### Q7. What LLM response quality do you need?

- [ ] Fast answers are more important than quality (small model, 1–3B)
- [ ] Balance of speed and quality (medium model, 7–8B)
- [ ] Quality is more important, I can wait (large model, 13B+)
- [ ] I don't know yet

*Determines: which Ollama models to pre-download, VRAM/RAM requirements.*

---

### Q8. Do you need to run multiple LLM tasks in parallel?

- [ ] No — one task at a time is enough
- [ ] Sometimes — parallel team tab or scheduled agents
- [ ] Yes — always running background agents

*Determines: RAM requirements (each parallel model instance needs its own memory).*

---

### Q9. Hardware you already have or are considering (select closest)

- [ ] Mini-PC with 8 GB RAM, no dedicated GPU
- [ ] Mini-PC with 16 GB RAM, no dedicated GPU
- [ ] Mini-PC with dedicated GPU (4–6 GB VRAM)
- [ ] Mini-PC with dedicated GPU (8+ GB VRAM)
- [ ] Not decided yet

*Determines: feasible model sizes, batch_size limits, whether GPU acceleration is possible.*

---

### Q10. How important is data security for your use case?

- [ ] Low — personal hobby project, no sensitive data
- [ ] Medium — work documents, I prefer encryption but it is not critical
- [ ] High — confidential code / documents, encryption required
- [ ] Very high — need USB key boot lock + audit log

*Determines: whether to configure LUKS encryption, ttyd auth, avahi visibility restrictions.*

---

## Recommended configurations (derived from questionnaire)

| Profile | Services | Min RAM | Model size |
|---|---|---|---|
| Personal assistant | ollama + vyrii | 8 GB | 7B Q4 |
| Code navigator | ollama + vyrii + simargl + svitovyd | 16 GB | 7B Q4 |
| Document researcher | ollama + vyrii + recoll | 8 GB | 7B Q4 |
| Full stack | all services | 32 GB | 13B Q4 |
| Agent farm | all + parallel agents | 32 GB+ | multiple models |

---

*This document describes the UX and deployment design concept for the PANTEON image.
Actual installation scripts, Ansible playbooks, and image build pipeline are out of scope here.*
