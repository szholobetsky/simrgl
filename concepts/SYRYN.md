# SYRYN — Bluetooth Identity Beacon

> Named after **Сирин** (Syryn) — the Slavic mythological bird with a human face
> whose voice announces presence to those who are near.

---

## The problem it solves

A headless mini-PC running on a local network has no screen.
When two similar devices exist on the same LAN, a user standing next to one of them
has no way to tell from a browser which device belongs to them.

Syryn is a passive Bluetooth beacon. It announces the identity of the machine
to any phone or laptop within physical proximity. No network access is required —
the communication happens over Bluetooth only.

---

## Behaviour

Syryn runs as a background daemon. It makes the device permanently discoverable
and accepts any incoming Bluetooth connection without a PIN or pairing confirmation.
This is intentional: the information it exposes is not sensitive, and the headless
device has no screen or keyboard to participate in PIN-based pairing.

When a client connects and sends the text `STATUS`, Syryn responds with a short
plain-text block containing the device hostname, its mDNS address, and the current
IP addresses on all active network interfaces (LAN, WiFi, and any others present).
After the response, the connection may remain open or close — either is acceptable.

No other commands are recognised. Syryn does not execute processes, does not modify
system state, and does not expose any file system paths.

---

## What STATUS returns

A typical response looks like this:

```
hostname    : panteon-john
mdns        : panteon-john.local
eth0        : 192.168.88.1
wlan0       : 192.168.1.42
external_ip : 93.184.216.34
```

The field names are fixed. Interfaces that are down or have no IP are omitted.
External IP is fetched from a public HTTP endpoint (e.g. `api.ipify.org`) and
cached for a few minutes to avoid repeated outbound requests.

## No internal state

Syryn stores nothing. The hostname is read from the operating system at the moment
of each request via `socket.gethostname()`. The mDNS address is derived from it
directly as `{hostname}.local`. Network interface addresses are read from the OS
at request time via `psutil.net_if_addrs()`.

This means Syryn is always consistent with the actual system state. If the machine
is renamed with `hostnamectl set-hostname panteon-studio`, the very next STATUS
response will reflect the new name — no restart, no config file update required.

The only exception is the external IP, which is cached for a few minutes to avoid
hammering a public endpoint on every connection. Everything else is live.

## Bluetooth device name

When a phone scans for nearby Bluetooth devices, each device announces a human-readable
name. By default `bluetoothd` on Linux takes this name from the system hostname.
Syryn relies on this behaviour without any additional configuration.

The result is a single source of truth across all three identity channels:

```
hostnamectl set-hostname panteon-john

→ browser reaches it at   panteon-john.local
→ phone sees it in BT list as   panteon-john
→ STATUS responds with   hostname: panteon-john
```

Nothing is hardcoded in Syryn. If the machine is renamed, all three update
automatically — the BT name on the next phone scan, the mDNS address immediately,
and the STATUS response on the next connection.

---

## Why this information is not sensitive

The hostname and local IP addresses of a device connected to a LAN or WiFi network
are already visible to every other device on that network via ARP tables and mDNS
announcements. Syryn exposes nothing that is not already broadcast at the network layer.
The only slightly new piece of information is the external IP, which is the same
for every device behind the same router.

The value Syryn adds is not secrecy — it is **physical proximity as identity**.
A user standing next to their machine can confirm it is theirs before opening a browser.

---

## Transport

Classic Bluetooth with the Serial Port Profile (RFCOMM). This is the simplest
possible Bluetooth communication channel: both sides exchange plain text lines,
the same way a serial terminal works. Any BT Serial terminal application on
Android or iOS can connect to Syryn without installing anything special.

The device is set to `NoInputNoOutput` capability in the Bluetooth agent,
which activates the Just Works association model — pairing completes automatically
without user interaction on either side.

---

## Python package

Syryn is distributed as a standard Python package installable with `pip install syryn`.
After installation a single command-line entry point `syryn` is available.

Running `syryn` is the only thing it does: open a Bluetooth RFCOMM channel,
wait for a connection, respond to STATUS, and keep waiting. There are no subcommands
because there is only one mode of operation.

For automatic startup on boot, the systemd unit simply calls `syryn` as its
`ExecStart`. No arguments needed.

The only non-Python dependency is the system Bluetooth stack (`libbluetooth-dev`
on Debian/Ubuntu) required by the underlying RFCOMM library.

Dependencies are minimal: a Bluetooth RFCOMM library, `psutil` for reading
network interface addresses, and `requests` for the external IP lookup.

---

## Position in the tool ecosystem

| Tool     | Slavic deity | Role                        |
|----------|--------------|-----------------------------|
| simargl  | Симаргл      | Semantic code retrieval     |
| vyrii    | Вирій        | LLM UI and tools            |
| svitovyd | Світовид     | Code structure map          |
| yasna    | Ясна         | Memory and context system   |
| syryn    | Сирин        | Bluetooth identity beacon   |

Syryn is the voice of the machine — heard only by those who are near.
