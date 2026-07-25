# Network Troubleshooting: Connecting a Headless Mini PC

How to find and connect to a headless mini PC (e.g. panteon) from a laptop.

---

## Scenario 1: Mini PC connected to the same Wi-Fi router

This is the easiest case. Both devices are in the same subnet — no configuration needed.

```bash
ping panteon.local       # mDNS via Avahi (Linux) or Bonjour (macOS)
ssh user@panteon.local
```

If `panteon.local` doesn't resolve, find the IP via the router admin panel (`192.168.0.1` or `192.168.1.1`) and look for `panteon` in the DHCP lease table. Then:

```bash
ssh user@192.168.0.XXX
```

---

## Scenario 2: Direct LAN cable — laptop to mini PC

### Why it doesn't work out of the box

Without a router, there is no DHCP server. Both devices fall back to APIPA (`169.254.x.x`) but pick random addresses independently — they end up in different effective subnets and can't discover each other. mDNS (`panteon.local`) also fails because the addresses don't match.

---

### Option A: ICS (Internet Connection Sharing) — simplest, recommended

The laptop becomes a DHCP server on the Ethernet port and also shares its internet connection with the mini PC.

#### Windows 11

1. Press `Win+R` → type `ncpa.cpl` → Enter
2. Right-click **Wi-Fi** → **Properties** → **Sharing** tab
3. Check **"Allow other network users to connect through this computer's Internet connection"**
4. From the dropdown, select **Ethernet**
5. Click OK

The Ethernet adapter gets `192.168.137.1` and the mini PC receives `192.168.137.x` via DHCP automatically.

Verify:
```
ping panteon.local
ssh user@panteon.local
```

If the mini PC was already connected before ICS was enabled, replug the LAN cable on the mini PC side to trigger a new DHCP request.

#### macOS

1. System Settings → General → Sharing → **Internet Sharing**
2. Share from: **Wi-Fi**
3. To computers using: **Ethernet** (or Thunderbolt Ethernet / USB Ethernet)
4. Toggle Internet Sharing **on**

The mini PC gets `192.168.2.x`. Same verification as above.

#### Linux (laptop)

Install and configure `dnsmasq` as a lightweight DHCP server:

```bash
sudo apt install dnsmasq       # or dnf install dnsmasq

# Set static IP on Ethernet
sudo ip addr add 192.168.137.1/24 dev eth0
sudo ip link set eth0 up

# Start DHCP server on Ethernet only
sudo dnsmasq --interface=eth0 \
             --dhcp-range=192.168.137.100,192.168.137.200,12h \
             --no-daemon
```

Enable NAT to share internet:
```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
```

---

### Option B: Static IPs — no internet sharing needed

Use this when you don't want to share internet, just need SSH access.

Requires physical access to the mini PC at least once to configure its IP.

**On the mini PC (Xubuntu / any Linux):**

Edit `/etc/netplan/01-netcfg.yaml` (or equivalent):
```yaml
network:
  ethernets:
    eth0:
      addresses: [192.168.100.2/24]
  version: 2
```
```bash
sudo netplan apply
```

Or via NetworkManager:
```bash
nmcli con add type ethernet ifname eth0 ip4 192.168.100.2/24
nmcli con up ethernet-eth0
```

**On the laptop:**

Windows:
```
netsh interface ip set address "Ethernet" static 192.168.100.1 255.255.255.0
```

macOS:
```
System Settings → Network → Ethernet → Details → TCP/IP → Manual
IP: 192.168.100.1 / Mask: 255.255.255.0
```

Linux:
```bash
sudo ip addr add 192.168.100.1/24 dev eth0
sudo ip link set eth0 up
```

Verify:
```bash
ping 192.168.100.2
ssh user@192.168.100.2
```

`panteon.local` will also work once both are in the same subnet and Avahi is running on the mini PC.

---

## Diagnosing the connection

### Check physical link (Windows)
```powershell
Get-NetAdapter | Select-Object Name, Status, LinkSpeed
```
The Ethernet adapter must show `Up` and a link speed (e.g. `1 Gbps`). If `Disconnected` — check the cable or the port on the mini PC.

### Check IP on Ethernet (Windows)
```powershell
Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4
```
- `169.254.x.x` → APIPA, no DHCP server found
- `192.168.137.x` → ICS is active
- `192.168.100.x` → static IP configured

### Find the mini PC's IP after ICS
```powershell
arp -a | Select-String "192.168.137"
```
Or scan:
```powershell
2..254 | ForEach-Object { ping -n 1 -w 100 "192.168.137.$_" > $null }
arp -a | Select-String "192.168.137"
```

### Check Avahi on mini PC
```bash
systemctl status avahi-daemon
avahi-resolve -n panteon.local
```

---

## Quick reference

| Situation | Solution |
|---|---|
| Both on same Wi-Fi router | Just use `panteon.local` or check router DHCP |
| Direct cable, Windows laptop | Enable ICS: Wi-Fi → Ethernet |
| Direct cable, macOS laptop | Enable Internet Sharing: Wi-Fi → Ethernet |
| Direct cable, Linux laptop | dnsmasq + iptables NAT, or static IPs |
| No internet sharing needed | Static IPs on both sides |
| IP unknown, can't ping | Replug cable on mini PC to force DHCP renew |
| `panteon.local` not resolving | Check `avahi-daemon` on mini PC; check both in same subnet |
