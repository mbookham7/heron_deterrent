# Coral Dev Board OS Flashing Guide

A comprehensive step-by-step guide to flash a new operating system onto your Google Coral Dev Board.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Download Required Files](#download-required-files)
3. [Install Flashing Tool](#install-flashing-tool)
4. [Prepare the Coral Dev Board](#prepare-the-coral-dev-board)
5. [Flash the OS](#flash-the-os)
6. [First Boot Setup](#first-boot-setup)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Required

- **Coral Dev Board** (any version)
- **USB-C cable** (for power and data)
- **USB-C power supply** (5V/3A recommended)
- **Host computer** (Linux, macOS, or Windows)
- **USB-C to USB-A adapter** (if your computer doesn't have USB-C)
- Optional: **Serial console cable** (USB to TTL UART) for debugging

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10/11
- **Python**: Version 3.6 or higher
- **USB Port**: Working USB port with data transfer capability
- **Internet Connection**: For downloading files
- **Disk Space**: At least 2GB free space

---

## Download Required Files

### 1. Download Mendel Linux Image

The Coral Dev Board runs Mendel Linux, a derivative of Debian.

```bash
# Create a working directory
mkdir ~/coral-flash
cd ~/coral-flash

# Download the latest Mendel Linux image
# Visit: https://coral.ai/software/#mendel-linux
# Or use wget (replace URL with latest version):
wget https://github.com/google-coral/mendel-enterprise/releases/download/enterprise-eagle-5.3/mendel-enterprise-eagle-flashcard-20211117215217.zip

# Extract the image
unzip mendel-enterprise-eagle-flashcard-*.zip
```

**Current Versions (as of 2024):**
- **Enterprise Eagle**: Latest stable release
- **Chef**: Older stable release (still supported)
- **Day**: Legacy release

### 2. Verify Download

```bash
# Check the extracted files
ls -lh

# You should see:
# - flash.sh (flashing script)
# - flashcard_arm64.img (boot image)
# - partition-table.img (partition layout)
# - Various .img files for system partitions
```

---

## Install Flashing Tool

### For Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt-get update

# Install required dependencies
sudo apt-get install -y python3 python3-pip udev fastboot adb

# Install pip packages
pip3 install --user mendel-development-tool
```

### For macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3
brew install android-platform-tools  # Provides fastboot and adb

# Install MDT (Mendel Development Tool)
pip3 install mendel-development-tool
```

### For Windows

```bash
# Install Python 3.6+ from python.org
# Download and install: https://www.python.org/downloads/

# Open PowerShell as Administrator and run:
pip install mendel-development-tool

# Install Android Platform Tools (for fastboot)
# Download from: https://developer.android.com/studio/releases/platform-tools
# Extract to C:\platform-tools
# Add C:\platform-tools to your PATH environment variable
```

---

## Prepare the Coral Dev Board

### 1. Locate the Boot Pins

The Coral Dev Board has two small pins labeled **Pin 1** and **Pin 2** near the USB-C port.

```
    USB-C Port
        ||
    [1] [2]  <- Boot pins
```

### 2. Set Boot Mode

**To enter flashing mode:**

1. **Disconnect power** from the Coral Dev Board
2. **Connect a jumper or short wire** between **Pin 1 and Pin 2**
3. **Keep the pins shorted** while proceeding to next step

### 3. Connect to Computer

```bash
# With pins still shorted:
# 1. Connect USB-C cable from Coral to your computer
# 2. Connect power to the Coral Dev Board
# 3. Wait 5 seconds
# 4. Remove the jumper/short from the pins

# The board should now be in fastboot mode
```

### 4. Verify Fastboot Mode

```bash
# Check if device is detected
fastboot devices

# You should see output like:
# 1b0741d749c90c14    fastboot

# If nothing appears, troubleshoot:
# - Try a different USB cable
# - Try a different USB port
# - Repeat the boot pin shorting procedure
```

**Linux users may need sudo:**
```bash
sudo fastboot devices
```

---

## Flash the OS

### Method 1: Using Flash Script (Recommended)

```bash
# Navigate to the extracted Mendel directory
cd ~/coral-flash/mendel-*

# Make the flash script executable
chmod +x flash.sh

# Run the flashing script
sudo ./flash.sh

# The script will:
# 1. Verify fastboot connection
# 2. Flash bootloader
# 3. Flash partition table
# 4. Flash system partitions
# 5. Flash boot image
# 6. Reboot the device

# This process takes 5-10 minutes
```

**Expected Output:**
```
Flashing bootloader...
Sending 'bootloader' (1024 KB)                    OKAY [  0.123s]
Writing 'bootloader'                              OKAY [  0.456s]
Flashing partition table...
Sending 'partition' (33 KB)                       OKAY [  0.012s]
Writing 'partition'                               OKAY [  0.089s]
...
Rebooting...
Finished. Total time: 8.234s
```

### Method 2: Manual Flashing

If the script fails, flash manually:

```bash
cd ~/coral-flash/mendel-*

# Flash bootloader
sudo fastboot flash bootloader u-boot.imx
sudo fastboot reboot-bootloader
sleep 5

# Flash partition table
sudo fastboot flash gpt partition-table.img

# Flash boot partition
sudo fastboot flash boot boot_arm64.img

# Flash rootfs partitions
sudo fastboot flash rootfs_0 rootfs_0.img
sudo fastboot flash rootfs_1 rootfs_1.img

# Flash userdata
sudo fastboot flash userdata_1 userdata_1.img

# Reboot
sudo fastboot reboot
```

### 3. Wait for First Boot

```bash
# The board will automatically reboot after flashing
# First boot takes 2-3 minutes as it:
# - Initializes filesystems
# - Generates SSH keys
# - Sets up services
# - Configures networking

# Wait for the green LED to stop blinking rapidly
```

---

## First Boot Setup

### 1. Connect via MDT (Mendel Development Tool)

```bash
# Wait 3 minutes after first boot, then:
mdt devices

# You should see:
# coral-dev-board-1    (192.168.100.2)

# Connect to the board
mdt shell

# You're now connected via serial console
```

### 2. Initial Configuration

```bash
# Once connected, you'll be logged in as 'mendel' user

# Set root password (optional but recommended)
sudo passwd root

# Update package lists
sudo apt-get update

# Upgrade installed packages
sudo apt-get upgrade -y

# Set timezone
sudo timedatectl set-timezone America/New_York

# Set hostname (optional)
sudo hostnamectl set-hostname my-coral-board
```

### 3. Configure Network

**For Wi-Fi:**
```bash
# Scan for networks
nmcli device wifi list

# Connect to Wi-Fi
nmcli device wifi connect "YOUR_SSID" password "YOUR_PASSWORD"

# Verify connection
nmcli connection show
ip addr show
```

**For Ethernet:**
```bash
# Ethernet should work automatically via DHCP
# Check connection:
ip addr show eth0

# Test internet connectivity
ping -c 4 google.com
```

### 4. Enable SSH Access

```bash
# SSH is enabled by default
# Find your board's IP address
ip addr show

# From your host computer, connect:
ssh mendel@192.168.1.XXX
# Default password: mendel

# Change default password immediately
passwd
```

### 5. Verify Edge TPU

```bash
# Check if Edge TPU is detected
lspci -nn | grep 089a

# You should see:
# 01:00.0 System peripheral: Global Unichip Corp. Coral Edge TPU [1ac1:089a]

# Install Edge TPU runtime (if not already installed)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libedgetpu1-std python3-pycoral
```

---

## Troubleshooting

### Device Not Detected in Fastboot

**Problem:** `fastboot devices` shows nothing

**Solutions:**

1. **Check USB cable**: Use a data-capable USB-C cable (not charge-only)
2. **Try different port**: Use a USB 2.0 port or hub
3. **Repeat boot procedure**: 
   ```bash
   # Power off completely
   # Short pins 1-2
   # Power on while pins are shorted
   # Wait 5 seconds
   # Remove short
   ```
4. **Check drivers (Windows)**:
   - Install Google USB Driver
   - Update device driver in Device Manager
5. **Linux permissions**:
   ```bash
   # Add udev rules
   sudo nano /etc/udev/rules.d/51-android.rules
   
   # Add this line:
   SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", ATTR{idProduct}=="4ee7", MODE="0666", GROUP="plugdev"
   
   # Reload rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

### Flash Script Fails

**Problem:** Flash script exits with errors

**Solutions:**

1. **Use manual flashing** method (see Method 2 above)
2. **Check available disk space**:
   ```bash
   df -h
   ```
3. **Verify image integrity**:
   ```bash
   md5sum *.img
   # Compare with published checksums
   ```
4. **Try older Mendel version**: Sometimes newer versions have issues

### Board Won't Boot After Flashing

**Problem:** No LED activity or stuck at boot

**Solutions:**

1. **Re-flash bootloader**:
   ```bash
   # Enter fastboot mode again
   sudo fastboot flash bootloader u-boot.imx
   sudo fastboot reboot-bootloader
   ```
2. **Factory reset**:
   ```bash
   sudo fastboot erase userdata
   sudo fastboot erase cache
   sudo fastboot reboot
   ```
3. **Serial console debugging**:
   - Connect USB-TTL serial cable
   - Use screen/minicom to view boot logs
   ```bash
   sudo screen /dev/ttyUSB0 115200
   ```

### Cannot Connect via MDT

**Problem:** `mdt devices` shows nothing

**Solutions:**

1. **Wait longer**: First boot can take 5+ minutes
2. **Check USB connection**: Board must be connected via USB-C
3. **Check network**: 
   ```bash
   # Board should appear as USB ethernet device
   ip addr show
   # Look for usb0 interface with 192.168.100.x
   ```
4. **Reconnect**:
   ```bash
   # Disconnect and reconnect USB cable
   # Wait 30 seconds
   mdt devices
   ```
5. **Use SSH instead**: If you know the IP address
   ```bash
   ssh mendel@<IP_ADDRESS>
   ```

### Wi-Fi Not Working

**Problem:** Cannot connect to Wi-Fi network

**Solutions:**

1. **Check supported bands**: Coral supports 2.4GHz and 5GHz
2. **Use Ethernet first**: Update system, then try Wi-Fi
3. **Check network manager**:
   ```bash
   sudo systemctl status NetworkManager
   sudo systemctl restart NetworkManager
   ```
4. **Manual configuration**:
   ```bash
   # Edit wpa_supplicant config
   sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
   
   # Add:
   network={
       ssid="YourNetworkName"
       psk="YourPassword"
   }
   ```

### Edge TPU Not Detected

**Problem:** `lspci` doesn't show Edge TPU

**Solutions:**

1. **Check PCIe connection**: Edge TPU uses M.2 socket
2. **Re-flash**: Edge TPU drivers are in base image
3. **Check kernel logs**:
   ```bash
   dmesg | grep apex
   ```
4. **Reinstall runtime**:
   ```bash
   sudo apt-get install --reinstall libedgetpu1-std
   ```

---

## Additional Resources

### Official Documentation
- **Coral Dev Board Guide**: https://coral.ai/docs/dev-board/get-started/
- **Mendel Software**: https://coral.ai/software/#mendel-linux
- **GitHub Repository**: https://github.com/google-coral/mendel

### Community Resources
- **Coral Forum**: https://coral.ai/community/
- **Stack Overflow**: Tag `google-coral`
- **Reddit**: r/GoogleCoral

### Useful Commands

```bash
# Check system info
cat /etc/os-release
uname -a

# Check disk usage
df -h

# Check memory
free -h

# Monitor system
htop

# Check running services
systemctl list-units --type=service

# View system logs
journalctl -xe

# Test Edge TPU with demo
git clone https://github.com/google-coral/pycoral.git
cd pycoral
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```

---

## Next Steps

After successfully flashing and setting up your Coral Dev Board:

1. **Install your application** (e.g., Heron Deterrent System)
2. **Configure auto-start** for your application
3. **Set up remote access** for maintenance
4. **Configure backups** for important data
5. **Monitor system health** and logs

**Congratulations!** Your Coral Dev Board is now ready to use! 🎉