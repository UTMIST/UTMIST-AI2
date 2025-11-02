#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

# --- Optional: Load secrets from .env file ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("💡 Tip: install python-dotenv for .env support (pip install python-dotenv)")

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="RL Tournament Battle Pipeline (Local Version for Windows)")
parser.add_argument("--username1", required=True, help="GitHub username of first participant (fork owner)")
parser.add_argument("--username2", required=True, help="GitHub username of second participant (fork owner)")
args = parser.parse_args()

u1 = args.username1
u2 = args.username2

# --- Step 1: Validate Inputs ---
if u1 == u2:
    print("❌ Error: Usernames must be different.")
    sys.exit(1)
print(f"✅ Usernames validated: {u1} vs {u2}")

# --- Step 2: Check environment variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️ Warning: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables.")
    print("Please set them in your environment or a .env file.")
else:
    print("✅ Supabase environment variables loaded.")

# --- Step 3: Install dependencies (if install.bat or install.sh exists) ---
# if Path("install.bat").exists():
#     print("🔧 Running install.bat...")
#     subprocess.run(["install.bat"], shell=True, check=True)
# elif Path("install.sh").exists():
#     print("🔧 Running install.sh via bash...")
#     subprocess.run(["bash", "install.sh"], shell=True, check=True)
# else:
#     print("⚠️ No install script found, skipping dependency installation.")

# --- Step 4: Validate Participants via API ---
print("🧩 Checking participant validation via api.py...")

server_path = Path("server").resolve()
sys.path.insert(0, str(server_path))

try:
    import api
except ImportError as e:
    print(f"❌ Could not import api.py from {server_path}. Error: {e}")
    sys.exit(1)

try:
    is_valid = api.validate_battle(u1, u2)
except Exception as e:
    print(f"❌ Error calling validate_battle: {e}")
    sys.exit(1)

if not is_valid:
    print(f"❌ Validation failed: One or both users have not passed validation (u1={u1}, u2={u2})")
    sys.exit(1)
else:
    print(f"✅ Validation passed: Both users have passed validation (u1={u1}, u2={u2})")

# --- Step 5: Clone Forks ---
repo_name = Path.cwd().name
print("🌀 Cloning participant forks...")

def clone_repo(username, dest):
    if Path(dest).exists():
        shutil.rmtree(dest)
    url = f"https://github.com/{username}/{repo_name}.git"
    print(f"→ Cloning {url} into {dest}")
    subprocess.run(["git", "clone", url, dest], shell=True, check=True)

clone_repo(u1, "fork1")
clone_repo(u2, "fork2")

# --- Step 6: Copy agents to main repo ---
print("📦 Copying agents...")
os.makedirs(f"agents/{u1}", exist_ok=True)
os.makedirs(f"agents/{u2}", exist_ok=True)

try:
    shutil.copy("fork1/user/my_agent.py", f"agents/{u1}/my_agent.py")
    shutil.copy("fork2/user/my_agent.py", f"agents/{u2}/my_agent.py")
except FileNotFoundError as e:
    print(f"❌ Missing agent file: {e}")
    sys.exit(1)

print("✅ Agents copied successfully.")
for root, _, files in os.walk("agents"):
    for f in files:
        print("  -", Path(root) / f)

# --- Step 7: Run Battle ---
print("⚔️ Running battle...")

os.environ["AGENT1_PATH"] = f"agents/{u1}/my_agent.py"
os.environ["AGENT2_PATH"] = f"agents/{u2}/my_agent.py"

try:
    subprocess.run(["pytest", "-s", "user/battle.py"], check=True, shell=True)
    print("🏁 Battle completed successfully!")
except subprocess.CalledProcessError:
    print("❌ Battle failed. Check logs above.")
    sys.exit(1)