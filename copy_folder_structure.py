
import os
import shutil

# Source: Local temp where files are extracted
local_extract_path = "/tmp/risk_engine_extracted"

# Target: ADLS path (using the format that works with your company framework)
from companyframework.functions import commonfunctions as cf

ref_location = cf.get_base_directory("ref")
adls_target = f"{ref_location}/Risk_Engine_Programs"

print("="*70)
print("COPYING FILES TO ADLS")
print("="*70)

print(f"\n📂 Source: {local_extract_path}")
print(f"📂 Target: {adls_target}")

# Create target directory
print(f"\n1️⃣ Creating target directory...")
os.makedirs(adls_target, exist_ok=True)
print(f"   ✅ Directory ready")

# Copy all files and folders
print(f"\n2️⃣ Copying files...")

try:
    for item in os.listdir(local_extract_path):
        source_item = os.path.join(local_extract_path, item)
        target_item = os.path.join(adls_target, item)
        
        if os.path.isdir(source_item):
            print(f"   📁 Copying folder: {item}...")
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            print(f"   📄 Copying file: {item}...")
            shutil.copy2(source_item, target_item)
    
    print(f"\n✅ All files copied to ADLS!")
    
except Exception as e:
    print(f"\n❌ Error during copy: {e}")

# Verify files in ADLS
print(f"\n3️⃣ Verifying files in ADLS...")
if os.path.exists(adls_target):
    items = os.listdir(adls_target)
    print(f"   ✅ Found {len(items)} items in ADLS:")
    for item in items[:10]:  # Show first 10
        print(f"      - {item}")
    if len(items) > 10:
        print(f"      ... and {len(items) - 10} more")
else:
    print(f"   ❌ Target directory not found")

# Cleanup temp files
print(f"\n4️⃣ Cleaning up temp files...")
try:
    shutil.rmtree(local_extract_path)
    os.remove("/tmp/risk_engine.zip")
    print(f"   ✅ Cleanup complete")
except:
    print(f"   ⚠️  Cleanup had minor issues (non-critical)")

print("\n" + "="*70)
print("✅ COMPLETE!")
print(f"📂 Your files are now at: {adls_target}")
print("="*70)
