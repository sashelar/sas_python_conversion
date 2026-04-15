import os
from pyspark.sql import SparkSession

# Source: Local temp where files are extracted
local_extract_path = "/tmp/risk_engine_extracted"

# Target: ADLS path
from companyframework.functions import commonfunctions as cf
ref_location = cf.get_base_directory("ref")
adls_target = f"{ref_location}/Risk_Engine_Programs"

print("="*70)
print("COPYING FILES TO ADLS USING SPARK")
print("="*70)

print(f"\n📂 Source: {local_extract_path}")
print(f"📂 Target: {adls_target}")

# Get all files recursively
print(f"\n1️⃣ Scanning files...")

all_files = []
for root, dirs, files in os.walk(local_extract_path):
    for file in files:
        local_file_path = os.path.join(root, file)
        # Calculate relative path
        relative_path = os.path.relpath(local_file_path, local_extract_path)
        all_files.append((local_file_path, relative_path))

print(f"   Found {len(all_files)} files to copy")

# Copy each file using Spark
print(f"\n2️⃣ Copying files to ADLS...")

copied_count = 0
failed_count = 0

for local_path, relative_path in all_files:
    try:
        # Read file as binary
        with open(local_path, 'rb') as f:
            file_content = f.read()
        
        # Target path in ADLS
        adls_file_path = f"{adls_target}/{relative_path}"
        
        # Create parent directory path
        adls_dir = os.path.dirname(adls_file_path)
        
        # Write using Spark
        # Create a DataFrame with the binary content
        df = spark.createDataFrame([(adls_file_path, file_content)], ["path", "content"])
        
        # Write as binary file
        df.coalesce(1).write.mode("overwrite").format("binaryFile").save(adls_file_path)
        
        copied_count += 1
        if copied_count % 10 == 0:
            print(f"   Progress: {copied_count}/{len(all_files)} files...")
        
    except Exception as e:
        print(f"   ❌ Failed: {relative_path} - {e}")
        failed_count += 1

print(f"\n✅ Copy complete!")
print(f"   Copied: {copied_count} files")
print(f"   Failed: {failed_count} files")

print("\n" + "="*70)
