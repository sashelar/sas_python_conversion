import sys
import os
from datetime import datetime
from companyframework.functions import commonfunctions as cf

# =============================================================================
# CREATE PERIOD PROGRAM - PySpark Conversion
# =============================================================================

# Excel file name
THISRUN_EXCEL = "DaMS-x1sx"

# Root paths
THISRUN_IN_ROOT_PATH = "SFSHNRR/input/Risk Engine"
THISRUN_IN_SYSTEM_PATH = f"{THISRUN_IN_ROOT_PATH}/SASEnvironments"

THISRUN_OUT_ROOT_PATH = "FSHNRR/output/Risk Engine"
THISRUN_OUT_SYSTEM_PATH = f"{THISRUN_OUT_ROOT_PATH}/SAS_Environments"

THISRUN_PRG_ROOT_PATH = "FSHNRR/sasprograms/Risk_Engines"
THISRUN_PRG_SYSTEM_PATH = f"{THISRUN_PRG_ROOT_PATH}/SAS_Environments"

# Get period dates from prompts (assuming you have the prompts object from earlier)
# If prompts are submitted, use those values; otherwise use defaults
if 'prompts' in globals() and prompts.submitted:
    THISRUN_PERIOD_TO_CREATE_date = prompts.values['THISRUN_PERIOD_TO_CREATE_DATE']
    THISRUN_PERIOD_TO_COPY_date = prompts.values['THISRUN_PERIOD_TO_COPY_DATE']
    THISRUN_ENVIRONMENT_ID = prompts.values['THISRUN_ENVIRONMENT_ID']
    THISRUN_ENVIRONMENT_ID_TO_COPY = prompts.values['THISRUN_ENVIRONMENT_ID_TO_COPY']
else:
    # Default values if prompts not available
    THISRUN_PERIOD_TO_CREATE_date = datetime.now()
    THISRUN_PERIOD_TO_COPY_date = datetime.now()
    THISRUN_ENVIRONMENT_ID = "BELINS_ENV_DEV"
    THISRUN_ENVIRONMENT_ID_TO_COPY = "BELINS_ENV_DEV"

# Format dates as YYYYMMDD (equivalent to yymmddn8. format in SAS)
THISRUN_PERIOD_TO_CREATE = THISRUN_PERIOD_TO_CREATE_date.strftime('%Y%m%d')
THISRUN_PERIOD_TO_COPY = THISRUN_PERIOD_TO_COPY_date.strftime('%Y%m%d')

print(f"THISRUN_PERIOD_TO_CREATE: {THISRUN_PERIOD_TO_CREATE}")
print(f"THISRUN_PERIOD_TO_COPY: {THISRUN_PERIOD_TO_COPY}")

# Build period paths
THISRUN_IN_PERIOD_TO_CREATE = f"{THISRUN_IN_SYSTEM_PATH}/{THISRUN_ENVIRONMENT_ID}/{THISRUN_PERIOD_TO_CREATE}"
THISRUN_OUT_PERIOD_TO_CREATE = f"{THISRUN_OUT_SYSTEM_PATH}/{THISRUN_ENVIRONMENT_ID}/{THISRUN_PERIOD_TO_CREATE}"
THISRUN_PRG_PERIOD_TO_CREATE = f"{THISRUN_PRG_SYSTEM_PATH}/{THISRUN_ENVIRONMENT_ID}/{THISRUN_PERIOD_TO_CREATE}"

THISRUN_IN_PERIOD_TO_COPY = f"{THISRUN_IN_SYSTEM_PATH}/{THISRUN_ENVIRONMENT_ID_TO_COPY}/{THISRUN_PERIOD_TO_COPY}"
THISRUN_OUT_PERIOD_TO_COPY = f"{THISRUN_OUT_SYSTEM_PATH}/{THISRUN_ENVIRONMENT_ID_TO_COPY}/{THISRUN_PERIOD_TO_COPY}"
THISRUN_PRG_PERIOD_TO_COPY = f"{THISRUN_PRG_SYSTEM_PATH}/{THISRUN_ENVIRONMENT_ID_TO_COPY}/{THISRUN_PERIOD_TO_COPY}"

THISRUN_PRG_PERIOD_PATH = THISRUN_PRG_PERIOD_TO_COPY

print(f"\nPeriod Paths (CREATE):")
print(f"  IN:  {THISRUN_IN_PERIOD_TO_CREATE}")
print(f"  OUT: {THISRUN_OUT_PERIOD_TO_CREATE}")
print(f"  PRG: {THISRUN_PRG_PERIOD_TO_CREATE}")

print(f"\nPeriod Paths (COPY FROM):")
print(f"  IN:  {THISRUN_IN_PERIOD_TO_COPY}")
print(f"  OUT: {THISRUN_OUT_PERIOD_TO_COPY}")
print(f"  PRG: {THISRUN_PRG_PERIOD_TO_COPY}")

# =============================================================================
# Add macro path to sys.path (equivalent to %add_sasautos)
# =============================================================================

def add_sasautos():
    """Add the macro path to sys.path (equivalent to SAS SASAUTOS)"""
    # Get existing sys.path (equivalent to existing_sasautos)
    existing_sasautos = sys.path.copy()
    
    print(f"\nExisting sys.path entries: {len(existing_sasautos)}")
    
    # Define REPLYSAS path
    replysas_path = f"{THISRUN_PRG_PERIOD_PATH}/programs/99_macros"
    
    # Check if REPLYSAS not already in sys.path (equivalent to %index check)
    if replysas_path not in existing_sasautos:
        # Add to sys.path (equivalent to options sasautos=)
        sys.path.insert(0, replysas_path)
        print(f"✓ Added REPLYSAS to sys.path: {replysas_path}")
    else:
        print(f"ℹ REPLYSAS already in sys.path: {replysas_path}")

# Call add_sasautos
add_sasautos()

# =============================================================================
# Set process flow and process ID
# =============================================================================

THISRUN_PROCESS_FLOW_ID = "PLATFORM_MANAGEMENT"
THISRUN_PROCESS_ID = "00_CREATE_PERIOD"

# =============================================================================
# Include/Execute: check_execution_process_byuser
# =============================================================================

check_execution_script = f"{THISRUN_PRG_PERIOD_PATH}/programs/97_platform_management/check_execution_process_byuser.py"

if os.path.exists(check_execution_script):
    print(f"\n📝 Executing: check_execution_process_byuser.py")
    exec(open(check_execution_script).read())
else:
    print(f"⚠️  Script not found: {check_execution_script}")

# =============================================================================
# Include/Execute: 00_Create_Period
# =============================================================================

create_period_script = f"{THISRUN_PRG_PERIOD_PATH}/programs/00_batches/00_Create_Period.py"

if os.path.exists(create_period_script):
    print(f"\n📝 Executing: 00_Create_Period.py")
    exec(open(create_period_script).read())
else:
    print(f"⚠️  Script not found: {create_period_script}")

# =============================================================================
# Set reporting date and period paths
# =============================================================================

THISRUN_REPORTING_DATE_NUM = THISRUN_PERIOD_TO_CREATE
THISRUN_IN_PERIOD_PATH = THISRUN_IN_PERIOD_TO_CREATE
THISRUN_OUT_PERIOD_PATH = THISRUN_OUT_PERIOD_TO_CREATE
THISRUN_PRG_PERIOD_PATH = THISRUN_PRG_PERIOD_TO_CREATE

print(f"\n📊 Reporting Configuration:")
print(f"  THISRUN_REPORTING_DATE_NUM: {THISRUN_REPORTING_DATE_NUM}")
print(f"  THISRUN_IN_PERIOD_PATH:     {THISRUN_IN_PERIOD_PATH}")
print(f"  THISRUN_OUT_PERIOD_PATH:    {THISRUN_OUT_PERIOD_PATH}")
print(f"  THISRUN_PRG_PERIOD_PATH:    {THISRUN_PRG_PERIOD_PATH}")

# =============================================================================
# Set configuration ID
# =============================================================================

THISRUN_CONFIG_ID = "CONFIG_DEMO"

# =============================================================================
# Include/Execute: 01_Create_Configuration
# =============================================================================

create_config_script = f"{THISRUN_PRG_PERIOD_PATH}/programs/00_batches/01_Create_Configuration.py"

if os.path.exists(create_config_script):
    print(f"\n📝 Executing: 01_Create_Configuration.py")
    exec(open(create_config_script).read())
else:
    print(f"⚠️  Script not found: {create_config_script}")

# =============================================================================
# Set context ID
# =============================================================================

THISRUN_CONTEXT_ID = "CALC_DEMO"

# =============================================================================
# Include/Execute: 02_Create_Context
# =============================================================================

create_context_script = f"{THISRUN_PRG_PERIOD_PATH}/programs/00_batches/02_Create_Context.py"

if os.path.exists(create_context_script):
    print(f"\n📝 Executing: 02_Create_Context.py")
    exec(open(create_context_script).read())
else:
    print(f"⚠️  Script not found: {create_context_script}")

print("\n" + "="*70)
print("✓ CREATE PERIOD PROGRAM COMPLETED")
print("="*70)
