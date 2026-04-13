# Create interactive widgets
from IPython.display import display
import ipywidgets as widgets

# Text input
file_input = widgets.Text(
    value='data.xlsx',
    placeholder='Enter file name',
    description='File Name:',
    disabled=False
)

sheet_input = widgets.Text(
    value='Sheet1',
    placeholder='Enter sheet name',
    description='Sheet:',
    disabled=False
)

# Dropdown
location_dropdown = widgets.Dropdown(
    options=['ref', 'stg'],
    value='ref',
    description='Location:',
)

# Number input
start_row_input = widgets.IntText(
    value=10,
    description='Start Row:',
    disabled=False
)

# Button
submit_button = widgets.Button(
    description='Process File',
    button_style='success',
    icon='check'
)

# Output area
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        print(f"Processing file: {file_input.value}")
        print(f"Sheet: {sheet_input.value}")
        print(f"Location: {location_dropdown.value}")
        print(f"Starting at row: {start_row_input.value}")
        
        # Your processing logic here
        from companyframework.functions import commonfunctions as cf
        
        location = cf.get_base_directory(location_dropdown.value)
        file_path = f"{location}/{file_input.value}"
        
        try:
            df = spark.read.format("excel") \
                .option("dataAddress", f"'{sheet_input.value}'!A{start_row_input.value}:H100") \
                .option("header", "false") \
                .load(file_path)
            
            print(f"\n✓ Successfully loaded {df.count()} rows")
            df.show(10, truncate=False)
        except Exception as e:
            print(f"✗ Error: {e}")

submit_button.on_click(on_button_click)

# Display all widgets
display(file_input)
display(sheet_input)
display(location_dropdown)
display(start_row_input)
display(submit_button)
display(output)






########### multiple choice with validation

def get_user_inputs():
    """Interactive prompt with validation"""
    
    # Get location
    print("Select storage location:")
    print("1. ref")
    print("2. stg")
    location_choice = input("Enter choice (1 or 2): ")
    
    location = "ref" if location_choice == "1" else "stg"
    
    # Get file name with validation
    while True:
        file_name = input("\nEnter Excel file name (e.g., data.xlsx): ")
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            break
        else:
            print("❌ Please enter a valid Excel file (.xlsx or .xls)")
    
    # Get sheet name
    sheet_name = input("Enter sheet name (default: Sheet1): ") or "Sheet1"
    
    # Get start row with validation
    while True:
        try:
            start_row = int(input("Enter starting row number (e.g., 10): "))
            if start_row > 0:
                break
            else:
                print("❌ Row number must be greater than 0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Confirm inputs
    print("\n" + "="*50)
    print("Please confirm your inputs:")
    print("="*50)
    print(f"Location: {location}")
    print(f"File: {file_name}")
    print(f"Sheet: {sheet_name}")
    print(f"Start Row: {start_row}")
    print("="*50)
    
    confirm = input("\nProceed? (yes/no): ").lower()
    
    if confirm in ['yes', 'y']:
        return {
            'location': location,
            'file_name': file_name,
            'sheet_name': sheet_name,
            'start_row': start_row
        }
    else:
        print("Cancelled by user")
        return None

# Use it
inputs = get_user_inputs()

if inputs:
    from companyframework.functions import commonfunctions as cf
    
    base_path = cf.get_base_directory(inputs['location'])
    file_path = f"{base_path}/{inputs['file_name']}"
    
    print(f"\n📂 Reading from: {file_path}")
    
    df = spark.read.format("excel") \
        .option("dataAddress", f"'{inputs['sheet_name']}!'A{inputs['start_row']}:H100") \
        .option("header", "false") \
        .load(file_path)
    
    print(f"✓ Loaded {df.count()} rows")
    df.show()


#########"using synapse parameteres


# Define parameters at the top of notebook
# These can be set when running via pipeline or manually

# In Synapse, you can use notebook parameters
from typing import Optional

# Default values
file_name: str = "data.xlsx"
sheet_name: str = "Sheet1"
location: str = "ref"
start_row: int = 10

# If run manually, prompt for input
import sys
if sys.stdin.isatty():  # Check if running interactively
    file_name = input(f"Enter file name [{file_name}]: ") or file_name
    sheet_name = input(f"Enter sheet name [{sheet_name}]: ") or sheet_name
    location = input(f"Enter location (ref/stg) [{location}]: ") or location
    start_row = int(input(f"Enter start row [{start_row}]: ") or start_row)

print(f"\n📋 Parameters:")
print(f"  File: {file_name}")
print(f"  Sheet: {sheet_name}")
print(f"  Location: {location}")
print(f"  Start Row: {start_row}")

# Your processing logic
from companyframework.functions import commonfunctions as cf

base_path = cf.get_base_directory(location)
file_path = f"{base_path}/{file_name}"

df = spark.read.format("excel") \
    .option("dataAddress", f"'{sheet_name}'!A{start_row}:H100") \
    .option("header", "false") \
    .load(file_path)

df.show()



##########"advanced interactive 

import ipywidgets as widgets
from IPython.display import display, clear_output

class NotebookInputForm:
    def __init__(self):
        self.inputs = {}
        self.create_widgets()
    
    def create_widgets(self):
        # File selection
        self.file_input = widgets.Text(
            value='data.xlsx',
            description='File:',
            style={'description_width': '120px'}
        )
        
        # Sheet selection
        self.sheet_input = widgets.Text(
            value='Sheet1',
            description='Sheet Name:',
            style={'description_width': '120px'}
        )
        
        # Location
        self.location = widgets.RadioButtons(
            options=['ref', 'stg'],
            value='ref',
            description='Storage:',
            style={'description_width': '120px'}
        )
        
        # Row range
        self.start_row = widgets.IntText(
            value=10,
            description='Start Row:',
            style={'description_width': '120px'}
        )
        
        self.end_row = widgets.IntText(
            value=100,
            description='End Row:',
            style={'description_width': '120px'}
        )
        
        # Column range
        self.start_col = widgets.Text(
            value='A',
            description='Start Column:',
            style={'description_width': '120px'}
        )
        
        self.end_col = widgets.Text(
            value='H',
            description='End Column:',
            style={'description_width': '120px'}
        )
        
        # Submit button
        self.submit_btn = widgets.Button(
            description='Load Data',
            button_style='primary',
            icon='download'
        )
        
        self.submit_btn.on_click(self.on_submit)
        
        # Output
        self.output = widgets.Output()
    
    def display(self):
        display(widgets.HTML("<h3>📊 Excel Data Loader</h3>"))
        display(self.location)
        display(self.file_input)
        display(self.sheet_input)
        display(widgets.HBox([self.start_row, self.end_row]))
        display(widgets.HBox([self.start_col, self.end_col]))
        display(self.submit_btn)
        display(self.output)
    
    def on_submit(self, button):
        with self.output:
            clear_output()
            
            # Get values
            location = self.location.value
            file_name = self.file_input.value
            sheet_name = self.sheet_input.value
            start_row = self.start_row.value
            end_row = self.end_row.value
            start_col = self.start_col.value
            end_col = self.end_col.value
            
            print(f"📂 Loading data...")
            print(f"  Location: {location}")
            print(f"  File: {file_name}")
            print(f"  Sheet: {sheet_name}")
            print(f"  Range: {start_col}{start_row}:{end_col}{end_row}")
            
            try:
                from companyframework.functions import commonfunctions as cf
                
                base_path = cf.get_base_directory(location)
                file_path = f"{base_path}/{file_name}"
                
                cell_range = f"'{sheet_name}'!{start_col}{start_row}:{end_col}{end_row}"
                
                df = spark.read.format("excel") \
                    .option("dataAddress", cell_range) \
                    .option("header", "false") \
                    .load(file_path)
                
                print(f"\n✓ Successfully loaded {df.count()} rows")
                df.show(10, truncate=False)
                
                # Store for later use
                self.inputs['df'] = df
                self.inputs['config'] = {
                    'location': location,
                    'file': file_name,
                    'sheet': sheet_name,
                    'range': cell_range
                }
                
            except Exception as e:
                print(f"✗ Error: {e}")

# Use it
form = NotebookInputForm()
form.display()

# Access the loaded dataframe later
# df = form.inputs.get('df')



#########very simple 

# Start of your notebook - simple and effective

print("="*60)
print("           EXCEL DATA LOADER")
print("="*60)

# Get inputs
location = input("\n📁 Enter location (ref/stg) [default: ref]: ") or "ref"
file_name = input("📄 Enter Excel file name: ")
sheet_name = input("📋 Enter sheet name [default: Sheet1]: ") or "Sheet1"
start_row = int(input("🔢 Enter start row [default: 10]: ") or "10")

print("\n" + "="*60)
print("Processing...")
print("="*60)

# Your code
from companyframework.functions import commonfunctions as cf

base_path = cf.get_base_directory(location)
file_path = f"{base_path}/{file_name}"

df = spark.read.format("excel") \
    .option("dataAddress", f"'{sheet_name}'!A{start_row}:H100") \
    .option("header", "false") \
    .load(file_path)

print(f"✓ Loaded {df.count()} rows from {file_name}")
df.show()
