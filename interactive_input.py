import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class SASStylePrompts:
    def __init__(self):
        self.submitted = False
        self.values = {}
        self.create_widgets()
    
    def create_widgets(self):
        """Create all prompt widgets matching SAS structure"""
        
        # Style for consistent look
        style = {'description_width': '250px'}
        layout = widgets.Layout(width='600px')
        
        # Environment dropdown options
        env_options = [
            'BELINS_ENV_DEV',
            'BELINS_ENV_TEST',
            'BELINS_ENV_PROD',
            'BELINS_ENV_DEMO'
        ]
        
        # Period dropdown options
        period_options = [
            'Current month',
            'Previous month',
            'Next month',
            'Current month of previous year',
            'Current month of next year',
            'N months ago',
            'N months from now'
        ]
        
        # 1. THISRUN_ENVIRONMENT_ID
        self.env_dropdown = widgets.Dropdown(
            options=env_options,
            value='BELINS_ENV_DEV',
            description='Please select an environment:',
            style=style,
            layout=layout
        )
        
        # 2. THISRUN_PERIOD_TO_CREATE_IN
        self.period_create = widgets.Dropdown(
            options=period_options,
            value='Current month',
            description='Please select a period to create:',
            style=style,
            layout=layout
        )
        
        # N months input for "N months ago/from now" (for create)
        self.n_months_create = widgets.IntText(
            value=1,
            description='Number of months (if applicable):',
            style=style,
            layout=layout,
            disabled=True
        )
        
        # 3. THISRUN_ENVIRONMENT_ID_TO_COPY
        self.env_copy_dropdown = widgets.Dropdown(
            options=env_options,
            value='BELINS_ENV_DEV',
            description='Please select an environment to copy from:',
            style=style,
            layout=layout
        )
        
        # 4. THISRUN_PERIOD_TO_COPY_IN
        self.period_copy = widgets.Dropdown(
            options=period_options,
            value='Previous month',
            description='Please select a period to copy from:',
            style=style,
            layout=layout
        )
        
        # N months input for "N months ago/from now" (for copy)
        self.n_months_copy = widgets.IntText(
            value=1,
            description='Number of months (if applicable):',
            style=style,
            layout=layout,
            disabled=True
        )
        
        # Submit button
        self.submit_btn = widgets.Button(
            description='Submit',
            button_style='success',
            tooltip='Click to submit parameters',
            icon='check',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Output area
        self.output = widgets.Output()
        
        # Event handlers
        self.period_create.observe(self.on_period_create_change, names='value')
        self.period_copy.observe(self.on_period_copy_change, names='value')
        self.submit_btn.on_click(self.on_submit)
    
    def on_period_create_change(self, change):
        """Enable/disable N months input based on period selection"""
        if change['new'] in ['N months ago', 'N months from now']:
            self.n_months_create.disabled = False
        else:
            self.n_months_create.disabled = True
    
    def on_period_copy_change(self, change):
        """Enable/disable N months input based on period selection"""
        if change['new'] in ['N months ago', 'N months from now']:
            self.n_months_copy.disabled = False
        else:
            self.n_months_copy.disabled = True
    
    def calculate_period_date(self, period_text, n_months=1):
        """Calculate the actual date based on period selection"""
        today = datetime.now()
        
        if period_text == 'Current month':
            return today.replace(day=1)
        elif period_text == 'Previous month':
            return (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        elif period_text == 'Next month':
            return (today.replace(day=1) + relativedelta(months=1))
        elif period_text == 'Current month of previous year':
            return today.replace(year=today.year - 1, day=1)
        elif period_text == 'Current month of next year':
            return today.replace(year=today.year + 1, day=1)
        elif period_text == 'N months ago':
            return (today.replace(day=1) - relativedelta(months=n_months))
        elif period_text == 'N months from now':
            return (today.replace(day=1) + relativedelta(months=n_months))
        else:
            return today.replace(day=1)
    
    def on_submit(self, button):
        """Handle submit button click"""
        with self.output:
            clear_output()
            
            # Get values
            env_id = self.env_dropdown.value
            period_create = self.period_create.value
            n_create = self.n_months_create.value
            env_copy_id = self.env_copy_dropdown.value
            period_copy = self.period_copy.value
            n_copy = self.n_months_copy.value
            
            # Calculate actual dates
            create_date = self.calculate_period_date(period_create, n_create)
            copy_date = self.calculate_period_date(period_copy, n_copy)
            
            # Store values
            self.values = {
                'THISRUN_ENVIRONMENT_ID': env_id,
                'THISRUN_PERIOD_TO_CREATE_IN': period_create,
                'THISRUN_PERIOD_TO_CREATE_DATE': create_date,
                'THISRUN_ENVIRONMENT_ID_TO_COPY': env_copy_id,
                'THISRUN_PERIOD_TO_COPY_IN': period_copy,
                'THISRUN_PERIOD_TO_COPY_DATE': copy_date,
                'N_MONTHS_CREATE': n_create if period_create in ['N months ago', 'N months from now'] else None,
                'N_MONTHS_COPY': n_copy if period_copy in ['N months ago', 'N months from now'] else None
            }
            
            self.submitted = True
            
            # Display results
            print("="*70)
            print("PROJECT PROMPTS SUBMITTED")
            print("="*70)
            print(f"\n📋 CREATE Parameters:")
            print(f"  Environment:        {env_id}")
            print(f"  Period:             {period_create}")
            print(f"  Calculated Date:    {create_date.strftime('%Y-%m-%d')}")
            if self.values['N_MONTHS_CREATE']:
                print(f"  Months Offset:      {n_create}")
            
            print(f"\n📋 COPY FROM Parameters:")
            print(f"  Environment:        {env_copy_id}")
            print(f"  Period:             {period_copy}")
            print(f"  Calculated Date:    {copy_date.strftime('%Y-%m-%d')}")
            if self.values['N_MONTHS_COPY']:
                print(f"  Months Offset:      {n_copy}")
            
            print("\n" + "="*70)
            print("✓ Parameters ready for processing")
            print("="*70)
            
            # Show how to access values
            print("\n💡 Access values using:")
            print("   prompts.values['THISRUN_ENVIRONMENT_ID']")
            print("   prompts.values['THISRUN_PERIOD_TO_CREATE_DATE']")
    
    def display(self):
        """Display all widgets"""
        # Header
        header = widgets.HTML(
            value="<h2 style='color: #2e5cb8;'>🔧 Project Prompts</h2>"
                  "<p style='color: #666;'>Please configure the parameters below:</p>"
        )
        
        display(header)
        
        # Section 1: Create Parameters
        section1_header = widgets.HTML(
            value="<h3 style='color: #4a90e2; margin-top: 20px;'>📝 Create Parameters</h3>"
        )
        display(section1_header)
        display(self.env_dropdown)
        display(self.period_create)
        display(self.n_months_create)
        
        # Section 2: Copy From Parameters
        section2_header = widgets.HTML(
            value="<h3 style='color: #4a90e2; margin-top: 20px;'>📂 Copy From Parameters</h3>"
        )
        display(section2_header)
        display(self.env_copy_dropdown)
        display(self.period_copy)
        display(self.n_months_copy)
        
        # Submit button
        display(widgets.HTML(value="<div style='margin-top: 20px;'></div>"))
        display(self.submit_btn)
        
        # Output area
        display(self.output)
    
    def get_values(self):
        """Return the submitted values"""
        if not self.submitted:
            print("⚠️  Please submit the form first!")
            return None
        return self.values


# =============================================================================
# USAGE
# =============================================================================

# Create and display the prompts
prompts = SASStylePrompts()
prompts.display()



#####how to use it 

# 1. Run the above code to display the prompts
# 2. User fills in the form and clicks Submit
# 3. Access the values in your processing code:

# Wait for user to submit (or check if submitted)
if prompts.submitted:
    # Get all values
    params = prompts.get_values()
    
    # Access individual values
    environment = params['THISRUN_ENVIRONMENT_ID']
    create_date = params['THISRUN_PERIOD_TO_CREATE_DATE']
    copy_environment = params['THISRUN_ENVIRONMENT_ID_TO_COPY']
    copy_date = params['THISRUN_PERIOD_TO_COPY_DATE']
    
    print(f"Processing for environment: {environment}")
    print(f"Create date: {create_date}")
    print(f"Copy from environment: {copy_environment}")
    print(f"Copy from date: {copy_date}")
    
    # Use in your Spark code
    from companyframework.functions import commonfunctions as cf
    
    # Example: Build file path based on environment and date
    base_path = cf.get_base_directory("ref")
    
    year_month = create_date.strftime('%Y%m')
    file_path = f"{base_path}/{environment}/{year_month}/data.xlsx"
    
    print(f"File path: {file_path}")
    
    # Your processing logic here...



######"advanced version

class EnhancedSASStylePrompts(SASStylePrompts):
    def on_submit(self, button):
        """Enhanced submit with validation"""
        with self.output:
            clear_output()
            
            # Validate: Can't copy from same environment and period as create
            if (self.env_dropdown.value == self.env_copy_dropdown.value and 
                self.period_create.value == self.period_copy.value):
                print("❌ ERROR: Cannot copy to the same environment and period!")
                print("   Please select different parameters.")
                return
            
            # Continue with normal submit
            super().on_submit(button)

# Use the enhanced version
prompts = EnhancedSASStylePrompts()
prompts.display()
