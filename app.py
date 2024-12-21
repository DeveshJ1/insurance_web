from flask import Flask, render_template, request, g, redirect, url_for
import sqlite3
from datetime import date
from flask import session


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database configuration
DATABASE = 'InsuranceDB.db'

def get_db():
    """Open a database connection if not already open."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/', methods=['GET', 'POST'])
def home():
    """Render home page with options for login or registration."""
    if request.method == 'POST':
        action = request.form['action']
        if action == 'login':
            return redirect(url_for('login'))
        elif action == 'register':
            return redirect(url_for('register'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Render registration page and handle form submissions."""
    if request.method == 'POST':
        # Collect form data
        last_name = request.form['last_name']
        first_name = request.form['first_name']
        middle_initial = request.form['middle_initial']
        email = request.form['email']
        gender = request.form['gender']
        age = request.form['age']
        date_of_birth = request.form['date_of_birth']
        marital_status = request.form['marital_status']
        education_level = request.form['education_level']
        occupation = request.form['occupation']
        ssn = request.form['ssn']
        income = request.form['income']
        phone_number = request.form['phone_number']
        credit_score = request.form['credit_score']
        password=request.form['password']
        username=request.form['username']
        # Insert data into the Customers table
        db = get_db()
        cursor = db.cursor()
        today = date.today()
        try:
            cursor.execute('''INSERT INTO Customers 
                (CustLastName, CustFirstName, CustMiddleInitial, EmailAddress, Gender, Age, CustDOB, 
                MaritalStatus, EducationLvl, Occupation, SSN, Income, PhoneNumber, CreditScore)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (last_name, first_name, middle_initial, email, gender, age, date_of_birth, 
                 marital_status, education_level, occupation, ssn, income, phone_number, credit_score))
            cursor.execute('''INSERT INTO CustomerAccts
                           (CustSSN, AcctName, StartDate, CompanyCode, numPolicies, numClaims, OutstandingDeductibleAmount, Password)
                          VALUES (?, ?, ?, ?, ?, ?, ?,?)''',
                           (ssn,username,today,0,0,0,0,password))
            db.commit()
            return redirect(url_for('login'))  # Redirect after successful registration
        except sqlite3.IntegrityError as e:
            return f"Error: {e}", 400

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login functionality."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check username and password in the database
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT CustSSN FROM CustomerAccts WHERE AcctName = ? AND Password = ?", (username, password))
        account = cursor.fetchone()

        if account:
            customer_id=account[0]
            session['customer_id'] = account[0]  # Store the CustSSN in session
            return redirect(url_for('policies',customer_id=customer_id))
        else:
            return render_template('login.html', error="Invalid credentials. Please try again or register.")
    return render_template('login.html')

@app.route('/policies')
def policies():
    """Render the policies page with data from CompanyPolicies table."""
    customer_id = request.args.get('customer_id')
    db = get_db()
    cursor = db.cursor()

    # Query policies excluding Premium and Deductible
    query = """
        SELECT PolicyType, EffectiveDate, ExpiredDate, PolicyLimit, Region, DependentsAllowed, PolicyNum
        FROM ComapnyPolicies
    """
    cursor.execute(query)
    policies = cursor.fetchall()
    # Pass the policies to the template
    return render_template('policy.html', policies=policies, customer_id=customer_id)

@app.route('/check_invoices', methods=['GET'])
def check_invoices():
    """Display all the user's invoices."""
    customer_id = session.get('customer_id')

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM CustomerInvoices WHERE Cust_SSN = ?", (customer_id,))
    invoices = cursor.fetchall()

    return render_template('invoices.html', invoices=invoices)
@app.route('/file_claim', methods=['GET'])
def file_claim():
    """Display all the user's policies for filing claims."""
    customer_id = session.get('customer_id')

    db = get_db()
    cursor = db.cursor()

    # Fetch health policies
    cursor.execute("SELECT * FROM CustomerHealthPolicies WHERE CustSSN = ?", (customer_id,))
    health_policies = cursor.fetchall()

    # Fetch vehicle policies
    cursor.execute("SELECT * FROM CustomerVehiclePolicies WHERE CustSSN = ?", (customer_id,))
    vehicle_policies = cursor.fetchall()

    return render_template('file_claim.html', health_policies=health_policies, vehicle_policies=vehicle_policies)

@app.route('/generate_quote', methods=['POST'])
def generate_quote():
    """Render a form based on the policy type for quote generation."""
    policy_type = request.form['policy_type']
    policy_num = request.form['policy_num']
   
    if policy_type.lower() == 'health':
        fields = ['Height_Inches', 'Weight_Pounds', 'BMI', 'NumChildren', 'Smoker']
    elif policy_type.lower() == 'vehicle':
        fields = [
            'VehicleCompany', 'VehiclePrice', 'VehicleCategory', 'PastNumClaims',
            'VehicleModel', 'YearsExperience', 'NumTickets', 'NumPastAccidents',
            'VehicleDamageRating', 'VehicleID'
        ]
    else:
        return "Unsupported Policy Type", 400

    return render_template('quote_form.html', fields=fields, policy_type=policy_type, policy_num=policy_num)

@app.route('/submit_quote', methods=['POST'])
def submit_quote():
    """Process the submitted quote data."""
    policy_type = request.form['policy_type']
    policy_num = request.form['policy_num']
   
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT Premium, Deductible FROM ComapnyPolicies WHERE PolicyNum = ?",
        (policy_num,)
    )
    policy_data = cursor.fetchone()
    if policy_data:
        premium, deductible = policy_data
        return render_template(
            'quote_results.html',
            policy_num=policy_num,
            policy_type=policy_type,
            premium=premium,
            deductible=deductible
        )
    else:
        return f"Policy with Policy Number {policy_num} not found.", 404
    #data = {key: value for key, value in request.form.items() if key not in ['policy_type', 'policy_num']}

    # For now, just return the data as a confirmation
    # You can add quote calculations or database saving here
   # return f"Quote data for Policy #{policy_num} ({policy_type}): {data}"
@app.route('/buy_policy/<policy_num>', methods=['GET'])
def buy_policy(policy_num):
    """Display the details of the policy for purchase."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT PolicyType, EffectiveDate, ExpiredDate, Premium, Deductible, 
               PolicyLimit, Region, DependentsAllowed, PolicyNum
        FROM ComapnyPolicies WHERE PolicyNum = ?
        """,
        (policy_num,)
    )
    policy = cursor.fetchone()
    if policy:
        return render_template('buy_policies.html', policy=policy)
    else:
        return f"Policy with Policy Number {policy_num} not found.", 404
    
@app.route('/confirm_purchase/<policy_num>', methods=['POST'])
def confirm_purchase(policy_num):
    customer_id = session.get('customer_id')
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT PolicyType FROM ComapnyPolicies WHERE PolicyNum = ?", (policy_num,))
    policy = cursor.fetchone()

    if not policy:
        return "<h1>Error: Policy not found!</h1>"

    policy_type = policy[0]

    # Insert data based on the policy type
    if policy_type == 'Health':
        cursor.execute("""
            INSERT INTO CustomerHealthPolicies (CustSSN,PolicyNumber,Agent_Sold)
            VALUES (?, ?, ?)
        """, (customer_id, policy_num, 0))
    elif policy_type == 'Vehicle':
        cursor.execute("""
            INSERT INTO CustomerVehiclePolicies (CustSSN, Vehicle_ID,PolicyNumber,Agent_Sold)
            VALUES (?, ?, ?, ?)
        """, (customer_id,0,policy_num,0))

    db.commit()
    return f"""
        <html>
            <head>
                <title>Purchase Confirmation</title>
            </head>
            <body>
                <h1>Policy {policy_num} has been successfully purchased!</h1>
                <form action="/policies" method="GET">
                    <button type="submit">Back to Browse Policies</button>
                </form>
            </body>
        </html>
    """
    
@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('login'))
if __name__ == '__main__':
    app.run(debug=True)