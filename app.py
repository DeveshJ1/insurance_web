from flask import Flask, render_template, request, g, redirect, url_for
import sqlite3
from datetime import date


app = Flask(__name__)

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
            return redirect(url_for('register'))  # Redirect after successful registration
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
        cursor.execute("SELECT * FROM CustomerAccts WHERE AcctName = ? AND Password = ?", (username, password))
        account = cursor.fetchone()

        if account:
            return "Login successful!"  # Redirect to a dashboard or account page in the future
        else:
            return render_template('login.html', error="Invalid credentials. Please try again or register.")
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)