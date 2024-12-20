from flask import Flask, render_template, request, g
import sqlite3

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

@app.route('/')
def index():
    """Render the homepage and display policies."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT SSN, CustLastName, CustFirstName, Age, Gender FROM Customers")
    policies = cursor.fetchall()
    return render_template('index.html', policies=policies)

if __name__ == '__main__':
    app.run(debug=True)