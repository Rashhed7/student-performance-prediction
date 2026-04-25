from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pickle, os, sqlite3
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = pickle.load(open("model.pkl", "rb"))
explainer = shap.Explainer(model)

def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    # Create table
    cur.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")

    # Check if user exists
    cur.execute("SELECT * FROM users WHERE username=?", ('admin',))
    user = cur.fetchone()

    # Insert default login
    if not user:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', '1234'))

    conn.commit()
    conn.close()

init_db()

# ---------------- LOGIN ----------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        data = cur.fetchone()
        conn.close()

        if data:
            session['user'] = user
            return redirect('/dashboard')
        else:
            return "Invalid Login"

    return render_template('login.html')

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html')

# ---------------- SINGLE STUDENT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    features = np.array([[int(request.form[x]) for x in ['studytime','failures','absences','G1','G2']]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    result = "Pass" if pred==1 else "At Risk"

    # SHAP
    shap_values = explainer(features)
    importance = shap_values.values[0]

    # Chart
    plt.figure()
    plt.bar(['Study','Failures','Abs','G1','G2'], importance)
    chart_path = "static/chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template('result.html',
                           name=name,
                           prediction=result,
                           probability=round(prob*100,2),
                           chart=chart_path)

# ---------------- CSV UPLOAD ----------------
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        df = pd.read_csv(path)

        X = df[['studytime','failures','absences','G1','G2']]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:,1]

        df['Prediction'] = ["Pass" if p==1 else "At Risk" for p in preds]
        df['Confidence'] = (probs*100).round(2)

        result_path = os.path.join(RESULT_FOLDER, "output.csv")
        df.to_csv(result_path, index=False)

        # Chart
        counts = df['Prediction'].value_counts()
        plt.figure()
        counts.plot(kind='bar')
        chart_path = "static/multi_chart.png"
        plt.savefig(chart_path)
        plt.close()

        return render_template('upload.html',
                               table=df.to_html(index=False),
                               chart=chart_path)

    return render_template('upload.html')

# ---------------- DOWNLOAD ----------------
@app.route('/download')
def download():
    return send_file("results/output.csv", as_attachment=True)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)