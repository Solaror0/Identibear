from flask import app, Flask, redirect, url_for, request, render_template, flash

app = Flask(__name__)

@app.route('/')
def default():
    return render_template('welcome.html')


if __name__ == '__main__':
    app.run(debug=True)

