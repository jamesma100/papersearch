from flask import Flask, request, render_template
from indexer import run_entrypoint

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("form.html")

@app.route("/query", methods=["GET", "POST"])
def handle_query():
    if request.method == "GET":
        return "<p>GET request not allowed. Please go to / to enter your query."
    elif request.method == "POST":
        data = run_entrypoint(
            request.form["max_size"],
            int(request.form["max_results"]),
            int(request.form["process_cnt"]),
            request.form["query"].split(" ")
        )
        return render_template("data.html", data=data)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
