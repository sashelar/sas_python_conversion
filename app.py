from flask import Flask, request, jsonify, render_template
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sas_to_pyspark import SASToSparkConverter

app = Flask(__name__)
converter = SASToSparkConverter()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json()
    sas_code = data.get("sas_code", "").strip()
    if not sas_code:
        return jsonify({"error": "No SAS code provided"}), 400
    try:
        result = converter.convert(sas_code)
        return jsonify({"pyspark_code": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)