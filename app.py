"""
app.py
------
CogniSearch AI — document Q&A app.
Flask REST API + FAISS/ChromaDB-style vector index for fast, contextual
semantic search over uploaded documents.

Run:
    python app.py
Then open http://localhost:5000
"""
from flask import Flask, render_template

import config
from routes.api import api


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024

    app.register_blueprint(api)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
