from flask import Flask

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    return "target sales project"


if __name__=="__main__":
    app.run(debug=True)