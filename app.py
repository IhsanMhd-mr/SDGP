from flask import Flask
app=Flask(__name__)

@app.route('/')
def welcome():
    return'welcome there-first time huh'

if __name__=='__main__':
    app.run()