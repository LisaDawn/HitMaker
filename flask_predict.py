from flask import Flask, url_for, request
import parse_lyrics as pl
app = Flask(__name__)

# this defines input entry page
@app.route('/')
def welcome():
    return '''
    		<form method='POST' action='/results'>
	    		You think you have a hit? Enter your lyrics here:
    			<p>
    			<textarea cols="40" rows="10" name="search" type = "text">
    			</textarea>
    			<br>
	    		<br>
    			<input type= "submit" value="run me!"/>
    		</form>'''

# this is what is returned from post
@app.route('/results', methods = ['POST'])
def classify():

    input_text = request.form.get("search")
    prediction = pl.predictor(input_text)
    if prediction[0]:
        return "You're a star!"
        # and here are the words that made you a star
    else:
        return "Sorry, better luck next time. Don't quit your day job."
  

# @app.route('/articles')
# def api_articles():
#     return 'List of ' + url_for('api_articles')

# @app.route('/articles/<articleid>')
# def api_article(articleid):
#     return 'You are reading ' + articleid

# this runs it in the command line
if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run()