from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os

from main import sudoku_processing

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

class UploadFileForm(FlaskForm):
    file = FileField(
        "File", 
        validators = [InputRequired()]
    )
    submit = SubmitField("Upload File")

@app.route('/', methods=["GET","POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = file.filename.split(".")[0] + ".png"
        filepath = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(filename)
            )
        file.save(filepath)

        if sudoku_processing(filepath):
            return "Processing the image"
        else:
            return "Cannot process the image"

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)