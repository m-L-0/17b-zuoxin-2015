from flask import Flask, render_template, request, flash, redirect, session, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, FileField
from flask_wtf.file import FileRequired, FileAllowed
from werkzeug.utils import secure_filename
import sys
sys.path.append('../')
from validcnn import cnn
from multiprocessing import Pool
import matplotlib.image as mpimg
from scipy import misc

bootstrap = Bootstrap()
app = Flask(__name__)
bootstrap.init_app(app)
app.config['SECRET_KEY']='123'

class Form(FlaskForm):
    image = FileField('请上传测试图片(jpg、png、jpeg)',
                      validators=[FileRequired(), FileAllowed(['jpg', 'png', 'jpeg'], '只允许上传图片!')])
    submit = SubmitField(u"分析")


@app.route('/', methods=['GET','POST'])
def index():
    form = Form()
    if form.validate_on_submit():
        filename = secure_filename(form.image.data.filename)
        form.image.data.save('upload/' + filename)
        pic = mpimg.imread('upload/' + filename)
        tmppic = misc.imresize(pic, (36, 48))
        p = Pool()
        result = p.apply_async(cnn, (tmppic,))
        session['out'] = result.get()
        return redirect(url_for('index'))
    return render_template('index.html', form=form, out=session.get('out'))

if __name__ == '__main__':
    app.run(debug=True)