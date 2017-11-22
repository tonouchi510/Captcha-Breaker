from flask import Flask, request, url_for, render_template, make_response
import CaptchaBreaker_v2 as cb
import segment as seg
import numpy as np
import os
import cv2

DEBUG = True
SECRET_KEY = 'development key'
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png'])  # 今回はpngのみ

app = Flask(__name__)
app.config.from_object(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def show_index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def do_upload():
    file = request.files['xhr2upload']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(UPLOAD_FOLDER+filename, 0)
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        # 収縮、膨張処理と画像の表示
        kernel = np.ones((4, 4), np.uint8)
        img = cv2.dilate(img, kernel, 1)  # 画像の膨張処理(小さな点や線を消す)
        kernel = np.ones((4, 2), np.uint8)
        img = cv2.erode(img, kernel, 1)  # 画像の収縮処理
        ''' 線や点の除去後の画像
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        # セグメンテーションと予測
        y = seg.segment(img)
        result = ''.join(y)
        print('result: {}'.format(result))

        f = open('static/result.html', 'w')
        f.write('<p>'+result+'</p>')
        f.close()

        ''' # これはテスト用(1文字だけ)
        x = img / 255.0
        out, index = cb.inference(x)
        y = chr(index+65)
        print(y)
        '''
        response = make_response(url_for('static', filename='uploads/'+filename, _external=True))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

if __name__ == '__main__':
    app.run()

