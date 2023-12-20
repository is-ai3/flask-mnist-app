import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#--- RenderのTensoflow Version確認
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

#-- Program Start -------------------
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#--- renderで、エラーになるので、下記の2行を修正 20230806---これ重要！
#--- 2回目のデプロイは、1.4.2
#model = load_model('./model.h5')#学習済みモデルをロードする ---　元々のコード
model = load_model('./model.h5', compile=False)    #学習済みモデルをロード, Compile=False

#---- 私は研修生の時、上の行と下の行を修正したが、20230922 x0516-yoshizawa-san の
#---- カウンセリングで森田さんは、上の行だけで治ると言った。
#---- しかし、このファイルの一番下では、後ほど、compileしたほうが良いと言っている。
#---- しかし、Aidemyでは、だれも必要とは言っていない。
#model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


#---- 2023.12.21
#---- renderのpythonの版数が上がったため、downgradeさせて修正するという方法にトライ
#
#bardの指示でローカルPCのVerを上げたが効果あるんだろうか？
#python -m pip install --upgrade pip


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, grayscale=True, target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)


"""
if __name__ == "__main__":
    app.run()
"""






"""
Bardでは、
absl-py==0.9.0 を absl-py==0.14.0
といいている  2023.12.20



compile=falseの件、GPT4では、下記の回答あり。


Google Colabは、クラウド上でJupyterノートブックを実行するための環境を提供しています。
KerasやTensorFlowなどのライブラリもプリインストールされていますが、これらのライブラリのバージョンはColabの更新によって変わることがあります。

compile=Falseが必要になるかどうかは、以下の要因によって異なります：

1. モデルの保存時の環境: モデルを保存した際のKerasやTensorFlowのバージョン、およびその他の依存ライブラリのバージョンが、Colab上でのバージョンと異なる場合、互換性の問題が発生する可能性があります。

2. カスタムオブジェクトの使用: モデルがカスタムレイヤーやカスタム損失関数などのカスタムオブジェクトを使用している場合、これらのオブジェクトを指定せずにモデルをロードしようとするとエラーが発生します。compile=Falseを使用すると、このようなコンパイル時の依存関係を無視してモデルをロードできます。

3. モデルの保存方法: 一部のモデル保存方法では、コンパイル情報が保存されていない場合があります。このようなモデルをロードする際にcompile=Falseを使用すると、エラーを回避できます。

compile=Falseを常に使用する必要はありませんが、上記のような特定の状況や問題が発生した場合に有用です。
もしColab上でモデルのロードに問題が発生する場合、具体的なエラーメッセージを確認することで、問題の原因を特定しやすくなります。
"""

"""
その人の環境により、発生したりしなかったりするのですか？

ChatGPT4
はい、モデルをロードする際の問題は、使用している環境によって発生したりしなかったりすることがあります。
以下は、環境によって問題が発生する主な理由です：

1. ライブラリのバージョン: KerasやTensorFlowなどのライブラリのバージョンが異なると、モデルの保存やロードの方法が変わることがあります。特に、メジャーバージョンが異なる場合（例：TensorFlow 1.xと2.x）には、互換性の問題が発生しやすいです。

2. カスタムオブジェクト: カスタムレイヤーやカスタム損失関数を使用しているモデルは、そのカスタムオブジェクトが定義されていない環境ではロードできません。

3. 依存ライブラリ: モデルが特定のライブラリやツールに依存している場合、それらがインストールされていない環境では問題が発生する可能性があります。

4. ハードウェアの違い: 特定のハードウェア（例：GPU）でトレーニングされたモデルは、そのハードウェアがない環境でのロードや実行に問題を引き起こすことがあります。

5.オペレーティングシステム: 珍しいですが、OSの違い（Windows、Linux、Macなど）によっても問題が発生することがあります。

これらの環境差異により、ある環境では問題なくモデルをロードできるのに、別の環境ではエラーが発生することがあります。そのため、モデルを共有する際や、異なる環境での実行を考慮する場合は、環境の詳細（ライブラリのバージョン、OSのバージョンなど）を明記することが重要です。

"""


"""
compile=Falseをload_model関数でデフォルトとして常に使用することには、以下のような利点と潜在的な問題点が考えられます。

利点:
互換性の問題を回避: 以前に述べたように、モデルの保存時とロード時の環境が異なる場合や、カスタムオブジェクトが関与する場合など、互換性の問題を回避することができます。

高速なロード: モデルのコンパイル処理が省略されるため、モデルのロードがわずかに高速になる可能性があります。

潜在的な問題点:
後でのコンパイルが必要: compile=Falseでモデルをロードした場合、そのモデルを訓練や評価する前に、明示的にモデルをコンパイルする必要があります。これを忘れると、訓練や評価を試みた際にエラーが発生します。

コンパイル情報の喪失: モデルを保存する際のコンパイル情報（最適化アルゴリズム、損失関数など）が失われるため、後で同じ設定でモデルをコンパイルする必要がある場合、その情報を別途保存しておくか、記憶しておく必要があります。

誤解の原因: 他の開発者やユーザーがコードを見たとき、compile=Falseが常に使用されている理由を理解できない場合があります。そのため、この選択の理由をコメントやドキュメントで明記しておくと良いでしょう。

結論として、compile=Falseをデフォルトとして常に使用することに特に大きな問題はありませんが、上記の潜在的な問題点を理解し、適切に対応することが重要です。





Regenerate


"""

"""
compile=False をデフォルトとして、必ず設定すると、なにか問題がありますか？
---

compile=Falseをload_model関数でデフォルトとして常に使用することには、以下のような利点と潜在的な問題点が考えられます。

利点:
1. 互換性の問題を回避: 以前に述べたように、モデルの保存時とロード時の環境が異なる場合や、カスタムオブジェクトが関与する場合など、互換性の問題を回避することができます。

2. 高速なロード: モデルのコンパイル処理が省略されるため、モデルのロードがわずかに高速になる可能性があります。

潜在的な問題点:
1. 後でのコンパイルが必要: compile=Falseでモデルをロードした場合、そのモデルを訓練や評価する前に、明示的にモデルをコンパイルする必要があります。これを忘れると、訓練や評価を試みた際にエラーが発生します。

2. コンパイル情報の喪失: モデルを保存する際のコンパイル情報（最適化アルゴリズム、損失関数など）が失われるため、後で同じ設定でモデルをコンパイルする必要がある場合、その情報を別途保存しておくか、記憶しておく必要があります。

3. 誤解の原因: 他の開発者やユーザーがコードを見たとき、compile=Falseが常に使用されている理由を理解できない場合があります。そのため、この選択の理由をコメントやドキュメントで明記しておくと良いでしょう。

結論として、compile=Falseをデフォルトとして常に使用することに特に大きな問題はありませんが、上記の潜在的な問題点を理解し、適切に対応することが重要です。





Regenerate


"""



