from flask import Flask, render_template,url_for,redirect,session,request,flash,send_file
from authlib.integrations.flask_client import OAuth
from passlib.hash import sha256_crypt
from flask_pymongo import PyMongo
from bson.json_util import dumps
import json
import zipfile
import os
import pickle
app = Flask(__name__)
app.config['MONGO_URI']='mongodb+srv://zaidAhmed:pakistan1999$@mb.pvwut.mongodb.net/MB?retryWrites=true&w=majority'
mongo = PyMongo(app)
oauth = OAuth(app)
app.secret_key=("123")

google = oauth.register(
    name = 'google',
    client_id='1061641137723-hhefi45ih1n7a7ej4ois859o3h3r1u73.apps.googleusercontent.com',
    client_secret='chs57MRobQyNf4dbh-WAd43n',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope':'openid profile email'},
)
@app.route('/')
def main():
    if "profile" in session :
        return redirect(url_for('dashboard'))
    else:    
        return render_template('login.html')

#Sign in with google button

@app.route('/login')
def login():
    google = oauth.create_client('google')
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/authorize')
def authorize():
    google = oauth.create_client('google')
    token = google.authorize_access_token()
    resp = google.get('userinfo')
    user_info = resp.json()
    # do something with the token and profile
    session['profile'] = user_info['email']
    # make the session permanant so it keeps existing after broweser gets closed
    session.permanent = True  
    cursor = mongo.db.userRegistration
    email = dict(session)['profile']
    if cursor.find_one({"email":email}):
        flash("You're now logged in" , "info")
        return redirect('/dashboard')
    else:
        cursor.insert({"email":email,"name":"N/A","gender":"N/A","location":"N/A","profile":'N/A'})
        name = session["profile"] 
        myquery = {"email":name}
        user_info = cursor.find(myquery)
        flash("You're now logged in" , "info")
        return redirect('/dashboard')

#dashboard

@app.route('/dashboard',methods=["GET","POST"])
def dashboard():
    if "profile" in session :
        cursor = mongo.db.modelsInputs
        name = session["profile"] 
        myquery = {"User":name}
        user_info = cursor.find(myquery)
        return render_template('dashboard.html' ,user_info=user_info)
    else:
        return redirect(url_for("main"))

#signup form

@app.route('/RegisterUser',methods=["GET","POST"])
def sigup_form():
    data = request.form
    name = data["name"]
    email = data["email"]
    password = sha256_crypt.encrypt(str(data["password"]))
    cursor = mongo.db.userRegistration
    user_info = cursor.find_one({"email":email})
    if user_info==None:
        cursor.insert_one({"name":name,"email":email,"password":password,"gender":"N/A","location":"N/A",'profile':'N/A'})
        user_info = cursor.find_one({"email":email})
        session['profile'] = user_info["email"]
        session.permanent = True  # make the session permanant so it keeps existing after broweser gets closed
        flash("You're now logged in" , "info")
        return redirect(url_for('dashboard'))
    
    else:
        flash("You're now logged in" , "info")
        return render_template('dashboard.html' ,user_info=user_info)

    
#signin form

@app.route("/signinUser", methods = ["GET","POST"])
def signin():
    data = request.form
    name = data["email"]
    password_candid = data["password"]
    cursor = mongo.db.userRegistration
    user_info = cursor.find_one({"email":name})
    if user_info :
        password = user_info['password']
        if sha256_crypt.verify(password_candid , password):
            session['profile'] = user_info['email']
            session.permanent = True
            return redirect('/dashboard')
        else :
            flash("Incorrect Password...!","danger")
            return redirect(url_for('main'))
    else:
        flash("Username not found...!" , "danger")
        return redirect(url_for('main'))

#ML Inputs

@app.route('/MlForm')
def MlForm():
    if "profile" in session :
        return render_template('Minputs.html')
    else:
        return(redirect('/'))


@app.route('/MlInputs' , methods = ["GET" , "POST"])
def MlInputs():
    if "profile" in session :
        try:
            data =  request.form
            TrainingDataSet = request.files['TrainingDataSet']
            ValidationDataSet = request.files['ValidationDataSet']
            CnnNumberOfLayers = data['NumberOfCNNLayer']
            CnnLayerOutputs = data['CNNLayersOutput']
            DenseNumberOfLayers =data['NumberOfDenseLayer']
            DenseLayerOutputs = data['DenseLayersOutput']
            BeginingLayersOutputFunction = data['BeginingActivation']
            EndLayerOutputFunction = data['EndActivation']
            InputShape = data['InputShape']
            LossFunction = data['LossFunction']
            LearingRate = data['LearningRate']
            TargetSize = data['TargetSize']
            ClassMode = data['ClassMode']  
            BatchSize = data['BatchSize']
            Epochs = data['Epochs']
            StepsPerEpoch = data['StepPerEpochs']
            ValidationSteps =data['ValidationSteps']
                #inserting data into database
            cursor = mongo.db.modelsInputs
            cursor.insert_one({"User":session['profile'],"Model":"Machine Learning","CnnNumberOfLayers":CnnNumberOfLayers,"CnnLayerOutputs":CnnLayerOutputs,
            "DenseNumberOfLayers":DenseNumberOfLayers,"DenseLayerOutputs":DenseLayerOutputs,
            "BeginingLayersOutputFunction":BeginingLayersOutputFunction,"EndLayerOutputFunction":EndLayerOutputFunction,
            "InputShape":InputShape,"LossFunction":LossFunction,"LearingRate":LearingRate,"TargetSize":TargetSize,
            "ClassMode":ClassMode,"BatchSize":BatchSize,"Epochs":Epochs,"StepsPerEpoch":StepsPerEpoch,
            "ValidationSteps":ValidationSteps,"accuracy":0})
            result = cursor.find({"User":session['profile']}).sort([("_id",-1)])
                #DataSet Handling
            target1 = TrainingDataSet
            target2 = ValidationDataSet
            handle1 = zipfile.ZipFile(target1)
            handle2 = zipfile.ZipFile(target2)
            handle1.extractall(f'Uploads/MachineLearning/{result[0]["_id"]}/training')
            handle2.extractall(f'Uploads/MachineLearning/{result[0]["_id"]}/validation')
                #creating a file
            a = str(result[0]["_id"])
            b = CnnLayerOutputs.split(",")
            c = DenseLayerOutputs.split(",")
            f = open(f'Uploads/{a}.py',"w")
            f.write("from keras import layers \nfrom keras import models \nmodel = models.Sequential()")
            f.write(f'\ntrain_dir ="Uploads/MachineLearning/{result[0]["_id"]}/training" ')
            f.write(f'\nvalidation_dir = "Uploads/MachineLearning/{result[0]["_id"]}/validation"')
            for i in range(0,int(CnnNumberOfLayers)):
                if i == 0:
                    f.write(f'\nmodel.add(layers.Conv2D({b[i]}, (3, 3), activation="{BeginingLayersOutputFunction}", input_shape=({InputShape})))')
                    f.write(f'\nmodel.add(layers.MaxPooling2D((2, 2))) ')
                else:
                    f.write(f'\nmodel.add(layers.Conv2D({b[i]}, (3, 3), activation="{BeginingLayersOutputFunction}"))')
                    f.write(f'\nmodel.add(layers.MaxPooling2D((2, 2))) ')

            f.write("\nmodel.add(layers.Flatten())")

            for i in range(0,int(DenseNumberOfLayers)):
                if i == int(DenseNumberOfLayers)-1:
                    f.write(f'\nmodel.add(layers.Dense({c[i]}, activation="{EndLayerOutputFunction}"))')
                else:
                    f.write(f'\nmodel.add(layers.Dense({c[i]}, activation="{BeginingLayersOutputFunction}"))')

            f.write("\nfrom keras import optimizers \nfrom keras import backend as k")
            f.write(f'\nmodel.compile(loss="{LossFunction}", optimizer=optimizers.RMSprop(lr=1e{LearingRate}),metrics=["acc"])')
            f.write(f'\nfrom keras.preprocessing.image import ImageDataGenerator \ntrain_datagen = ImageDataGenerator(rescale=1./255) \ntest_datagen = ImageDataGenerator(rescale=1./255)')
            f.write(f'\ntrain_generator = train_datagen.flow_from_directory(train_dir, target_size=({TargetSize}),batch_size={BatchSize} , class_mode="{ClassMode}") \nvalidation_generator = test_datagen.flow_from_directory( validation_dir,target_size=({TargetSize}) , batch_size={BatchSize} , class_mode="{ClassMode}")')
            f.write(f'\nhistory = model.fit_generator(train_generator,steps_per_epoch={StepsPerEpoch},epochs={Epochs},validation_data=validation_generator,validation_steps={ValidationSteps})')
            f.write(f'\npickle.dump(history,open("Uploads/{a}.p","wb"))')
            f.write(f'\nmodel.save("Uploads/h5/{a}.h5")')
            f.write(f'\nk.clear_session()')
            f.close()
            exec(compile(open(f'Uploads/{a}.py', "rb").read(), f'{a}.py', 'exec'))
            z = pickle.load(open(f'Uploads/{a}.p',"rb"))
            z = z.history['acc'][int(Epochs)-1]
            cursor.find_one_and_update({"_id":result[0]["_id"]},{"$set":{"accuracy":f'{str(z*100)}'}})
            return redirect(url_for(dashboard))    
                    
        except :
            flash("Make sure everything is as per requirment", "danger")
            return redirect(url_for('MlForm'))
        
    else:
        return redirect(url_for('main'))

#DL Inputs

@app.route('/DlForm')
def dlForm():
    if "profile" in session:
        return render_template('Dinputs.html')
    else:
        return redirect('/')
        
@app.route('/DlInputs' , methods = ['GET' , 'POST'])
def DlInputs():
    try:
        data =  request.form
        trainingFile = request.files['training']
        fileName = trainingFile.filename
        Name = data['name']
        Description = data['description']
        NumberOfFeatures = data['NumberOfFeatues']
        DenseNumberOfLayers =data['NumberOfDenseLayer']
        DenseLayerOutputs = data['DenseLayersOutput']
        BeginingLayersOutputFunction = data['BeginingActivation']
        EndLayerOutputFunction = data['EndActivation']
        InputShape = data['InputShape']
        LossFunction = data['LossFunction']
        BatchSize = data['BatchSize']
        Epochs = data['Epochs']
        cursor = mongo.db.modelsInputs
        cursor.insert_one({"User":session['profile'],"Model":"Deep Learning","Name":Name,"Description":Description,
        "DenseNumberOfLayers":DenseNumberOfLayers,"DenseLayerOutputs":DenseLayerOutputs,
        "BeginingLayersOutputFunction":BeginingLayersOutputFunction,"EndLayerOutputFunction":EndLayerOutputFunction,
        "InputShape":InputShape,"LossFunction":LossFunction,"BatchSize":BatchSize,"Epochs":Epochs,"accuracy":0})
        result = cursor.find({"User":session['profile']}).sort([("_id",-1)])
        #handling dataset
        validation_dir = os.path.join("Uploads/DeepLearning",f'{result[0]["_id"]}')
        os.mkdir(validation_dir)    
        trainingFile.save(os.path.join(validation_dir , trainingFile.filename)) 
        #creating file
        a = str(result[0]["_id"])
        c = DenseLayerOutputs.split(",")
        f = open(f'Uploads/{a}.py',"w")
        f.write("from keras.models import Sequential\nimport pickle \nfrom keras.layers import Dense\nimport pandas \nimport numpy \nfrom keras.wrappers.scikit_learn import KerasClassifier \nfrom sklearn.model_selection import cross_val_score \nfrom sklearn.preprocessing import LabelEncoder \nfrom sklearn.model_selection import StratifiedKFold \nfrom sklearn.preprocessing import StandardScaler \nfrom sklearn.pipeline import Pipeline \nfrom keras import models \nfrom keras import layers ")
        f.write("\nseed = 7 \nnumpy.random.seed(seed)")
        f.write(f'\ndataframe = pandas.read_csv("Uploads/DeepLearning/{a}/{fileName}", header=None)\ndataset = dataframe.values')
        f.write(f'\nx = dataset[:,0:{int(NumberOfFeatures)}]')
        f.write(f'\ny = dataset[:,{int(NumberOfFeatures)}]')
        f.write("\ndef create_baseline():\n    from keras import layers\n    from keras import models \n    network = models.Sequential() ")
        for i in range(0,int(DenseNumberOfLayers)):
            if i == int(DenseNumberOfLayers)-1:
                f.write(f'\n    network.add(layers.Dense({c[i]}, activation="{EndLayerOutputFunction}"))')
            elif i == 0:
                f.write(f'\n    network.add(layers.Dense({c[i]}, activation="{BeginingLayersOutputFunction}",input_shape=({InputShape},)))')
            else:
                f.write(f'\n    network.add(layers.Dense({c[i]}, activation="{BeginingLayersOutputFunction}"))')
                
        f.write(f'\n    network.compile(optimizer="Adam",loss="{LossFunction}",metrics=["accuracy"]) \n    return network')
        f.write(f'\nestimator = KerasClassifier(build_fn=create_baseline, epochs={int(Epochs)}, batch_size={int(BatchSize)}, verbose=0)')
        f.write("\nkfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)")
        f.write("\nresults = cross_val_score(estimator, x, y, cv=kfold)")
        f.write(f'\npickle.dump(results[{int(Epochs)-1}],open("Uploads/{a}.p","wb"))')
        f.write("\nmodel = create_baseline()")
        f.write(f'\nmodel.save("Uploads/h5/{a}.h5")')
        f.write('\nfrom keras import backend as k \nk.clear_session()')
        f.close()
        exec(open(f'Uploads/{a}.py', "rb").read())
        z = pickle.load(open(f'Uploads/{a}.p',"rb"))
        r=cursor.find_one_and_update({"_id":result[0]["_id"]},{"$set":{"accuracy":f'{str(z*100)}'}})
        return str(z)    
    except :
        flash("Make sure everything is as per requirment" , "danger")
        return redirect(url_for('dlForm'))
    
@app.route('/Signout')
def signout():
    if "profile" in session:
        flash("You're sucessfully logged out...!" , "info")
        session.clear()
        return redirect(url_for('main'))
    else:
        return redirect('/')

@app.route('/download/<file>',methods = ['GET' , 'POST'])
def download(file):
    if "profile" in session:
        return send_file(f"E:/buildin/Uploads/h5/{file}.h5")


@app.route('/profile')
def profile():
    if "profile" in session:
        cursor = mongo.db.userRegistration
        data =  cursor.find_one({"email":session["profile"]})
        cursor = mongo.db.modelsInputs
        DL = cursor.find({"User":session["profile"] , "Model":"Deep Learning"}).count()
        if DL > 0:
            data["Dl"] = DL +1
        else:
            data['Dl'] = 0
        ML = cursor.find({"User":session["profile"] , "Model":"Machine Learning"}).count()
        if ML > 0:
            data["Ml"] = ML +1
        else:
            data['Ml'] = 0
        if DL + ML >0:
            data["Total"] = DL + ML + 2
        else :
            data['Total'] = 0
        return render_template('profile.html' , data=data)

@app.route('/updateProfile' , methods = ['GET' , 'POST'])
def updateProfile():
    if "profile" in session:
        form = request.form
        if 'image' in request.files:
            image = request.files['image']
            mongo.save_file(session['profile'] , image)
        cursor = mongo.db.userRegistration
        data =  cursor.find_one({"email":session["profile"]})
        if form['name']:
            name = form['name']
        else:
            name = data['name']
        try:
            gander = form['gender']    
        except :
            gander = data['gender']
        try:
            location = form['location']    
        except :
            location = data['location']
        cursor = mongo.db.userRegistration
        cursor.find_one_and_update({"email":session["profile"]},{"$set":{"name":f'{str(name)}',"gender":f'{str(gander)}',"location":f'{str(location)}'}})
        return redirect(url_for('profile'))
    else:
        redirect(url_for('profile'))
        
@app.route("/file/<filename>")
def file(filename):
    return mongo.send_file(filename)

@app.route("/search" , methods = ['GET' , 'POST'])
def search():
    if "profile" in session:
        data = request.form
        name = data['search']
        cursor = mongo.db.modelsInputs
        user_info = cursor.find({"Name":f'{name}'}) 
        if len(list(user_info))>0:
            user_info = cursor.find({"Name":f'{name}'})
            return render_template('search.html' ,user_info=user_info)
        else:
            flash('no data found' , "info")
            return redirect(url_for('dashboard'))
        

if __name__ == '__main__':
    app.run(debug=True , port=3000)