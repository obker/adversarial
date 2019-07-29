# Add your imports here
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend
from keras.models import load_model
from keras.optimizers import Adam
from extract import load_data
import time
from numpy import linalg as LA

import foolbox
import pandas as pd



def untargeted_attack(model, imgs, labels, attack):

    nb_img = len(imgs)
    adversarials = np.empty((nb_img, 28, 28, 1))
    index = 0
    debut = time.time()
    
    for img in imgs:
        adversarial = attack(input_or_adv=img, label=labels[index])
        adversarials[index,:,:,:] = adversarial
        index+=1
    
    fin = time.time()
    temps = fin - debut

    return adversarials, temps


def tests(model, x_test, y_test, nombreImages):
    
    images =  x_test[:nombreImages]
    labels = y_test[:nombreImages]
    fmodel = foolbox.models.KerasModel(model= model, bounds= (0,1))
    attacks = [] 
    attacks_name = []
    critere=[]
    graph_datas=[]
    tempsExec = []
    
    attack0 = foolbox.attacks.FGSM(model=fmodel, criterion = foolbox.criteria.Misclassification(), distance = foolbox.distances.Linfinity)
    attacks.append(attack0)
    attacks_name.append("FGSM")
    critere.append("Linf")
    
    attack1 = foolbox.attacks.PGD(model=fmodel, criterion = foolbox.criteria.Misclassification(), distance = foolbox.distances.Linfinity)
    attacks.append(attack1)
    attacks_name.append("PGD")
    critere.append("Linf")
    
    attack3 = foolbox.attacks.DeepFoolL2Attack(model=fmodel, criterion = foolbox.criteria.Misclassification(), distance = foolbox.distances.MeanSquaredDistance)
    attacks.append(attack3)
    attacks_name.append("DeepFoolL2")
    critere.append("L2")
    
    
    #################################
    #feed the diferent attacks here#
    ################################
    

    for i in range(len(attacks)):
        print(i)
        attack = attacks[i]
        adversarials,tps = untargeted_attack(model, images, labels, attack)
        x,y = generate_epsilon(images, adversarials)
        graph_datas.append((x,y))
        tempsExec.append(tps)
        
        d1 = {'x_Linf': x[0]}
        d2 = {'x_L2': x[1]}
        d3 = {'y_Linf': y[0]}
        d4 = {'y_L2': y[1]}
        
        df1 = pd.DataFrame(data=d1)
        df2 = pd.DataFrame(data=d2)
        df3 = pd.DataFrame(data=d3)
        df4 = pd.DataFrame(data=d4)
        
        df1.to_csv(attacks_name[i]+'_xinf.csv')
        df2.to_csv(attacks_name[i]+'_x2.csv')
        df3.to_csv(attacks_name[i]+'_yinf.csv')
        df4.to_csv(attacks_name[i]+'_y2.csv')
        
        
        print("********************")
        print(attacks_name[i])
        print("Optimisation " + critere[i])
        print("====================")
        print("Données (temps : " + str(tempsExec[i])+"s) :")
        print('cf CSV file')
        print("********************")
        print()
        

#Use this function to generate epsilon variations according to different norms. For now l2 and linf
def generate_epsilon(imgs, advs):
    dist_inf = []
    dist_2 = []
    
    for i in range(len(advs)):
        bruit = imgs[i] - advs[i]
        dist_inf.append(np.absolute(bruit).max())
        dist_2.append(LA.norm(bruit))
     
    dists = [dist_inf, dist_2]
    tauxx = []
    for i in range(len(dists)):
        dist = list(map(lambda x: round(x,3), dists[i]))
        
        compte = {}.fromkeys(set(dist),0)
        for valeur in dist:
            compte[valeur] += 1
        
        dist  = sorted(list(set(dist)))
        dists[i] = dist
        
        taux = []
        a = dist[0]
        taux.append(compte[a])
        for i in range(1,len(dist)):
            taux.append(taux[-1]+compte[dist[i]])
        taux = list(map(lambda x : x/len(taux), taux))
        tauxx.append(taux)
                            
    return dists,tauxx



#Use this function to compare image labels
def compare_image_labels(imgs, advs):
    model = load_model('model.h5')
    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=0.001),metrics=['accuracy'])

    for i in range(len(advs)):
        #Predicting the  original image
        pred = model.predict_classes(imgs[i:i+1], batch_size=10)
        print (pred)

        #Predicting the adversarial Image
        pred = model.predict_classes(advs[i:i+1], batch_size=10)
        print (pred)

#Use this function to print your images
def print_images(img, adv):
    plt.subplot(1,2,1)
    plt.title('Original')
    img = img.reshape(28, 28)
    plt.imshow(img,cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Adversarial')
    adv = adv.reshape(28, 28)
    plt.imshow(adv,cmap='gray')
    plt.axis('off')
    plt.show()


######################################MAIN######################################
if __name__ == "__main__":
    # Loading Model
    backend.set_learning_phase(False)
    keras_model=load_model('model.h5')

    # Loading MNIST Dataset
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    tests(keras_model, x_test, y_test, len(x_test))
    #adversarials = untargeted_attack(keras_model, images)
    #print(check_eps(images, adversarials))
