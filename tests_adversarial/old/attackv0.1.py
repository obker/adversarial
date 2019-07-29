# Add your imports here
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend
from keras.models import load_model
from extract import load_data

import foolbox.attacks
from foolbox.criteria import TargetClassProbability
import foolbox.models


def untargeted_attack(model, imgs):
    """
    :param model: attacked model
    :param images: numpy array of orignial images with shape (10,28,28,1)

    return a numpy array with adversarial images
    """
    model = foolbox.models.KerasModel(model, bounds=(0, 1))
    target_class = 1

    adversarials = []
    for image in imgs:
        label_orig = np.argmax(model.predictions(image))

        if 0:
            criterion = TargetClassProbability(target_class, p=0.99)
            attack = foolbox.attacks.LBFGSAttack(model, criterion)
            adversarial = attack(image, label=label_orig)
        if 1:
            attack = foolbox.attacks.FGSM(model)
            adversarial = attack(image, label=label_orig)


        adversarials.append(adversarial)

    return adversarials

def check_eps(imgs, advs):
    dist = []
    for i in range(len(advs)):
        dist.append(np.absolute((imgs[i] - advs[i])).max())
    return dist

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

    #the 10 images to attack
    images =  x_test[:10]
    labels = y_test[:10]

    adversarials = untargeted_attack(keras_model, images)
    if 0:
        for img, adv in zip(images, adversarials):
            print_images(img, adv)
