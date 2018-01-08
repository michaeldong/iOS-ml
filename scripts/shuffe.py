import os
import shutil

from sklearn.cross_validation import train_test_split

origin_dir = "../origin_images"
image_dir = "../images/"

shutil.rmtree(image_dir)
os.mkdir(image_dir)


categorys = os.listdir(origin_dir)

for tag in categorys:
        filepath = os.path.join(origin_dir,tag)
        os.listdir(filepath)
        X = y = os.listdir(filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        os.makedirs(image_dir+"Train/"+tag,0755)
        os.makedirs(image_dir+"Val/"+tag,0755)
        for x in X_train:
           # print(x)
           # print(filepath)
           # os.rename(os.path.join(filepath,x) , os.path.join(image_dir+"Train/"+tag,x))
           shutil.copy(os.path.join(filepath,x),os.path.join(image_dir+"Train/"+tag,x))

        for x in X_test:
           shutil.copy(os.path.join(filepath,x),os.path.join(image_dir+"Val/"+tag,x))
           #os.rename(os.path.join(filepath,x) , os.path.join(image_dir+"Val/"+tag,x))
