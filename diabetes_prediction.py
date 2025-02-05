
""" Bazı yerlerde bu prediction işlemine scoring de denir. """


from utils import diabetes_data_prep
import pandas as pd
import joblib


new_df = pd.read_csv("machine_learning/datasets/diabetes.csv")   # Bunu yeni bir veri gibi düşünelim!!! Mesela yeni hastalar geldi diyebiliriz!!!

random_user = new_df.sample(1, random_state = 70)

new_model = joblib.load("voting_clf.pkl")

# print(new_model.predict(random_user))


X, y = diabetes_data_prep(new_df) 
# Yenii veri setimizi diabetes_data_prep fonksiyonuna sokuyoruz ki modelimiz onu anlayabilsin. Bu sayede yeni veri setimizi modelimiz anlayabilecek hale getiriyoruz.

Random_user = X.sample(1, random_state=18)


new_model = joblib.load("machine_learning/diabetes/voting_clf.pkl")

print("Our new random user is: ", Random_user, "\n\n", "He/She's diabetes result: \n" )

print(new_model.predict(Random_user), "\n\n")








