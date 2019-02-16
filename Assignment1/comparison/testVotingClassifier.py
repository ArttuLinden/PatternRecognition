from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from own_functions import loadData,getRelevantData,getFeatures
from sklearn.model_selection import GridSearchCV
from os import getcwd
from sklearn.model_selection import GroupShuffleSplit

folder = getcwd() +'/robotsurface/'
data = loadData(folder)

X = data[0]
y = data[1]
groups = data[2]

X = getRelevantData(X,'vel_acc')
X_feat = getFeatures(X,'fftlog10')

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                           random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

cv = GroupShuffleSplit(n_splits=30,test_size=0.2,random_state=0)
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=cv)
grid = grid.fit(X_feat, y, groups)