from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

models = {
    "rf" : RandomForestClassifier(max_depth=10),
    "lsvc" : LinearSVC(multi_class="ovr",max_iter=1500)
}