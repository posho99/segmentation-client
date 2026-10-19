from flask import Flask, render_template, request
import joblib

app=Flask(__name__)

model=joblib.load("kmeansModel.pkl")
scaler=joblib.load("scaler.pkl")

def interpretationCluster(cluster_id):
	interpretation={
		0:"Clients intermediaires stables",
		1:"Clients premium à fort pouvoir d'achat et forte consommation",
		2:"Clients impulsifs à faible revenu",
		3:"Clients intermediaires jeunes",
		4:"Clients aisés mais peu dépensiers",
		5:"Clients à faible revenu et faible engagement"
	}
	return interpretation.get(cluster_id,"Profil non identifié")

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
	age=float(request.form['age'])
	revenu=float(request.form['revenu'])
	depense=float(request.form['depense'])
	import numpy as np 
	X=np.array([[age, revenu, depense]])
	X_transformes=scaler.transform(X)
	clusters=int(model.predict(X_transformes)[0])
	profile=interpretationCluster(clusters)
	return render_template("resultat.html", cluster=clusters, profile=profile,
		age=age, revenu=revenu,depense=depense)

if __name__=="__main__":
	app.run(debug=True)
