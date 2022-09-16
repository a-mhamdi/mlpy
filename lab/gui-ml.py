# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:54:50 2020
"""
import tkinter as tk

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler
from sklearn import cluster, datasets

from scipy.spatial import distance


# Fonction pour l'affichage
def affichage_Classes(Data,LesLabels,Titre):
    fig = Figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(Data[:,0], Data[:,1], c=LesLabels)
    ax.set_title(Titre)
    global canvas_fig
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.get_tk_widget().pack()
    canvas_fig.draw()
    
# Fonction pour l'affichage
def affichage_Classes_pred(Data,LesLabels,Titre):
    fig = Figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(Data[:,0], Data[:,1], c=LesLabels)
    ax.set_title(Titre)
    global canvas_fig1
    canvas_fig1 = FigureCanvasTkAgg(fig, master=frame1)
    canvas_fig1.get_tk_widget().pack()
    canvas_fig1.draw()
    
def clear_fig():
    global canvas_fig
    canvas_fig.get_tk_widget().pack_forget()

def clear_fig1():
    global canvas_fig1
    canvas_fig1.get_tk_widget().pack_forget()

# Génération de l'ensemble de données 
def gener_Data():
    X, y  = datasets.make_blobs(varEch.get(),centers=varClass.get(),random_state=0)
    return X, y

# Fonction pour la catégorisation avec le k-means
def Clustering_kmeans():
    Data = gener_Data()[0]
    kmeans = KMeans(n_clusters=varClass.get())
    kmeans.fit(Data)
    y_km = kmeans.fit_predict(Data)
    return y_km

# Fonction pour la catégorisation avec la CHA
def Clustering_CHA():
    Data = gener_Data()[0]
    CHA = AgglomerativeClustering(n_clusters=varClass.get(), affinity = 'euclidean', linkage = 'ward')
    y_hc=CHA.fit_predict(Data)
    return y_hc

# Fonction pour la catégorisation avec le DBSCAN    
def Clustering_DBSCAN():
    Data = gener_Data()[0]
    dbscan = cluster.DBSCAN(eps=.15,min_samples=4)
    y_dbscan=dbscan.fit_predict(Data);
    return y_dbscan

# Fonction qu permet de calculer les centres des classes
def ComputeClusterCenters(labels_pred):
    Data = gener_Data()[0]
    CentresClasses=np.zeros((varClass.get(),2), dtype='i')
    for i in range(0,varClass.get()):
        UneClasse=Data[labels_pred==i,:]
        CentresClasses[i,:]=sum(UneClasse)/UneClasse.shape[0]
    return(CentresClasses)

# Fonction qui permet de calculer L'inertie_Intraclasses
def Inertie_IntraClasses(labels_pred,CentresClasses):
    Data = gener_Data()[0]
    nbreC=CentresClasses.shape[0]
    nbreO=Data.shape[0]
    sum=0
    for i in range(0,nbreC):    
        UneClasse=Data[labels_pred==i,:]
        for j in range(0,UneClasse.shape[0]):
            sum=sum+np.sqrt(distance.euclidean(UneClasse[j,:],CentresClasses[i,:]))
    return(sum/nbreO)
               
# Fonction qui permet de calculer L'inertie_Interclasses
def Inertie_InterClasses(labels_pred,CentresClasses):
    Data = gener_Data()[0]
    nbreC=CentresClasses.shape[0]
    Centre=np.mean(Data)
    nbreO=Data.shape[0]
    sum=0
    for i in range(0,nbreC):    
            sizeClasse=Data[labels_pred==i,:].shape[0]
            sum=sum+(np.sqrt(distance.euclidean(CentresClasses[i,:],Centre))/sizeClasse)
    return(sum/nbreO)

root = tk.Tk()
root.title('Machine Learning')
root.resizable(False, False)
tit = tk.Label(root, text="Classificateur", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=700, bg='grey')
canvas.pack()

frame = tk.Frame(root, bg='grey')
frame.place(relwidth=0.4, relheight=0.75, relx=0.1, rely=0.1)

frame1 = tk.Frame(root, bg='grey')
frame1.place(relwidth=0.4, relheight=0.75, relx=0.55, rely=0.1)

varEchLabel = tk.StringVar(frame,value="Echantillons")
labelEch = tk.Label(root,textvariable=varEchLabel)
labelEch.config(width=15, font=("", 12))
labelEch.pack(side=tk.LEFT)

varEch = tk.IntVar(frame,value=500)
objEch = tk.Entry(root,textvariable=varEch)
objEch.config(width=10, font=("", 12))
objEch.pack(side=tk.LEFT)

varLabel = tk.StringVar(frame,value="Classes")
labelClass = tk.Label(root,textvariable=varLabel)
labelClass.config(width=15, font=("", 12))
labelClass.pack(side=tk.LEFT)

varClass = tk.IntVar(frame,value=2)
objClass = tk.Entry(root,text="Classificateur",textvariable=varClass)
objClass.config(width=10, font=("", 12))
objClass.pack(side=tk.LEFT)

label = tk.Label(root)
label.config(width=5, font=("", 12))
label.pack(side=tk.LEFT)

OptionList = ["DBSCAN", "k-Means", "CHA"] 

algo_select = tk.StringVar(frame)
algo_select.set(OptionList[0])

opt = tk.OptionMenu(frame1, algo_select, *OptionList)
opt.config(width=90, font=("", 12))
opt.pack(side=tk.TOP)

affich_Data = tk.Button(root, text='Générer données',
                        padx=35, pady=10,
                        fg="white", bg="grey")
affich_Data.pack(side=tk.LEFT)

gener_Data()
affichage_Classes(gener_Data()[0],gener_Data()[1],"Jeu de données")
affichage_Classes_pred(gener_Data()[0],Clustering_DBSCAN(),"DBSCAN")

affich_Data['command'] = lambda: [clear_fig(), gener_Data(),affichage_Classes(
    gener_Data()[0],gener_Data()[1],"Jeu de données")]

varIntraLabel = tk.StringVar(frame)
labelIntra = tk.Label(frame,textvariable=varIntraLabel)
labelIntra.config(width=60, font=("", 12))
labelIntra.pack(side=tk.BOTTOM)

varInterLabel = tk.StringVar(frame1)
labelInter = tk.Label(frame1,textvariable=varInterLabel)
labelInter.config(width=60, font=("", 12))
labelInter.pack(side=tk.BOTTOM)

def callback():
    if algo_select.get() == OptionList[0]:
        clear_fig1()
        affichage_Classes_pred(gener_Data()[0],Clustering_DBSCAN(),"DBSCAN")
        varIntraLabel.set("Inertie intraclasses : {}".format(
    Inertie_IntraClasses(Clustering_DBSCAN(),ComputeClusterCenters(Clustering_DBSCAN()))))
        varInterLabel.set("Inertie interclasses : {}".format(
    Inertie_InterClasses(Clustering_DBSCAN(),ComputeClusterCenters(Clustering_DBSCAN()))))
    elif algo_select.get() == OptionList[1]:
        clear_fig1()
        affichage_Classes_pred(gener_Data()[0],Clustering_kmeans(),"k-Means")
        varIntraLabel.set("Inertie intraclasses : {}".format(
    Inertie_IntraClasses(Clustering_kmeans(),ComputeClusterCenters(Clustering_kmeans()))))
        varInterLabel.set("Inertie interclasses : {}".format(
    Inertie_InterClasses(Clustering_kmeans(),ComputeClusterCenters(Clustering_kmeans()))))
    else:
        clear_fig1()
        affichage_Classes_pred(gener_Data()[0],Clustering_CHA(),"CHA")
        varIntraLabel.set("Inertie intraclasses : {}".format(
    Inertie_IntraClasses(Clustering_CHA(),ComputeClusterCenters(Clustering_CHA()))))
        varInterLabel.set("Inertie interclasses : {}".format(
    Inertie_InterClasses(Clustering_CHA(),ComputeClusterCenters(Clustering_CHA()))))

class_Data = tk.Button(root, text='Classer données',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=callback)
class_Data.pack(side=tk.LEFT)

root.mainloop()


