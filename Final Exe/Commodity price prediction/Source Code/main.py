from tkinter import *
import pricePrediction as pp
import os 
import threading
from threading import Thread
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
from tkinter import messagebox

window=Tk()
window.configure(background='#222222')
title_bar = Frame(window, bg='blue', relief='raised', bd=2)

window.geometry("1280x720")
window.wm_title("Commodity Price Prediction")
window.wm_iconbitmap("images\\science.ico")

#C = Canvas(window, bg="blue", height=1280, width=720)
filename = PhotoImage(file = "images\\india_agri.png")
background_label = Label(window, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
#C.pack()

header = Label(text="Commodity Stock Recommendations and Price Prediction using Prescriptive Analytics ",fg="#FFFFFF",bg="#222222")
header.configure(font=("Forte", 24))
header.place(x=60,y=30)

import tkinter as tk
RightTopFrame = tk.Frame(window,bg='#123456',width=500,height=100)
RightTopFrame.place(x=550,y=550)


commodityOptions = []

commodityOptions = os.listdir("data\\")
# datatype of menu text
commodityClicked = StringVar()
# initial menu text
commodityClicked.set( "Select Commodity" )
# Create Dropdown menu
commodityDrop = OptionMenu( RightTopFrame , commodityClicked , *commodityOptions )
commodityDrop.place(x=20,y=10)
commodityDrop.configure(font=("Cambria", 10),bg="#ffffff")


#------------ Select year ------------

yearOptions = ["Select Commodity First"]

# datatype of menu text
yearClicked = StringVar()
# initial menu text
yearClicked.set( "Select Year" )
# Create Dropdown menu
yearDrop = OptionMenu( RightTopFrame , yearClicked , *yearOptions )
yearDrop.place(x=370,y=10)
yearDrop.configure(font=("Cambria", 10),bg="#ffffff")




comparisonOptions = ["Select Commodity First"]

comparisonClicked = StringVar()

comparisonClicked.set( "Select City" )

def getCity():
    comparisonOptions.clear()
    #yearOptions.clear()

    record = pd.read_csv("data\\"+commodityClicked.get())
    

    for city in record['mkt_name'].unique():
        comparisonOptions.append(city)

    comparisonDrop = OptionMenu( RightTopFrame , comparisonClicked , *comparisonOptions )
    comparisonDrop.place(x=200,y=10)
    comparisonDrop.configure(font=("Cambria", 10),bg="#ffffff")
    comparisonClicked.set( "Select City" )

    


def getYear():
    

    record_years = pd.read_csv("data\\"+commodityClicked.get())

    str1 = 'mkt_name == "'+comparisonClicked.get()+'"'
    record_years.query(str1).to_csv("temp\\years.csv", index=False)

    record = pd.read_csv("temp\\years.csv")
    
    for years in record['mp_year'].unique():
        yearOptions.append(years)
    
    yearDrop = OptionMenu( RightTopFrame , yearClicked , *yearOptions )
    yearDrop.place(x=370,y=10)
    yearDrop.configure(font=("Cambria", 10),bg="#ffffff")
    yearClicked.set("Select Year")
    yearOptions.clear()



commodityClicked.trace("w", lambda *args: getCity())

comparisonClicked.trace("w", lambda *args: getYear())


#comparisonOptions = ['accuracy','fscore','precision','recall']




comparisonDrop = OptionMenu( RightTopFrame , comparisonClicked , *comparisonOptions )
comparisonDrop.place(x=200,y=10)
comparisonDrop.configure(font=("Cambria", 10),bg="#ffffff")

upcomingPredicitions = Text(window, height = 7, width = 55)




def startAnalysis():
    comparison = comparisonClicked.get()
    commodity =  commodityClicked.get()
    year = yearClicked.get()

    file = 'temp\\out.csv'
    if(os.path.exists(file) and os.path.isfile(file)):
        os.remove(file)
        #print("file deleted")
    else:
        print("file not found")

    

    df = pd.read_csv("data\\"+commodityClicked.get())
    #df[(df['mkt_name'] == comparison)].to_csv("temp\\out.csv", index=False)
    str1 = 'mkt_name == "'+comparison
    str2= '" & mp_year =='+year
    df.query(str1+str2).to_csv("temp\\out.csv", index=False)

    df_out = pd.read_csv("temp\\out.csv")
    unitofdata = df_out['um_name'].unique()
    
    pred = pp.PredictData(commodity,comparison)

    data1,data2,scores,algoName,stock_data = pred.prediction()

    print("Highest Score "+str(data1))
    #print("Upcoming price "+str(data2))
    #print(data2.shape)

    
    upcomingPredicitions.delete('1.0', END)
    upcomingPredicitions.place(x=10,y=555)
    upcomingPredicitions.configure(font=("Cambria", 12,"bold"),bg="#333333",fg="#FFFFFF")
    upcomingPredicitions.insert(tk.END, "\t"+str(algoName)+" has Highest accuracy of "+str(data1) +"%" +"\n\n")
    upcomingPredicitions.insert(tk.END, "Upcoming Price & Stock for upcoming months :"+"\n\n")
    try :
        upcomingPredicitions.insert(tk.END, "Prices (₹): "+str(data2[0:6].tolist())+"\n")
        upcomingPredicitions.insert(tk.END, "Stocks (per-"+unitofdata+"): " +str(stock_data[0:6].tolist()))
    except:
        print("Insufficient Data")
    #print(data2)
    #print(stock_data)
    plot(scores,comparison)
    if str(data1) == "0.0" or str(data1) == "0":
        messagebox.showwarning("showwarning", "Data is Insufficient \n So the Predicted values may be incorrect")


def close():
    #messagebox.showinfo(message='Tkinter is reacting.')
    window.destroy()
    exit()


def plot(scores,comparison):
  
    data1 = {'Algorithms': ["Logistic R","Naive B","SVM","KNN","Decision T","XGBoost"],
         "Accuracy": scores
        }
    df1 = DataFrame(data1,columns=['Algorithms','Accuracy'])

    figure1 = plt.Figure(figsize=(5,4), dpi=100)
    ax1 = figure1.add_subplot(211)
    #bar1 = FigureCanvasTkAgg(figure1, window)
    #bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df1 = df1[['Algorithms','Accuracy']].groupby('Algorithms').sum()
    df1.plot(kind='bar', legend=False, ax=ax1)
    ax1.set_title('Algorithms Vs. Accuracy')
  
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    
    canvas = FigureCanvasTkAgg(figure1,
                               master = window) 

    canvas.get_tk_widget().delete()
    canvas.draw()
  
    # placing the canvas on the Tkinter window
  
    # creating the Matplotlib toolbar
    #toolbar = NavigationToolbar2Tk(canvas,
    #                               window)
    #toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().place(x=10,y=100)


    
    
    


startButton = Button(RightTopFrame,text="Start Analysis",bd=1,bg="#123456",fg="#FFFFFF",command=lambda:Thread(target = startAnalysis).start())
startButton.place(x=200,y=60)
startButton.configure(font=("Cambria", 10,'bold'))
window.protocol('WM_DELETE_WINDOW', close) 

window.mainloop()
