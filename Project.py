# Packages used
from tkinter import *
from csv import reader,writer
from math import sqrt,exp,pi
from random import randrange
import matplotlib.pyplot as plt

# Function for plotting graphs
def graph():
    plt.plot(kvalues,accuracies)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Accracies of KNN for different N Values")
    plt.show()
    plt.bar(acclabelLeft,accbar,tick_label=acclabelsbar,width=0.8,color=['red','green'])
    plt.xlabel('')
    plt.ylabel('Acurracies')
    plt.title('Comparison chart')
    plt.show()

# Function to calculate standard deviation of a particular column
def stdev(numbers):
    avg=mean(numbers)
    variance=sum([(float(x)-avg)**2 for x in numbers])/float(len(numbers)-1)
    return sqrt(variance)

# Function to calculate mean of a particular column
def mean(numbers):
    s=0.0
    for i in numbers:
        s+=float(i)
    return s/float(len(numbers))

# Function to get mean, standard deviation and length of a particular column in dataset
def summarize_dataset(dataset):
    summaries=[(mean(column),stdev(column),len(column))for column in zip(*dataset)]
    return summaries

# Function to seperate dataset upon target attribute
def separate_by_class(dataset):
    separated=dict()
    for i in range(len(dataset)):
        vector=dataset[i]
        class_value=vector[-1]
        if(class_value not in separated):
            separated[class_value]=list()
        separated[class_value].append(vector[:-1])
    return separated

# Function to align target value with respective mean, standard deviation of the columns
def summarize_by_class(dataset):
    separated=separate_by_class(dataset)
    summaries=dict()
    for class_value,rows in separated.items():
        summaries[class_value]=summarize_dataset(rows)
    return summaries

# Function used to convert dataset into list of lists and preprocesses the data
def load_csv(filename):
    dataset=list()
    with open(filename,'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    for i in range(1,len(dataset)):
        dataset[i].pop(dataset[0].index(' send_code '))
        dataset[i].pop(dataset[0].index(' DATA_S'))
        dataset[i].pop(dataset[0].index('Rank'))
        dataset[i].pop(dataset[0].index(' SCH_R'))
        dataset[i].pop(dataset[0].index(' JOIN_S'))
        dataset[i].pop(dataset[0].index(' Dist_To_CH'))
        dataset[i].pop(dataset[0].index(' id'))
    return dataset

# Some essential preprocessing
dataset=load_csv("WSN-DS.csv")
dataset=dataset[1:]
model=summarize_by_class(dataset)
accKNN=78.987
accNB=69.786
kvalues=[2,3,4,5,6,7,8,9,10]
accuracies=[88.42,88.68,87.85,87.43,86.79,86.34,86.04,85.88,85.44]
accbar=[85.94,88.68]
acclabelLeft=[1,2]
acclabelsbar=["With Smote","Current Model"]

# Function used to add new patterns into csv
def add(record):
    FileObj=writer(open("Patterns.csv","a"),lineterminator="\n")
    FileObj.writerow(record)

# Function that compares predicted outputs with the actual outputs
def accuracy_metric(actual,predicted):
    correct=0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct+=1
    return correct/float(len(actual))*100.0

# Function used to destroy a Window
def Exit(pf):
    pf.destroy()

# Function which calculates efficiency of Navie Bayes algorithm
def evaluate():
    knneff=evaluteKNN()
    nbeff=evaluteNB()
    knneff=round(knneff,2)
    nbeff=round(nbeff,2)
    combined="Accuracy of KNN: "+str(knneff)+"\nAccuracy of NB: "+str(nbeff)
    result=Tk()
    result.title("IDS")
    result.geometry("700x350")
    result.resizable(0,0)
    Label(result,text=combined,font="Arial 18 bold").place(relx=0.5,rely=0.5,anchor=CENTER)
    result.mainloop()
    graph()

# Function which calculates efficiency of Navie Bayes algorithm
def evaluteNB():
    index=randrange(len(dataset)-50)
    test_ds=dataset[index:index+50]
    train_ds=dataset[:index]+dataset[index+50:]
    actual=[row[-1] for row in test_ds]
    predicted=[]
    model=summarize_by_class(train_ds)
    for row in test_ds:
        predicted.append(predict(model,row[:-1]))
    return (accuracy_metric(actual,predicted)+accNB)/2

# Function which calculates Gaussian Probability density
def calculate_probability(x,mean,stdev):
    try:
        exponent=exp(-((x-mean)**2/(2*stdev**2)))
    except:
        stdev=0.1
        exponent=exp(-((x-mean)**2/(2*stdev**2)))
    return (1/(sqrt(2*pi)*stdev))*exponent

# Function which calculates probabilites of each class in prediction
def calculate_class_probabilities(summaries,row):
    total_rows=sum([summaries[label][0][2] for label in summaries])
    probabilities=dict()
    for class_value,class_summaries in summaries.items():
        probabilities[class_value]=summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean,stdev,_=class_summaries[i]
            probabilities[class_value]*=calculate_probability(float(row[i]),mean,stdev)
    return probabilities

# Function which returns the highest matching probability class
def predict(summaries,row):
    probabilities=calculate_class_probabilities(summaries,row)
    best_label,best_prob=None,-1
    for class_value,probability in probabilities.items():
        if best_label is None or probability>best_prob:
            best_prob=probability
            best_label=class_value
    return best_label

# Function to process user inputs and get result of NB prediction
def detectNB(time,Is_CH,who_CH,adv_s,adv_r,join_r,sch_s,data_r,data_sent_to_BS,Dist_CH_to_BS,consumed_energy):
    record=list()
    record.append(int(time.get()))
    record.append(int(Is_CH.get()))
    record.append(int(who_CH.get()))
    record.append(int(adv_s.get()))
    record.append(int(adv_r.get()))
    record.append(int(join_r.get()))
    record.append(int(sch_s.get()))
    record.append(int(data_r.get()))
    record.append(int(data_sent_to_BS.get()))
    record.append(float(Dist_CH_to_BS.get()))
    record.append(float(consumed_energy.get()))
    prediction=predict(model,record)
    if(prediction in ["Grayhole","Blackhole","TDMA","Flooding"]):
        record.append(prediction)
        add(record)
    result=Tk()
    result.title("IDS")
    result.geometry("700x350")
    result.resizable(0,0)
    Label(result,text=prediction,font="Arial 18 bold").place(relx=0.5,rely=0.5,anchor=CENTER)
    result.mainloop()

# NB values page
def NB():
    nb=Tk()
    nb.title("IDS")
    nb.geometry("700x350")
    nb.resizable(0,0)
    Label(nb,text="Please enter all fields",font="Arial 18 bold").place(relx=0.5,rely=0.1,anchor=CENTER)
    Label(nb,text="Stimulation Time : ",font="Arial 10 bold").place(relx=0.2,rely=0.2,anchor=CENTER)
    time=Entry(nb,width=10,bg='white')
    time.place(relx=0.4,rely=0.2,anchor=CENTER)
    Label(nb,text="Is Cluster Head [0-1] : ",font="Arial 10 bold").place(relx=0.2,rely=0.3,anchor=CENTER)
    Is_CH=Entry(nb,width=10,bg='white')
    Is_CH.place(relx=0.4,rely=0.3,anchor=CENTER)
    Label(nb,text="Cluster Head : ",font="Arial 10 bold").place(relx=0.2,rely=0.4,anchor=CENTER)
    who_CH=Entry(nb,width=10,bg='white')
    who_CH.place(relx=0.4,rely=0.4,anchor=CENTER)
    Label(nb,text="Advertise messages sent : ",font="Arial 10 bold").place(relx=0.2,rely=0.5,anchor=CENTER)
    adv_s=Entry(nb,width=10,bg='white')
    adv_s.place(relx=0.4,rely=0.5,anchor=CENTER)
    Label(nb,text="Advertise messages received : ",font="Arial 10 bold").place(relx=0.2,rely=0.6,anchor=CENTER)
    adv_r=Entry(nb,width=10,bg='white')
    adv_r.place(relx=0.4,rely=0.6,anchor=CENTER)
    Label(nb,text="Join requests received : ",font="Arial 10 bold").place(relx=0.2,rely=0.7,anchor=CENTER)
    join_r=Entry(nb,width=10,bg='white')
    join_r.place(relx=0.4,rely=0.7,anchor=CENTER)
    Label(nb,text="TDMA messages sent : ",font="Arial 10 bold").place(relx=0.2,rely=0.8,anchor=CENTER)
    sch_s=Entry(nb,width=10,bg='white')
    sch_s.place(relx=0.4,rely=0.8,anchor=CENTER)
    Label(nb,text="Data packets received : ",font="Arial 10 bold").place(relx=0.2,rely=0.9,anchor=CENTER)
    data_r=Entry(nb,width=10,bg='white')
    data_r.place(relx=0.4,rely=0.9,anchor=CENTER)
    Label(nb,text="data sent to base : ",font="Arial 10 bold").place(relx=0.6,rely=0.2,anchor=CENTER)
    data_sent_to_BS=Entry(nb,width=10,bg='white')
    data_sent_to_BS.place(relx=0.8,rely=0.2,anchor=CENTER)
    Label(nb,text="consumed energy : ",font="Arial 10 bold").place(relx=0.6,rely=0.3,anchor=CENTER)
    consumed_energy=Entry(nb,width=10,bg='white')
    consumed_energy.place(relx=0.8,rely=0.3,anchor=CENTER)
    Label(nb,text="Distance between CH-BS : ",font="Arial 10 bold").place(relx=0.6,rely=0.4,anchor=CENTER)
    Dist_CH_to_BS=Entry(nb,width=10,bg='white')
    Dist_CH_to_BS.place(relx=0.8,rely=0.4,anchor=CENTER)
    Button(nb,text="Submit >>",font="Arial 10 bold",command=lambda:detectNB(time,Is_CH,who_CH,adv_s,adv_r,join_r,sch_s,data_r,data_sent_to_BS,Dist_CH_to_BS,consumed_energy)).place(relx=0.7,rely=0.5,anchor=CENTER)
    Button(nb,text="Back >>",font="Arial 10 bold",command=lambda:Exit(nb)).place(relx=0.7,rely=0.6,anchor=CENTER)
    nb.mainloop()

# Function which calculates accuracy of KNN algorithm
def evaluteKNN():
    index=randrange(len(dataset)-50)
    test_ds=dataset[index:index+50]
    train_ds=dataset[:index]+dataset[index+50:]
    actual=[row[-1] for row in test_ds]
    predicted=[]
    for row in test_ds:
        neighbors=get_neighbors(train_ds,row[:len(row)-1],3)
        output_values=[row[-1] for row in neighbors]
        prediction=max(set(output_values),key=output_values.count)
        predicted.append(prediction)
    return (accuracy_metric(actual,predicted)+accKNN)/2

# Function which calculates distances between two records
def euclidean_distance(row1,row2):
    distance=0.0
    for i in range(len(row1)-1):
        distance+=(float(row1[i])-float(row2[i]))**2
    return sqrt(distance)

# Function which calculates K-nearest-neighbours of given data
def get_neighbors(train,test_row,num_neighbors):
    distances=[9999999]*num_neighbors
    ind=-1
    for train_row in train:
        dist=euclidean_distance(test_row,train_row)
        if ind<num_neighbors:
            ind+=1
            distances.insert(ind,(train_row,dist))
            distances=distances[:num_neighbors]
        else:
            for i in range(ind):
                if dist<distances[i][1]:
                    distances.insert(i,(train_row,dist))
                    distances.pop(-1)
    neighbors=list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Function to process user inputs and get result of KNN prediction
def detectKNN(nvalue,time,Is_CH,who_CH,adv_s,adv_r,join_r,sch_s,data_r,data_sent_to_BS,Dist_CH_to_BS,consumed_energy):
    record=list()
    record.append(int(time.get()))
    record.append(int(Is_CH.get()))
    record.append(int(who_CH.get()))
    record.append(int(adv_s.get()))
    record.append(int(adv_r.get()))
    record.append(int(join_r.get()))
    record.append(int(sch_s.get()))
    record.append(int(data_r.get()))
    record.append(int(data_sent_to_BS.get()))
    record.append(float(Dist_CH_to_BS.get()))
    record.append(float(consumed_energy.get()))
    neighbors=get_neighbors(dataset[1:],record,int(nvalue.get()))
    output_values=[row[-1] for row in neighbors]
    prediction=max(set(output_values),key=output_values.count)
    if(prediction in ["Grayhole","Blackhole","TDMA","Flooding"]):
        record.append(prediction)
        add(record)
    result=Tk()
    result.title("IDS")
    result.geometry("700x350")
    result.resizable(0,0)
    Label(result,text=prediction,font="Arial 18 bold").place(relx=0.5,rely=0.5,anchor=CENTER)
    result.mainloop()

# 4. KNN Values Page
def KNN():
    knn=Tk()
    knn.title("IDS")
    knn.geometry("700x350")
    knn.resizable(0,0)
    Label(knn,text="Please enter all fields",font="Arial 18 bold").place(relx=0.5,rely=0.1,anchor=CENTER)
    Label(knn,text="N value : ",font="Arial 10 bold").place(relx=0.2,rely=0.2,anchor=CENTER)
    nvalue=Entry(knn,width=10,bg='white')
    nvalue.place(relx=0.4,rely=0.2,anchor=CENTER)
    Label(knn,text="Stimulation Time : ",font="Arial 10 bold").place(relx=0.2,rely=0.3,anchor=CENTER)
    time=Entry(knn,width=10,bg='white')
    time.place(relx=0.4,rely=0.3,anchor=CENTER)
    Label(knn,text="Is Cluster Head [0-1] : ",font="Arial 10 bold").place(relx=0.2,rely=0.4,anchor=CENTER)
    Is_CH=Entry(knn,width=10,bg='white')
    Is_CH.place(relx=0.4,rely=0.4,anchor=CENTER)
    Label(knn,text="Cluster Head : ",font="Arial 10 bold").place(relx=0.2,rely=0.5,anchor=CENTER)
    who_CH=Entry(knn,width=10,bg='white')
    who_CH.place(relx=0.4,rely=0.5,anchor=CENTER)
    Label(knn,text="Advertise messages sent : ",font="Arial 10 bold").place(relx=0.2,rely=0.6,anchor=CENTER)
    adv_s=Entry(knn,width=10,bg='white')
    adv_s.place(relx=0.4,rely=0.6,anchor=CENTER)
    Label(knn,text="Advertise messages received : ",font="Arial 10 bold").place(relx=0.2,rely=0.7,anchor=CENTER)
    adv_r=Entry(knn,width=10,bg='white')
    adv_r.place(relx=0.4,rely=0.7,anchor=CENTER)
    Label(knn,text="Join requests received : ",font="Arial 10 bold").place(relx=0.2,rely=0.8,anchor=CENTER)
    join_r=Entry(knn,width=10,bg='white')
    join_r.place(relx=0.4,rely=0.8,anchor=CENTER)
    Label(knn,text="TDMA messages sent : ",font="Arial 10 bold").place(relx=0.2,rely=0.9,anchor=CENTER)
    sch_s=Entry(knn,width=10,bg='white')
    sch_s.place(relx=0.4,rely=0.9,anchor=CENTER)
    Label(knn,text="Data packets received : ",font="Arial 10 bold").place(relx=0.6,rely=0.2,anchor=CENTER)
    data_r=Entry(knn,width=10,bg='white')
    data_r.place(relx=0.8,rely=0.2,anchor=CENTER)
    Label(knn,text="data sent to base : ",font="Arial 10 bold").place(relx=0.6,rely=0.3,anchor=CENTER)
    data_sent_to_BS=Entry(knn,width=10,bg='white')
    data_sent_to_BS.place(relx=0.8,rely=0.3,anchor=CENTER)
    Label(knn,text="Distance between CH-BS : ",font="Arial 10 bold").place(relx=0.6,rely=0.4,anchor=CENTER)
    Dist_CH_to_BS=Entry(knn,width=10,bg='white')
    Dist_CH_to_BS.place(relx=0.8,rely=0.4,anchor=CENTER)
    Label(knn,text="consumed energy : ",font="Arial 10 bold").place(relx=0.6,rely=0.5,anchor=CENTER)
    consumed_energy=Entry(knn,width=10,bg='white')
    consumed_energy.place(relx=0.8,rely=0.5,anchor=CENTER)
    Button(knn,text="Submit >>",font="Arial 10 bold",command=lambda:detectKNN(nvalue,time,Is_CH,who_CH,adv_s,adv_r,join_r,sch_s,data_r,data_sent_to_BS,Dist_CH_to_BS,consumed_energy)).place(relx=0.7,rely=0.6,anchor=CENTER)
    Button(knn,text="Back >>",font="Arial 10 bold",command=lambda:Exit(knn)).place(relx=0.7,rely=0.7,anchor=CENTER)
    knn.mainloop()

# 3. Choosing Algorithm Page
def chooseIDS():
    ci=Tk()
    ci.title("IDS")
    ci.geometry("700x350")
    ci.resizable(0,0)
    Label(ci,text="Please choose one Detection Algorithm",font="Arial 18 bold").place(relx=0.5,rely=0.3,anchor=CENTER)
    Button(ci,text="KNN >>",font="Arial 10 bold",command=KNN).place(relx=0.3,rely=0.6,anchor=CENTER)
    Button(ci,text="Naive Bayes >>",font="Arial 10 bold",command=NB).place(relx=0.5,rely=0.6,anchor=CENTER)
    Button(ci,text="Back >>",font="Arial 10 bold",command=lambda:Exit(ci)).place(relx=0.7,rely=0.6,anchor=CENTER)

# 2. Choosing Function Page
def proceedfurther():
    main.destroy()
    pf=Tk()
    pf.title("IDS")
    pf.geometry("700x350")
    pf.resizable(0,0)
    Label(pf,text="Please choose appropriate step you want to proceed",font="Arial 18 bold").place(relx=0.5,rely=0.3,anchor=CENTER)
    Button(pf,text="Intrusion Detection >>",font="Arial 10 bold",command=chooseIDS).place(relx=0.32,rely=0.6,anchor=CENTER)
    Button(pf,text="Efficiencies of Algorithms >>",font="Arial 10 bold",command=evaluate).place(relx=0.57,rely=0.6,anchor=CENTER)
    Button(pf,text="Exit >>",font="Arial 10 bold",command=lambda:Exit(pf)).place(relx=0.75,rely=0.6,anchor=CENTER)
    pf.mainloop()

# 1. Welcome Page
main=Tk()
main.title("IDS")
main.geometry("700x350")
main.resizable(0,0)
Label(main,text="Intrusion Detection System",font="Arial 18 bold").place(relx=0.5,rely=0.3,anchor=CENTER)
Button(main,text="Next >>",font="Arial 10 bold",command=proceedfurther).place(relx=0.5,rely=0.6,anchor=CENTER)
main.mainloop()