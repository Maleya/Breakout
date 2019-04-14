# if you want to look at the data
import matplotlib

matplotlib.use("TkAgg")  # for mac users
import numpy as np
import csv
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 15})
# ------------- load the data -------------
mean_score_list = []
learning_count_list = []
var_score_list = []
max_score_list = []
q_val_list = []
###########
#OBS TVÅ FILER PGA KÖRNING SOM KRASHADE OCH BÖRJADE SPARA NYA RESULTAT
#I NY FIL EFTER 1,000,000 LEARNING ITERATIONS
###########
with open("./data/plot_data_final_2.csv", "r") as csvFile:
    next(csvFile)
    reader = csv.reader(csvFile)
    for row in reader:
        learning_count, mean_score, var_score, max_score, q_val = row
        mean_score_list.append(float(mean_score))
        learning_count_list.append(float(learning_count))
        var_score_list.append(float(var_score))
        max_score_list.append(float(max_score))
        q_val_list.append(float(q_val))


with open("./data/plot_data_final_2_run2.csv", "r") as csvFile:
    next(csvFile)
    reader = csv.reader(csvFile)
    for row in reader:
        learning_count, mean_score, var_score, max_score, q_val = row
        mean_score_list.append(float(mean_score))
        learning_count_list.append(float(learning_count))
        var_score_list.append(float(var_score))
        max_score_list.append(float(max_score))
        q_val_list.append(float(q_val))

# ------------- plots -------------
print(len(mean_score_list))
np.linspace(0,len(mean_score_list),len(mean_score_list))
plt.figure('Mean Score') 
plt.plot(np.linspace(0,len(mean_score_list),len(mean_score_list)), mean_score_list, ".-", label="Mean score")
plt.axhline(y=1.3217, alpha=0.3, label="mean random agent score (10000 episodes)")
plt.xlabel("Training Epochs")
plt.ylabel("Average Reward per Episode")
plt.title("Average Reward on Breakout")
#plt.legend()
plt.tight_layout()
plt.show()

plt.figure('Max Score') 
plt.plot(np.linspace(0,len(mean_score_list),len(mean_score_list)), max_score_list, ".-", label="Max score")
plt.xlabel("Training Epochs")
plt.ylabel("Max Score")
plt.title("Max Score")
#plt.legend()
plt.tight_layout()
plt.show()

plt.figure('Mean Q') 
plt.plot(np.linspace(0,len(mean_score_list),len(mean_score_list)), q_val_list, ".-", label="Max q prediction")
plt.xlabel("Training Epochs")
plt.ylabel("Average Action Value (Q)")
plt.title("Average Q on Breakout")
#plt.legend()
plt.tight_layout()
plt.show()