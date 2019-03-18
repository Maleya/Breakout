# if you want to look at the data
import matplotlib
matplotlib.use("TkAgg")  # for mac users
import numpy as np
import csv
from matplotlib import pyplot as plt
#matplotlib.use("TkAgg")  # for mac users

# ------------- Helper functions -------------
def mean_batches(in_list, batch_size):
    old_i = 0
    mean_list = []
    for i, score in enumerate(in_list):
        if i % batch_size == 0 and i != 0:
            mean_list.append(np.mean(in_list[old_i:i]))
            old_i = i
    return mean_list


# ------------- load the data -------------
score_list = []
run_list = []
with open('./data/plot_data.csv', 'r') as csvFile:
    next(csvFile)
    reader = csv.reader(csvFile)
    for row in reader:
        run, score = row
        score_list.append(float(score))
        run_list.append(float(run))


# ------------- plots -------------
mean_data = mean_batches(score_list, 125)
x_axis = [i for i in range(len(mean_data))]

# linear regression:
fit = np.polyfit(x_axis, mean_data, 1)
fit_fn = np.poly1d(fit)  # takes x returns estimate for y
print(f'line of best fit: y={fit_fn}')
print(f"highest score: {max(score_list)}(index:{np.argmax(score_list)}/{len(score_list)}), variance:{np.var(score_list)}")
plt.plot(mean_data, '.', label='mean of 125 episodes')
plt.plot(x_axis, fit_fn(x_axis), label=f'best fit 1d: {fit_fn}')
plt.axhline(y=1.3217, alpha=0.3, label="mean random agent score (10000 episodes)")
plt.xlabel('Number of played epochs [1 epoch aprox = 125 episodes]')
plt.ylabel('Mean episode score')
plt.title('Mean game score as function of episodes played')
plt.legend()
plt.tight_layout()
plt.show()
