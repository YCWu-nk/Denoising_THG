import re
import matplotlib.pyplot as plt

with open('train_gauss25-2023-8-10-96-1_230813-200545.log', 'r') as file:
    lines = file.readlines()

loss_all_values = []

for line in lines:
    match = re.search(r'Loss_All=([\d.]+)', line)
    if match:
        loss_all_values.append(float(match.group(1)))

x_values = range(0, 100)
y_values = loss_all_values[:100]  
plt.plot(x_values, y_values)
plt.xlabel('epoch')
plt.ylabel('Loss_All')
plt.title('Loss')
plt.show()