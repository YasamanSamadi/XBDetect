import matplotlib.pyplot as plt
import numpy as np

# Sample AUC results (replace with your actual model results)
datasets = ['RoninBridge', 'Wormhole', 'Qubit Finance']
unweighted_bce = [86.2, 83.8, 81.5]
weighted_bce = [89.3, 86.3, 84.5]
focal_loss = [90.1, 87.0, 85.2]

# Number of groups
x = np.arange(len(datasets))  # label locations
width = 0.25  # width of bars

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
rects1 = ax.bar(x - width, unweighted_bce, width, label='Unweighted BCE', color='#9ecae1')
rects2 = ax.bar(x, weighted_bce, width, label='Weighted BCE', color='#6baed6')
rects3 = ax.bar(x + width, focal_loss, width, label='Focal Loss', color='#2171b5')

# Add labels, title, and custom ticks
ax.set_ylabel('AUC (%)', fontsize=12)
ax.set_title('Impact of Loss Functions on Anomaly Detection', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylim(75, 95)
ax.legend(fontsize=10)

# Add value labels on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.tight_layout()
plt.show()
