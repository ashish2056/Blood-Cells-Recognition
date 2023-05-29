import matplotlib.pyplot as plt

learning_rate = 0.001
max_batches = 40100
steps = [-1, 100, 20000, 30000]
scales = [0.1, 10, 0.1, 0.1]

batch_nums = range(max_batches)
lr_values = []

for batch in batch_nums:
    current_lr = learning_rate
    for i, step in enumerate(steps):
        if batch > step:
            current_lr *= scales[i]
    lr_values.append(current_lr)

plt.plot(batch_nums, lr_values)
plt.xlabel('Batches')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)
plt.show()
