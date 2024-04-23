#!/usr/bin/env python
import matplotlib.pyplot as plt

risks = ['Theft', 'Geopolitical conflicts', 'Civil unrest', 'Gun violence']
percentages = [72, 62, 54, 49]

plt.figure(figsize=(10,5))
plt.bar(risks, percentages, color='cornflowerblue')
plt.title("Top Travel-Related Risks Concerning Solo Travelers", fontsize=16)
plt.xlabel("Risk", fontsize=14)
plt.ylabel("Percentage Concerned", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for i, v in enumerate(percentages):
    plt.text(i, v+1.5, str(v)+'%', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

