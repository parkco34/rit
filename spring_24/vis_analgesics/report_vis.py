#!/usr/bin/env python
import matplotlib.pyplot as plt

# Data
risks = ['Theft', 'Geopolitical conflicts', 'Civil unrest', 'Gun violence']
risk_percentages = [72, 62, 54, 49]
precautions = ['Wear face covering', 'Avoid large events', 'Avoid close contact']
precaution_percentages = [30, 26, 21]
safety_measures = ['Share itinerary', 'Regular texts/calls', 'Research safe areas', 'Limit evening activities', 'Avoid disclosing solo status', 'Share live location', 'Make emergency plan']
safety_measure_percentages = [83, 63, 52, 51, 41, 40, 25]
tools = ['Language translation', 'Itinerary maker', 'Smart assistant', 'Pricing deals predictor']
tool_percentages = [74, 52, 51, 39]

# Creating multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plotting top travel-related risks
axs[0, 0].bar(risks, risk_percentages, color=['skyblue', 'salmon', 'lightgreen', 'gold'])
axs[0, 0].set_title('Top Travel-Related Risks', fontsize=16)
axs[0, 0].set_ylabel('Percentage (%)', fontsize=14)
axs[0, 0].grid(True, linestyle='--', alpha=0.5)

# Plotting safety precautions
axs[0, 1].bar(precautions, precaution_percentages, color=['purple', 'teal', 'orange'])
axs[0, 1].set_title('Safety Precautions Taken', fontsize=16)
axs[0, 1].set_ylabel('Percentage (%)', fontsize=14)
axs[0, 1].grid(True, linestyle='--', alpha=0.5)

# Plotting how solo travelers stay safe
axs[1, 0].bar(safety_measures, safety_measure_percentages, color='lightblue')
axs[1, 0].set_title('How Solo Travelers Stay Safe', fontsize=16)
axs[1, 0].set_ylabel('Percentage (%)', fontsize=14)
axs[1, 0].set_xticklabels(safety_measures, rotation=45, ha='right')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)

# Plotting appealing AI tools for travelers
axs[1, 1].bar(tools, tool_percentages, color=['brown', 'gray', 'cyan', 'magenta'])
axs[1, 1].set_title('AI Tools That Appeal to Travelers', fontsize=16)
axs[1, 1].set_ylabel('Percentage (%)', fontsize=14)
axs[1, 1].grid(True, linestyle='--', alpha=0.5)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#import matplotlib.pyplot as plt
#
## Data setup
#risks = ['Theft', 'Geopolitical conflicts', 'Civil unrest', 'Gun violence']
#percentages = [72, 62, 54, 49]
#colors = ['skyblue', 'salmon', 'lightgreen', 'gold']  # Distinct colors for each risk
#
#plt.figure(figsize=(12, 6))
#bars = plt.bar(risks, percentages, color=colors)
#
## Adding grid, enhancing fonts, and better title
#plt.title("Top Travel-Related Risks Concerning Solo Travelers", fontsize=18, fontweight='bold')
#plt.xlabel("Risk Categories", fontsize=16, labelpad=10)
#plt.ylabel("Percentage (%) of Concern", fontsize=16, labelpad=10)
#plt.xticks(fontsize=14, fontweight='bold')
#plt.yticks(fontsize=14)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#
## Annotating each bar with the respective percentage
#for bar in bars:
#    yval = bar.get_height()
#    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval}%', ha='center', fontsize=14, fontweight='bold')
#
#plt.tight_layout()
#plt.show()

#import matplotlib.pyplot as plt
#
#risks = ['Theft', 'Geopolitical conflicts', 'Civil unrest', 'Gun violence']
#percentages = [72, 62, 54, 49]
#
#plt.figure(figsize=(10,5))
#plt.bar(risks, percentages, color='cornflowerblue')
#plt.title("Top Travel-Related Risks Concerning Solo Travelers", fontsize=16)
#plt.xlabel("Risk", fontsize=14)
#plt.ylabel("Percentage Concerned", fontsize=14)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#
#for i, v in enumerate(percentages):
#    plt.text(i, v+1.5, str(v)+'%', ha='center', fontsize=12)
#
#plt.tight_layout()
#plt.show()
#
