#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_descriptive_stats(data):
    # Calculate descriptive statistics
    hits = data[(data['signal_present'] == True) & (data['response'] == 'y')].shape[0]
    misses = data[(data['signal_present'] == True) & (data['response'] == 'n')].shape[0]
    false_alarms = data[(data['signal_present'] == False) & (data['response'] == 'y')].shape[0]
    correct_rejections = data[(data['signal_present'] == False) & (data['response'] == 'n')].shape[0]

    # Calculate hit rate and false alarm rate
    hit_rate = hits / (hits + misses) if hits + misses > 0 else 0
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if false_alarms + correct_rejections > 0 else 0

    return hits, misses, false_alarms, correct_rejections, hit_rate, false_alarm_rate

def plot_roc_curve(hit_rates, false_alarm_rates):
    plt.figure(figsize=(6, 6))
    plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()

def calculate_d_prime_and_beta(hit_rate, false_alarm_rate):
    if hit_rate == 1:
        hit_rate = 1 - 1e-7  # Adjust hit rate to avoid infinite z-score
    if false_alarm_rate == 0:
        false_alarm_rate = 1e-7  # Adjust false alarm rate to avoid infinite z-score

    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(false_alarm_rate)
    d_prime = z_hit - z_fa
    beta = np.exp((z_fa**2 - z_hit**2) / 2)
    return d_prime, beta

# Read data from summary.txt
with open('summary.txt', 'r') as file:
    summary_lines = file.readlines()

# Read data from resultz.txt
with open('resultz.txt', 'r') as file:
    resultz_lines = file.readlines()

# Extract data for each condition from summary.txt
summary_data = []
current_condition = None

for line in summary_lines:
    if line.startswith('Condition:'):
        current_condition = line.strip().split(': ')[1]
    elif line.strip():
        hits, false_alarms, misses, correct_rejections = map(int, [x.split(':')[1].strip() for x in line.strip().split(',')])
        summary_data.append({
            'Condition': current_condition,
            'Hits': hits,
            'False Alarms': false_alarms,
            'Misses': misses,
            'Correct Rejections': correct_rejections
        })

# Extract data for each condition from resultz.txt
resultz_data = []
current_condition = None
condition_data = []

for line in resultz_lines:
    if line.startswith('Condition:'):
        if current_condition is not None:
            resultz_data.append((current_condition, condition_data))
            condition_data = []
        current_condition = line.strip().split(': ')[1]
    else:
        condition_data.append(line.strip())

if current_condition is not None:
    resultz_data.append((current_condition, condition_data))

# Combine summary and resultz data
results = []

for summary, resultz in zip(summary_data, resultz_data):
    condition = summary['Condition']
    hits = summary['Hits']
    misses = summary['Misses']
    false_alarms = summary['False Alarms']
    correct_rejections = summary['Correct Rejections']

    data_df = pd.DataFrame([row.split(',') for row in resultz[1]], columns=['trial', 'signal_present', 'response', 'reaction_time'])
    data_df['signal_present'] = data_df['signal_present'].map({'True': True, 'False': False})
    hit_rate = hits / (hits + misses) if hits + misses > 0 else 0
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if false_alarms + correct_rejections > 0 else 0
    d_prime, beta = calculate_d_prime_and_beta(hit_rate, false_alarm_rate)

    results.append({
        'Condition': condition,
        'Hits': hits,
        'Misses': misses,
        'False Alarms': false_alarms,
        'Correct Rejections': correct_rejections,
        'Hit Rate': hit_rate,
        'False Alarm Rate': false_alarm_rate,
        'd-prime': d_prime,
        'beta': beta
    })

    # Append resultz data to the results
    for trial_data in data_df.itertuples(index=False):
        results.append({
            'Condition': condition,
            'Trial': trial_data.trial,
            'Signal Present': trial_data.signal_present,
            'Response': trial_data.response,
            'Reaction Time': trial_data.reaction_time
        })

# Save the results as a text file
with open('results.txt', 'w') as file:
    for result in results:
        if 'Trial' in result:
            file.write(f"Condition: {result['Condition']}\n")
            file.write(f"Trial: {result['Trial']}, Signal Present: {result['Signal Present']}, Response: {result['Response']}, Reaction Time: {result['Reaction Time']}\n")
        else:
            file.write(f"Condition: {result['Condition']}\n")
            file.write(f"Hits: {result['Hits']}, Misses: {result['Misses']}, False Alarms: {result['False Alarms']}, Correct Rejections: {result['Correct Rejections']}\n")
            file.write(f"Hit Rate: {result['Hit Rate']:.2f}, False Alarm Rate: {result['False Alarm Rate']:.2f}, d-prime: {result['d-prime']:.2f}, beta: {result['beta']:.2f}\n\n")

# Plot the ROC curve
hit_rates = [result['Hit Rate'] for result in results if 'Trial' not in result]
false_alarm_rates = [result['False Alarm Rate'] for result in results if 'Trial' not in result]
plot_roc_curve(hit_rates, false_alarm_rates)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_descriptive_stats(data):
    # Calculate descriptive statistics
    hits = data[(data['signal_present'] == True) & (data['response'] == 'y')].shape[0]
    misses = data[(data['signal_present'] == True) & (data['response'] == 'n')].shape[0]
    false_alarms = data[(data['signal_present'] == False) & (data['response'] == 'y')].shape[0]
    correct_rejections = data[(data['signal_present'] == False) & (data['response'] == 'n')].shape[0]
    
    # Calculate hit rate and false alarm rate
    hit_rate = hits / (hits + misses) if hits + misses > 0 else 0
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if false_alarms + correct_rejections > 0 else 0
    
    return hits, misses, false_alarms, correct_rejections, hit_rate, false_alarm_rate

def plot_roc_curve(hit_rates, false_alarm_rates):
    plt.figure(figsize=(6, 6))
    plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()

def calculate_d_prime_and_beta(hit_rate, false_alarm_rate):
    if hit_rate == 1:
        hit_rate = 1 - 1e-7  # Adjust hit rate to avoid infinite z-score
    if false_alarm_rate == 0:
        false_alarm_rate = 1e-7  # Adjust false alarm rate to avoid infinite z-score
    
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(false_alarm_rate)
    d_prime = z_hit - z_fa
    beta = np.exp((z_fa**2 - z_hit**2) / 2)
    return d_prime, beta

# Read data from summary.txt
with open('summary.txt', 'r') as file:
    summary_lines = file.readlines()

# Read data from resultz.txt
with open('resultz.txt', 'r') as file:
    resultz_lines = file.readlines()

# Extract data for each condition from summary.txt
summary_data = []
current_condition = None

for line in summary_lines:
    if line.startswith('Condition:'):
        current_condition = line.strip().split(': ')[1]
    elif line.strip():
        hits, false_alarms, misses, correct_rejections = map(int, [x.split(':')[1].strip() for x in line.strip().split(',')])
        summary_data.append({
            'Condition': current_condition,
            'Hits': hits,
            'False Alarms': false_alarms,
            'Misses': misses,
            'Correct Rejections': correct_rejections
        })

# Extract data for each condition from resultz.txt
resultz_data = []
current_condition = None
condition_data = []

for line in resultz_lines:
    if line.startswith('Condition:'):
        if current_condition is not None:
            resultz_data.append((current_condition, condition_data))
            condition_data = []
        current_condition = line.strip().split(': ')[1]
    else:
        condition_data.append(line.strip())

if current_condition is not None:
    resultz_data.append((current_condition, condition_data))

# Combine summary and resultz data
results = []

for summary, resultz in zip(summary_data, resultz_data):
    condition = summary['Condition']
    hits = summary['Hits']
    misses = summary['Misses']
    false_alarms = summary['False Alarms']
    correct_rejections = summary['Correct Rejections']
    
    data_df = pd.DataFrame([row.split(',') for row in resultz[1]], columns=['trial', 'signal_present', 'response', 'reaction_time'])
    data_df['signal_present'] = data_df['signal_present'].map({'True': True, 'False': False})
    hit_rate = hits / (hits + misses) if hits + misses > 0 else 0
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if false_alarms + correct_rejections > 0 else 0
    d_prime, beta = calculate_d_prime_and_beta(hit_rate, false_alarm_rate)
    
    results.append({
        'Condition': condition,
        'Hits': hits,
        'Misses': misses,
        'False Alarms': false_alarms,
        'Correct Rejections': correct_rejections,
        'Hit Rate': hit_rate,
        'False Alarm Rate': false_alarm_rate,
        'd-prime': d_prime,
        'beta': beta
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results as a CSV file
results_df.to_csv('results.csv', index=False)

# Plot the ROC curve
hit_rates = results_df['Hit Rate'].tolist()
false_alarm_rates = results_df['False Alarm Rate'].tolist()
plot_roc_curve(hit_rates, false_alarm_rates)
