#!/usr/bin/env python
import time
import random
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_response(statement):
    valid_responses = ["good", "bad", "neutral"]
    while True:
        print(f"Statement: {statement}")
        speak(statement)
        start_time = time.time()
        response = input("Enter your response (good/bad/neutral): ").lower()
        if response in valid_responses:
            end_time = time.time()
            response_time = end_time - start_time
            return response, response_time
        else:
            print("Invalid response. Please enter 'good', 'bad', or 'neutral'.")

def calculate_hs_values(response_times):
    hs_values = [1 / rt for rt in response_times]
    return hs_values

def plot_and_analyze(hs_values, response_times):
    plt.figure(figsize=(8, 6))
    plt.scatter(hs_values, response_times)
    plt.xlabel("Hs")
    plt.ylabel("Response Time (seconds)")
    plt.title("Hick-Hyman Law Analysis - All Participants")
    slope, intercept, r_value, p_value, std_err = linregress(hs_values, response_times)
    x = [min(hs_values), max(hs_values)]
    y = [slope * xi + intercept for xi in x]
    plt.plot(x, y, color='red', label=f"Regression Line: y = {slope:.2f}x + {intercept:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hick_hyman_analysis_all_participants.png")
    plt.close()
    print(f"Regression Equation: Response Time = {slope:.2f} * Hs + {intercept:.2f}")
    print(f"R-squared: {r_value**2:.2f}")

def get_participant_info():
    participant_id = input("Enter participant ID: ")
    age = input("Enter age: ")
    gender = input("Enter gender: ")
    education = input("Enter education level: ")
    return participant_id, age, gender, education

def main():
    statements = [
        "The sky is blue.",
        "Humans need oxygen to survive.",
        "Cats are better pets than dogs.",
        "Pineapple belongs on pizza.",
        "Climate change is not real.",
    ]
    random.shuffle(statements)
    num_statements = len(statements)
    all_hs_values = []
    all_response_times = []

    with open("output.csv", "a", newline="") as file:
        fieldnames = ["Participant ID", "Age", "Gender", "Education", "Statement", "Response", "Response Time"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()

        while True:
            participant_id, age, gender, education = get_participant_info()
            response_times = []
            for i, statement in enumerate(statements, 1):
                print(f"Statement {i}/{num_statements}")
                while True:
                    response, response_time = get_response(statement)
                    if response:
                        writer.writerow({"Participant ID": participant_id, "Age": age, "Gender": gender, "Education": education,
                                         "Statement": statement, "Response": response, "Response Time": response_time})
                        response_times.append(response_time)
                        break
                    else:
                        print("Invalid response. Please try again.")

            hs_values = calculate_hs_values(response_times)
            all_hs_values.extend(hs_values)
            all_response_times.extend(response_times)

            choice = input("Do you want to add another participant? (yes/no): ")
            if choice.lower() != 'yes':
                break

    plot_and_analyze(all_hs_values, all_response_times)

if __name__ == "__main__":
    main()
