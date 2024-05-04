#!/usr/bin/env python

def patient_inprocessing():
    """
    Function: Patient In-processing
    Output: Patient information
    Connection: Outputs "Patient information", which is used as input in setup_therac25()
    """
    patient_info = {}
    return patient_info

def plug_in_monitor():
    """
    Function: Plug in Monitor
    Output: Monitor plugged in
    Connection: Outputs "Monitor plugged in", which is used as input in check_equipment()
    """
    monitor_plugged_in = True
    return monitor_plugged_in

def fix_intercom():
    """
    Function: Fix Intercom
    Output: Intercom operational
    Connection: Outputs "Intercom operational", which is used as input in check_equipment()
    """
    intercom_operational = True
    return intercom_operational

def check_equipment(monitor_plugged_in, intercom_operational):
    """
    Function: Check Equipment
    Input: Monitor plugged in, Intercom operational
    Output: Equipment status
    Connection: Inputs "Monitor plugged in" from plug_in_monitor() and "Intercom operational" from fix_intercom()
                Outputs "Equipment status", which is used as input in setup_therac25()
    """
    equipment_status = monitor_plugged_in and intercom_operational
    return equipment_status

def setup_therac25(patient_info, equipment_status):
    """
    Function: Setup Therac-25
    Input: Patient information, Equipment status
    Output: Machine settings
    Control: Oncologist prescription
    Connection: Inputs "Patient information" from patient_inprocessing() and "Equipment status" from check_equipment()
                Outputs "Machine settings", which are used as input in perform_treatment() and radiation_dosage_validation()
    """
    machine_settings = {}
    return machine_settings

def radiation_dosage_validation(machine_settings):
    """
    Function: Radiation Dosage Validation
    Input: Machine settings
    Output: Good-to-go
    Connection: Inputs "Machine settings" from setup_therac25()
                Outputs "Good-to-go", which is used as input in perform_treatment()
    """
    good_to_go = True
    return good_to_go

def perform_treatment(machine_settings, good_to_go):
    """
    Function: Perform Treatment
    Input: Machine settings, Good-to-go
    Output: Treatment status
    Connection: Inputs "Machine settings" from setup_therac25() and "Good-to-go" from radiation_dosage_validation()
                Outputs "Treatment status", which is used as input in monitor_patient()
    """
    treatment_status = "Treatment completed"
    return treatment_status

def monitor_patient(treatment_status):
    """
    Function: Monitor Patient
    Input: Treatment status
    Output: Patient status
    Connection: Inputs "Treatment status" from perform_treatment()
    """
    patient_status = "Stable"
    return patient_status

def equipment_fix():
    """
    Function: Equipment Fix
    Output: Equipment fixed
    Connection: Outputs "Equipment fixed", which is used as input in setup_therac25()
    """
    equipment_fixed = True
    return equipment_fixed

# Main program flow
patient_info = patient_inprocessing()
monitor_plugged_in = plug_in_monitor()
intercom_operational = fix_intercom()
equipment_status = check_equipment(monitor_plugged_in, intercom_operational)

machine_settings = setup_therac25(patient_info, equipment_status)
good_to_go = radiation_dosage_validation(machine_settings)
treatment_status = perform_treatment(machine_settings, good_to_go)
patient_status = monitor_patient(treatment_status)

if not equipment_status:
    equipment_fixed = equipment_fix()
    machine_settings = setup_therac25(patient_info, equipment_fixed)

