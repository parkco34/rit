#!/usr/bin/env python
from graphviz import Digraph

def create_hta_diagram():
    dot = Digraph(comment='Applying for Financial Aid', format='png')
    
    # Diagram attributes for layout and style
    dot.attr(rankdir='LR', pad='0.5', nodesep='0.16', ranksep='0.2')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightyellow', color='lightgrey', fontname='Helvetica', fontsize='10')
    dot.attr('edge', color='lightgrey', fontname='Helvetica', fontsize='8')

    # Main objective
    dot.node('A', 'Apply for Financial Aid')

    # Main tasks
    dot.node('B', 'Gather Necessary Documents and Information')
    dot.node('C', 'Complete the FAFSA')
    dot.node('D', 'Apply for State and Institutional Aid')
    dot.node('E', 'Seek Out and Apply for Scholarships')
    dot.node('F', 'Review and Accept Financial Aid Offers')
    dot.node('G', 'Complete Loan Counseling and Sign MPN')

    # Subtasks and actions, with at least five levels deep for one branch
    dot.node('B1', 'Identify required documentation')
    dot.node('B1a', 'Check financial aid application requirements on school\'s website')
    dot.node('B1a1', 'Login to school portal')
    dot.node('B1a2', 'Navigate to financial aid section')
    dot.node('B1a3', 'Download and review documentation checklist')

    # Adding edges to create the hierarchy
    dot.edges([('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'F'), ('A', 'G')])
    dot.edges([('B', 'B1')])
    dot.edges([('B1', 'B1a')])
    dot.edges([('B1a', 'B1a1'), ('B1a', 'B1a2'), ('B1a', 'B1a3')])

    # Return the dot object for further customizations or rendering
    return dot

# Create the HTA diagram
hta_dot = create_hta_diagram()

# Render the HTA diagram to a PNG file
output_path = '/Users/whitney/Desktop/financial_aid_hta_diagram'
hta_dot.render(output_path, view=False)

# Return the path to the PNG file
output_path + '.png'
