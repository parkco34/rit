from example2 import *

class Grades (object):
    def __init__(self):
        """Create empty grade book"""
        self._students = []
        self._grades = {}
        self._is_sorted = True
    
    # Mutator methods
    def add_student(self, student):
        """Assumes: student is of type Student 
        Add student to the grade book"""
        if student in self._students:
            raise ValueError('Duplicate student')
        
        self._students.append(student)
        self._grades[student.get_id_num()] = []
        self._is_sorted = False

    def add_grade(self, student, grade):
        """Assumes: grade is a float
        Add grade to the list of grades for student"""
        try:
            self._grades[student.get_id_num()].append(grade)
            
        except:
            raise ValueError('Student not in mapping')

    def get_grades(self, student):
        """Return a list of grades for student"""
        try:
            return self._grades[student.get_id_num()][:]
        
        except:
            raise ValueError('Student not in mapping')

    # Accessor method
    def get_students(self):
        """Return a sorted list of the students in the grade book"""
        if not self._is_sorted:
            self._students.sort()
            self._is_sorted = True
            
        return self._students[:]


def main():
    pass


if __name__ == "__main__":
    main()