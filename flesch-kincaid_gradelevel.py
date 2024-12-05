import textstat

# calculate the Flesch-Kincaid Grade Level of a text
text = """
The Flesch-Kincaid Grade Level is a readability test designed to indicate how difficult a passage in English is to understand. 
It uses the total number of words, sentences, and syllables in the text to calculate a grade level.
"""
grade_level = textstat.flesch_kincaid_grade(text)
print("Level:", grade_level)