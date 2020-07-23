import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import internal_functions


#Stages of the CBR Cycle:
def init_cbr(data):
    "Generates the Problem section of a Case with the data provided by the user"
    ##Create a problem from input data
    new_case= internal_functions.Case()
    new_case.set_problem(new_report=data['report'],new_image=data['images'],new_terms=data['ne_terms'],new_abbrs=data['abbrvs'])
    return new_case

def cbr_retrieval(st,new_case,query):
    "Retrieval Stage: Returns the most similar cases to the current problem"
    return st.find_top_cases(new_case,query)

def cbr_reuse(st,new_case,related_cases):
    "Reuse Stage: Generates a solution for the current problem."
    disambiguations =internal_functions.disambiguate_abbreviation(new_case.get_problem().get_report())
    section_test = internal_functions.section_text(new_case.get_problem().get_report())
    related_cases=[c['Case_ID'] for c in related_cases]
    existing_ners=internal_functions.get_case_related_entities(st,related_cases)
    entities=internal_functions.return_related_entities(existing_ners)
    score=internal_functions.score_case(section_test)
    new_case.get_solution().set_data(new_sections=section_test,new_ner=entities,new_abbvs=disambiguations,new_cases=related_cases)
    return new_case,score

def cbr_revise(st,case_ID,new_report, validation_value):
    st.change_case_solution(case_ID, new_report, change_validation=validation_value)

