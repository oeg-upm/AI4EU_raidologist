import sys
import os
import googletrans

from googletrans import Translator
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import internal_functions
translator=Translator()

#Stages of the CBR Cycle:
def init_cbr(data):
    "Generates the Problem section of a Case with the data provided by the user"
    ##Create a problem from input data
    new_case = internal_functions.Case()
    lang_data=translator.detect(data['report'])
    lang=lang_data.lang
    if lang != 'en':
        trans_report=translator.translate(data['report'],dest='en',src=lang).text
        terms=[]
        for term in data['ne_terms']:
            if term:
                terms.append(translator.translate(term,dest='en',src=lang).text)
        new_case.set_problem(new_report=trans_report, new_image=data['images'], new_terms=terms,
                             new_abbrs=data['abbrvs'])
        new_case.get_problem().set_lang_data(new_lang=lang,new_original_report=data['report'])
    else:
        new_case.set_problem(new_report=data['report'],new_image=data['images'],new_terms=data['ne_terms']
                         ,new_abbrs=data['abbrvs'],new_lang=lang)
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
    if new_case.get_problem().get_src_lang()!='en':
        lang=new_case.get_problem().get_src_lang()
        trans_report=translator.translate(section_test,dest=lang).text
        new_case.get_solution().set_data(new_sections=section_test, new_ner=entities,
                                         new_abbvs=disambiguations, new_cases=related_cases)
        new_case.get_solution().set_original_section(trans_report)
    else:
        new_case.get_solution().set_data(new_sections=section_test,new_ner=entities,
                                         new_abbvs=disambiguations,new_cases=related_cases)
    return new_case,score

def cbr_revise(st,case_ID,new_report, validation_value):
    st.change_case_solution(case_ID, new_report, change_validation=validation_value)

