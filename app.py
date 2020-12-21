import spacy
import streamlit as st
import pandas as pd
import json
import requests
import umls
from spacy.tokens import Span
from spacy import displacy
import urllib
from utils import get_html
from negspacy.negation import Negex


try:
    with open('umls_api.txt', 'r') as file:
        umls_apikey = file.read().replace('\n', '')
except:
    url = "https://www.dropbox.com/s/m10v41n5to4jfo8/umls_api.txt?dl=1"
    file = urllib.request.urlopen(url)

    for line in file:
        decoded_line = line.decode("utf-8")
    umls_apikey = decoded_line


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(name):
    return spacy.load(name)


spacy_model = 'en_core_sci_sm'
nlp = load_model(spacy_model)

try:
    negex = Negex(nlp, language="en_clinical",
                  chunk_prefix=["no"])
    nlp.add_pipe(negex)
except:
    st.write('')


def add_umls_entities(doc):
    new_ents = []
    for ent in doc.ents:
        if ent._.negex:
            new_ent = Span(doc, ent.start, ent.end, label='Negation')
            new_ents.append(new_ent)
        else:
            try:
                tgt = umls.get_tgt(umls_apikey)
                cui = umls.search_by_atom(ent.text, tgt).loc[0].ui
                new_label = umls.search_by_cui(cui, tgt)[
                    'semanticTypes'][0]['name']
                new_ent = Span(doc, ent.start, ent.end, label=new_label)
                new_ents.append(new_ent)
            except:
                new_label = 'other'
                new_ent = Span(doc, ent.start, ent.end, label=new_label)
                new_ents.append(new_ent)
    doc.ents = new_ents
    return doc


try:
    nlp.add_pipe(add_umls_entities
                 # , after='ner'
                 )
except:
    st.write('')


# text = st.text_area(
#     'Text', """
#     Past medical history includes hypertension, dyslipidemia and diabetes, and a family history of coronary artery disease.
#     He started having chest pain 4 hours ago, associated with dyspnea, nausea, and diaphoresis.
#     """)

text = st.text_area(
    'Text', """He started having chest pain 4 hours ago, associated with dyspnea, nausea, and diaphoresis.""")

doc = nlp(text)

target_labels = ['Finding', 'Disease or Syndrome',
                 'Sign or Symptom', 'Pathologic Function', 'Neoplastic Process', 'Other']


html = displacy.render(
    doc, style="ent",
    options={
        "ents": ['FINDING', 'DISEASE OR SYNDROME',
                 'SIGN OR SYMPTOM', 'PATHOLOGIC FUNCTION', 'NEOPLASTIC PROCESS', 'OTHER', 'NEGATION'],
        "colors": {'FINDING': '#D0ECE7', 'DISEASE OR SYNDROME': '#D6EAF8',
                   'SIGN OR SYMPTOM': '#E8DAEF', 'PATHOLOGIC FUNCTION': '#F7D8A6', 'NEOPLASTIC PROCESS': '#DAF7A6',
                   'NEGATION': '#FADBD8'}
    }
)
style = "<style>mark.entity { display: inline-block }</style>"
st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)

data = [
    [str(getattr(ent, attr)) for attr in ["text", "label_", "start", "end", "start_char", "end_char"]
     ]
    for ent in doc.ents
    # if ent.label_ in target_labels
]
df = pd.DataFrame(data, columns=["text", "label_", "start", "end", "start_char", "end_char"]
                  )
st.dataframe(df)


# st.write(nlp.pipeline)
