import spacy
import random
import time
from spacy.util import minibatch, compounding
from spacy.training.example import Example

TRAIN_DATA = [('Exxon Mobil', {'entities': [(11, 0, 'ORG')]}), ('Wal-Mart Stores', {'entities': [(15, 0, 'ORG')]}), ('Chevron', {'entities': [(7, 0, 'ORG')]}), ('ConocoPhillips', {'entities': [(14, 0, 'ORG')]}), ('General Motors', {'entities': [(14, 0, 'ORG')]}), ('Ford Motor', {'entities': [(10, 0, 'ORG')]}), ('Hewlett-Packard', {'entities': [(15, 0, 'ORG')]}), ('AT&T', {'entities': [(4, 0, 'ORG')]}), ('Valero Energy', {'entities': [(13, 0, 'ORG')]}), ('J.P. Morgan Chase & Co.', {'entities': [(22, 0, 'ORG')]}), ('Apple', {'entities': [(5, 0, 'ORG')]}), ('CVS Caremark', {'entities': [(12, 0, 'ORG')]}), ('Kroger', {'entities': [(6, 0, 'ORG')]}), ('American International Group', {'entities': [(28, 0, 'ORG')]}), ('MetLife', {'entities': [(7, 0, 'ORG')]}), ('Microsoft', {'entities': [(9, 0, 'ORG')]}), ('Target', {'entities': [(6, 0, 'ORG')]}), ('Pfizer', {'entities': [(6, 0, 'ORG')]}), ('Johnson & Johnson', {'entities': [(17, 0, 'ORG')]}), ('Dell', {'entities': [(4, 0, 'ORG')]}), ('Intel', {'entities': [(5, 0, 'ORG')]}), ('Amazon.com', {'entities': [(6, 0, 'ORG')]}), ('Lockheed Martin', {'entities': [(15, 0, 'ORG')]}), ('Cisco Systems', {'entities': [(13, 0, 'ORG')]}), ('Morgan Stanley', {'entities': [(14, 0, 'ORG')]}), ('Abbott Laboratories', {'entities': [(19, 0, 'ORG')]}), ('Google', {'entities': [(6, 0, 'ORG')]}), ('Goldman Sachs Group', {'entities': [(19, 0, 'ORG')]}), ('Oracle', {'entities': [(6, 0, 'ORG')]}), ('American Express', {'entities': [(16, 0, 'ORG')]}), ('Intel Corporation', {'entities': [(17, 0, 'ORG')]}), ('Sun Microsystems', {'entities': [(16, 0, 'ORG')]}), ('The Apache Software Foundation', {'entities': [(30, 0, 'ORG')]}), ('Oracle America', {'entities': [(14, 0, 'ORG')]}), ('Xerox Corporation', {'entities': [(17, 0, 'ORG')]}), ('International Business Machines', {'entities': [(31, 0, 'ORG')]}), ('Insight Software Consortium', {'entities': [(27, 0, 'ORG')]}), ('Internet Systems Consortium', {'entities': [(27, 0, 'ORG')]}), ('The NetBSD Foundation', {'entities': [(21, 0, 'ORG')]}), ('Japan Network Information Center', {'entities': [(32, 0, 'ORG')]}), ('Linux AG', {'entities': [(8, 0, 'ORG')]}), ('NOVELL', {'entities': [(6, 0, 'ORG')]}), ('Linux Products', {'entities': [(14, 0, 'ORG')]}), ('Canonical Ltd', {'entities': [(13, 0, 'ORG')]}), ('ARM Limited', {'entities': [(11, 0, 'ORG')]}), ('STMicroelectronics', {'entities': [(18, 0, 'ORG')]}), ('Adobe Systems', {'entities': [(13, 0, 'ORG')]}), ('Go', {'entities': [(2, 0, 'ORG')]}), ('Red Hat Inc', {'entities': [(11, 0, 'ORG')]}), ('Zen InternetNash-Finch', {'entities': [(22, 0, 'ORG')]}), ('KeyCorp', {'entities': [(7, 0, 'ORG')]}), ('Molina Healthcare', {'entities': [(17, 0, 'ORG')]})]

model = "en_core_web_sm"


def train_spacy(data,iterations):

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    TRAIN_DATA = data

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    if model is None:
        optimizer = nlp.begin_training()
    else:
        print ("resuming")
        optimizer = nlp.resume_training()
        print (optimizer.learn_rate)
    
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    sizes = compounding(1.0, 4.0, 1.001)

    with nlp.disable_pipes(*other_pipes):  # only train NER
        
        for itn in range(iterations):            
            start = time.time() # Iteration Time
            
            if(itn%100 == 0):
                print('Testing')
                modelfile = "training_model"+str(itn)
                nlp.to_disk(modelfile)
  
    
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}

            batches = minibatch(TRAIN_DATA, size=sizes)
            for batch in batches:
                for texts, annotations in batch:
                    doc = nlp.make_doc(texts)
                    examples = Example.from_dict(doc, annotations)
                    nlp.update([examples], drop=0.2, losses=losses)


            print("Losses",losses)
    return nlp

training = train_spacy(TRAIN_DATA, iterations=200)
modelName = "en_core_web_mine"
training.to_disk(modelName)