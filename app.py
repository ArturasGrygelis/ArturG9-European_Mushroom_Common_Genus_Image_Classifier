
__all__ = ['learner', 'labels', 'interface', 'classify']

from fastai.vision.all import *
import gradio as gr

learner = load_learner('ConvNext_RmsProps.pkl')

labels = learner.dls.vocab


def classify(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learner.predict(img)
    return {labels[i]: float(probs[i]) for i in range (len(labels))}


interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(),
    outputs=gr.Label(),
    examples=['Amanita.jpg','(Lactarius_aurantiacus).jpg','Agaricus_silvaticus.jpg','Agaricus_Silvicola.jpg','Amanita_phalloides(DeathCap).jpg','Amanita_polypyramis.jpg','Amanita2.jpg','Boletus_Betulicola.jpg','Boletus_edulis.jpg','Boletus_Luridus.jpg','Cortinarius_caperatus.jpg','cortinarius-armillatus.jpg','cortinarius-praestans.jpg','Eglyninė_rudmėsė_Lactarius_deterrimus.jpg','entoloma_caeruleum.jpg','Entoloma_griseocyaneum.jpg','Geltonasis_kazlėkas.jpg','Hygrocybe_nigrescens.jpg','Hygrocybe_psittacina(WaxCap).jpg','hygrocype_conica.jpg','Lactarius_acerrimus.jpg','Lactarius-blennius.jpg','Russula_Aeruginea.jpg','Russula_alutacea.jpg','Suillus_collinitus.jpg','Suillus_granulatus.jpg','Suillus_variegatus.jpg'],
    allow_flagging='never',
    title='European Mushroom Common Genus Image Classifier',
    description='Model trained to classify nine types of european mushroom common genus image classifier — Entoloma, Suillus, Hygrocybe, Agaricus, Amanita, Lactarius, Russula,Boletus,Cortinarius.'
)
interface.launch(share=False, inline=False)
