#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from vectoria_lib.applications.qa import QAApplication

def inference(
    **kwargs: dict
):
    qa_app = QAApplication(
        index_path=kwargs["index_path"]
    )
    qa_app.inference(kwargs)
