#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from vectoria_lib.applications.application_builder import ApplicationBuilder

def inference(
    **kwargs: dict
):
    qa_app = ApplicationBuilder.build_qa(
        index_path=kwargs["index_path"]
    )
    qa_app.inference(kwargs)
