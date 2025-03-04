from vectoria_lib.applications.application_builder import ApplicationBuilder

def test_application_builder(config, index_test_folder):
    app = ApplicationBuilder.build_qa(
        index_path=index_test_folder
    )
    assert app is not None
    assert app.chain is not None
    assert app.chain.first is not None
    assert len(app.chain.middle) == 1
    assert app.chain.last is not None

def test_application_builder_no_retriever(config, index_test_folder):
    config.set("retriever", "enabled", False)
    app = ApplicationBuilder.build_qa(
        index_path=index_test_folder
    )
    assert app is not None
    assert app.chain is not None
    assert app.chain.first is not None
    assert len(app.chain.middle) == 0
    assert app.chain.last is not None


def test_application_builder_with_reranker(config, index_test_folder):
    config.set("reranker", "enabled", True)
    app = ApplicationBuilder.build_qa(
        index_path=index_test_folder
    )
    assert app is not None
    assert app.chain is not None
    assert app.chain.first is not None
    assert len(app.chain.middle) == 2
    assert app.chain.last is not None

def test_application_builder_with_reranker_with_full_paragraphs_retriever(config, index_test_folder):
    config.set("reranker", "enabled", True)
    config.set("full_paragraphs_retriever", "enabled", True)
    app = ApplicationBuilder.build_qa(
        index_path=index_test_folder
    )
    assert app is not None
    assert app.chain is not None
    assert app.chain.first is not None
    assert len(app.chain.middle) == 3
    assert app.chain.last is not None
