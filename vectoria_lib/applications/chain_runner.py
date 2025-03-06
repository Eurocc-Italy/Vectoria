from langchain_core.runnables import Runnable
from langfuse.callback import CallbackHandler

class ChainRunner:

    def __init__(self, chain: Runnable, langfuse_config: dict = None):
        self.chain = chain
        self.callbacks = []
        if langfuse_config.get("enabled"):
            self.callbacks.append(
                CallbackHandler(
                    public_key=langfuse_config.get("public_key"),
                    secret_key=langfuse_config.get("secret_key"),
                    host=langfuse_config.get("host")
                )
            )
        
    def invoke(self, inputs: dict):

        return self.chain.invoke(inputs, config={"callbacks": self.callbacks})

