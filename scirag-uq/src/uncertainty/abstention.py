class AbstentionMechanism:
    """
    Determines whether the system should abstain from answering based on the 
    calibrated confidence score and a threshold tau.
    """
    def __init__(self, tau: float = 0.65):
        self.tau = tau
        self.refusal_message = "Insufficient supporting evidence found in the retrieved literature."
        
    def should_abstain(self, calibrated_confidence: float) -> bool:
        return calibrated_confidence < self.tau
        
    def format_answer(self, generated_answer: str, calibrated_confidence: float) -> str:
        """
        Returns the generated answer if confident, otherwise returns the refusal message.
        """
        if self.should_abstain(calibrated_confidence):
            return self.refusal_message
        return generated_answer
