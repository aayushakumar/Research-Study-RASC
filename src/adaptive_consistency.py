class BetaStoppingCriteria:
    def __init__(self, beta=0.95):
        self.beta = beta

class AC:
    def __init__(self, stop_criteria, max_gens=40):
        self.criteria = stop_criteria
        self.max_gens = max_gens
    
    def should_stop(self, answers):
        # Simplified ASC: stop when majority > beta
        from collections import Counter
        if len(answers) < 3:
            return False
        most_common_count = Counter(answers).most_common(1)[0][1]
        return most_common_count / len(answers) > self.criteria.beta