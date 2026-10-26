import numpy as np
from transformers import pipeline

# Thougts on improvemenit and expansion of scope
""" Implement Conditional statements to ensure all parties give arguments that follow the certain predefined conditions.(eg: statements must not contradict the geneva conventions) """
""" Implement grounding nodes as evidence for certain claims """
""" Implement verifiability for uncertain claims: Upload report or pdf that can be used for verifying claims (also can use claimify) """
""" Diplomatic use case: Ensure statement is consistent with old statements(done) and does not contradicts ally's stance """
""" Acceptability of a node (Dung Smantics). A node N is said to be acceptable wrt set S if all attackers of N have attackers in set S -> 
This can be used to identify the claims that are unsupported and to be focused on during effective argumentation"""


LABEL_DICT = {'ENTAILMENT': 1,'CONTRADICTION': -1, "NEUTRAL": 0}

def find_cycles_adj_matrix(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    stack = [False] * n
    cycles = []
    
    def dfs(v, path):
        visited[v] = True
        stack[v] = True
        path.append(v)

        for u in range(n):
            if adj_matrix[v][u]:  # edge exists
                if not visited[u]:
                    dfs(u, path)
                elif stack[u]:
                    cycle_start = path.index(u)
                    cycles.append(path[cycle_start:].copy())

        stack[v] = False
        path.pop()

    for node in range(n):
        if not visited[node]:
            dfs(node, [])

    return cycles

class Fallacy_Checker():
    
    def __init__(self, A_t, C_t):
        self.A_t = A_t
        self.C_t = C_t
        self.speaker_A_t = None
        
        assert self.C_t != [] or self.A_t != []
            
        self.C_t =  np.array(self.C_t) #ARGS x NUM_SPEAKER
        speaker_arg_mask = np.transpose(self.C_t , (1,0)) #NUM_SPEAKER x ARGS
        self.A_t = np.array(self.A_t) #ARGS X ARGS
        self.speaker_A_t =  np.einsum('ij,jk->ijk', speaker_arg_mask, self.A_t) #NUM_SPEAKER x ARGS
        self.num_speaker = len(self.speaker_A_t)
        
        
    def contradiction(self):
        contradictions = {}
        for speaker in range(self.num_speaker):
            for i in range(len(self.speaker_A_t[speaker])):
                for j in range(i, len(self.speaker_A_t[speaker])):
                    if((self.speaker_A_t[speaker][i][j], self.speaker_A_t[speaker][j][i]) in ((1,-1), (-1,-1), (-1,1))):
                        if(speaker in contradictions):
                            contradictions[speaker].append([i,j])
                        else:
                            contradictions[speaker] = [[i,j]]
                    
        if(len(contradictions)>0):
            return contradictions
        return None
    
    def circular_reasoning(self):
        cycles = {}
        
        for speaker in range(self.num_speaker):
            x = self.speaker_A_t[speaker]

            x = np.where(x == 1, x, 0)
            speaker_cycles = find_cycles_adj_matrix(x)
            speaker_cycles = [sublist for sublist in speaker_cycles if len(sublist)>1]
            cycles[speaker] = speaker_cycles
        if(len(cycles)>0):
            return cycles
        return None
        

class Debate():
    def __init__(self, num_speaker, stance_model):
        self.claims = []
        self.num_speaker = num_speaker
        self.n = len(self.claims)
        
        
        self.A_t = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.C_t = [[0 for _ in range(self.num_speaker)] for _ in range(self.n)]
        
        self.stance_model = stance_model
        
        self.circular_reasoning = None
        self.contradictions = None

        
    def add_claim(self, speaker:int, claim:str):
        self.A_t.append([0 for _ in range(self.n)])
        self.n += 1
        self.claims.append(claim)
        self.C_t.append([0 for _ in range(self.num_speaker)])
        self.update_A_t()
        self.C_t[-1][speaker] = 1
    
    def update_A_t(self):
        claim_combinations = []
        for i in self.claims:
            for j in self.claims:
                claim_combinations.append(f"{i} [SEP] {j}.")  
        
        pred = self.stance_model(claim_combinations)

        for idx, p in enumerate(pred):
            
            row = idx // self.n
            col = idx % self.n
            if col == self.n-1:
                self.A_t[row].append(LABEL_DICT[p["label"]])
            else:
                self.A_t[row][col] = LABEL_DICT[p["label"]]
    
    def check_fallacy(self):
        self.fallacy_checker = Fallacy_Checker(self.A_t, self.C_t)
        self.circular_reasoning = self.fallacy_checker.circular_reasoning()
        self.contradictions = self.fallacy_checker.contradiction()
        
        if(self.circular_reasoning):
            print("|----------The arguments have circular reasoning ----------|")
            print(self.circular_reasoning)
        
        if(self.contradictions):
            print("|----------The arguments have contradictions----------|")
            print(self.contradictions)
    
    def show_fallacy(self):
        if(self.circular_reasoning):
            for speaker in self.circular_reasoning:
                print(f"|---------Circular Reasoning Fallacy of Speaker: {speaker}--------|")
                for fallacies in self.circular_reasoning[speaker]:
                    s = ""
                    if(len(fallacies)>1):
                        s = " <-- "
                        for claim_idx in fallacies:
                            s += self.claims[claim_idx]
                            s += " --> "
                    if(s != ""):
                        print(s)
        
        if(self.contradictions):
            for speaker in self.contradictions:
                print(f"|---------Contradiction Fallacy of Speaker: {speaker}--------------|")
                for fallacies in self.contradictions[speaker]:
                    print(f" {self.claims[fallacies[0]]} != {self.claims[fallacies[1]]} ")
                    
                    

                    
                    
if __name__ == "__main__":
    STANCE_MODEL = pipeline("text-classification", model="roberta-large-mnli")
    circular_reasoning_nodes = [
        [0, "This law is fair because it promotes justice."],
        [0,"It promotes justice because the law ensures fairness."],
        [0,"Ensuring fairness is the law's purpose because lawmakers intended it."],
        [0,"Lawmakers intended the law to promote justice, which is fair."]  # loops back to first node
    ]
    contradiction_nodes = [
        [0,"All industrial pollution should be banned to protect nature."],
        [1,"All industrial pollution should be banned to protect nature."],
        [1,"Factory A’s pollution are desirable because it creates jobs."],  # contradicts first node
        [0,"Factory B’s pollution are harmful and should be stopped."],       # consistent with first node
        [0,"We should prioritize human welfare over environmental rules."]    # introduces tension
    ]
    debate = Debate(num_speaker = 2, stance_model = STANCE_MODEL)
    template = contradiction_nodes
    for round in range(len(template)):
        # claim = str(input(f"Enter claim for speaker {speaker}: "))
        speaker, claim = template[round]
        debate.add_claim(speaker, claim)
        
    # #print adjacency matrix that shows relationship between all claims
    # for i in debate.A_t:
    #     print(i)
    # print()
    # # Check claims and relationship between claims 
    # for i,j in zip(debate.A_t, debate.claims):
    #     print(j,i)
    
    
    debate.show_fallacy()
    debate.check_fallacy()
    
    print("---------")
    print(debate.circular_reasoning)
    print("---------")
    print(debate.contradictions)