from x import *

def start_debate(num_speaker:int , stance_model):
    debate = Debate(num_speaker = num_speaker, stance_model = stance_model)
    return debate
    
def add_speaker_statement(debate: Debate, speaker: int , statement: str):
    debate.add_claim(speaker, statement)

def addalot_of_statements(debate: Debate, speaker_statement_list: list[tuple[int,str]]):
    for speaker, statement in speaker_statement_list:
        debate.add_claim(speaker, statement)

def get_debate_output(debate: Debate):
    debate.check_fallacy()
    return debate.contradictions, debate.circular_reasoning

 
if __name__ == "__main__":
    model = pipeline("text-classification", model="roberta-large-mnli")
    debate = start_debate(num_speaker = 2, stance_model = model)
    contradiction_nodes = [
        (0,"All industrial pollution should be banned to protect nature."),
        (1,"All industrial pollution should be banned to protect nature."),
        (1,"Factory A’s pollution are desirable because it creates jobs."),  # contradicts first node
        (0,"Factory B’s pollution are harmful and should be stopped."),       # consistent with first node
        (0,"We should prioritize human welfare over environmental rules.")    # introduces tension
    ]
    addalot_of_statements(debate, contradiction_nodes)
    contradictions, circular = get_debate_output(debate)
    # print(contradictions)
    
    # print(circular)

    

