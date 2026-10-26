import numpy as np
import igraph 
from transformers import pipeline
from .datasets import *

class roberta_stance_model():
    def __init__(self, input_mode):
        self.model = pipeline("text-classification", model="roberta-large-mnli")
        self.label_dict = {'ENTAILMENT': 1,'CONTRADICTION': -1, "NEUTRAL": 0}
        self.input_mode = input_mode
    def forward(self, x:list[tuple[str,str]]): # 'list' or 'dict'
        output_label = None
        if self.input_mode == "list":
            input = [i[0] + " " + i[1] for i in x]
            output_probs  = self.model(input, top_k=None)
            output_label =  [self.label_dict[x[0]['label']] for x in output_probs]
        elif self.input_mode == "dict":
            input = [{"text":i[0] , "text_pair": i[1]} for i in x]
            output_probs  = self.model(input, top_k=None)
            output_label =  [self.label_dict[x[0]['label']] for x in output_probs]
        return output_probs, output_label
model = roberta_stance_model(input_mode = "list")


class Discourse():
    def __init__(self, stance_model):
        self.discourse_history = []
        self.stance_graph = igraph.Graph(directed = True)
        self.stance_model = stance_model
        self.no_stance_update_vertex_idx = 0
        self.no_edges = 0

    def add_statements_vertices(self, statements: list[tuple[int, str]]):
        no_statements = len(statements)
        self.discourse_history.extend(statements)
        start_vertex_idx = len(self.stance_graph.vs)
        end_vertex_idx = start_vertex_idx + no_statements
        new_vertex_id = range(start_vertex_idx, end_vertex_idx)
        new_vertex_membership = [user for user, statement in statements] #Define membership for new vertices
        self.stance_graph.add_vertices(no_statements, attributes = {"v_id": new_vertex_id,
                                                                    "membership": new_vertex_membership })
        assert len(self.discourse_history) == len(self.stance_graph.vs)
        
    def update_stance_graph_edges(self):
        start_vertex_idx = self.no_stance_update_vertex_idx
        end_vertex_idx = len(self.discourse_history)
        
        if(start_vertex_idx == 0):
            new_to_old_edges = [(v1,v2) for v1 in range(start_vertex_idx, end_vertex_idx) for v2 in range(start_vertex_idx, end_vertex_idx) if v1!=v2] 
            new_to_old_edge_stances, old_to_new_edge_stances = self.get_stance(new_to_old_edges)
            self.add_edges(new_to_old_edges, new_to_old_edge_stances)
        
        else:
            #edges from new vertex to old vertices #end_vertex_idx used in place of len(self.discourse_history)
            new_to_old_edges = [(new_vertex_idx, old_vertex_idx) for new_vertex_idx in range(start_vertex_idx, end_vertex_idx) for old_vertex_idx in range(0,start_vertex_idx)]
            #edges from old vertex to new vertices 
            old_to_new_edges = [(old_vertex_idx, new_vertex_idx) for new_vertex_idx, old_vertex_idx in new_to_old_edges]
        
            new_to_old_edge_stances, old_to_new_edge_stances = self.get_stance(new_to_old_edges)

            self.add_edges(new_to_old_edges, new_to_old_edge_stances)
            self.add_edges(old_to_new_edges, old_to_new_edge_stances)
        

        self.no_stance_update_vertex_idx = end_vertex_idx
        
    def add_edges(self, edges:list[tuple[int,int]], stances:list[int]):
        new_e_ids = range(self.no_edges, self.no_edges+len(edges))
        self.stance_graph.add_edges(edges, attributes = {"stance": stances, "e_id": new_e_ids})
        self.no_edges+=len(edges)
    
    def get_stance(self, edges: list[tuple[int,int]]):
        #change to the dict output expected by
        statements = [(self.discourse_history[i][1], self.discourse_history[j][1]) for i,j in edges]
        
        reverse_statements = statements[::-1]
        edge_probs, edge_stances = self.stance_model.forward(statements)
        reverse_edge_probs, reverse_edge_stances = self.stance_model.forward(statements)

        # reverse_edge_stances = edge_stances #Use to increase performace
        return edge_stances, reverse_edge_stances
    

    


class FallacyChecker():
    def __init__(self, stance_graph: igraph.Graph):
        self.stance_graph = stance_graph
    
    def get_cycle(self, allowed_vertex_membership: list[int], allowed_stances: list[int]) -> tuple[list[tuple], list[tuple]]:
        membership_subgraph = self.stance_graph.subgraph(self.stance_graph.vs.select(membership_in = allowed_vertex_membership))
        stance_subgraph = membership_subgraph.subgraph_edges(membership_subgraph.es.select(stance_in = allowed_stances), delete_vertices = True)

        basis_cycles_edge_idx = stance_subgraph.minimum_cycle_basis()
        basis_cycle_vertices_id = []
        basis_cycle_edges_id = []
        for edge_cycle in basis_cycles_edge_idx:
            temp_vertices = []
            temp_edge_id = []
            # print(edge_cycle)
            for edge in edge_cycle:
                # print(edge)
                v1_idx, v2_idx = stance_subgraph.get_edgelist()[edge]
                v1_id = stance_subgraph.vs["v_id"][v1_idx]
                v2_id = stance_subgraph.vs["v_id"][v2_idx]
                vertices_id = [v1_id, v2_id]
                edge_id = stance_subgraph.es["e_id"][edge]
                temp_edge_id.append(edge_id)                
                temp_vertices.extend(vertices_id)
                
            basis_cycle_edges_id.append(temp_edge_id)    
            basis_cycle_vertices_id.append(tuple(set(temp_vertices)))   
        # print("EID: ", basis_cycles_edge_idx["e_id"]) 
        return basis_cycle_edges_id, basis_cycle_vertices_id
    
    def get_contradiction(self, allowed_vertex_membership: list[int]):
        allowed_stances = [-1]
        
        membership_subgraph = self.stance_graph.subgraph(self.stance_graph.vs.select(membership_in = allowed_vertex_membership))
        contradiction_stance_subgraph = membership_subgraph.subgraph_edges(membership_subgraph.es.select(stance_in = allowed_stances), delete_vertices = True)
        
        contradiction_vertices_id =  []
        for edge in contradiction_stance_subgraph.get_edgelist():
            v1_idx, v2_idx = edge
            v1_id = contradiction_stance_subgraph.vs["v_id"][v1_idx]
            v2_id = contradiction_stance_subgraph.vs["v_id"][v2_idx]
            contradiction_vertices_id.append((v1_id, v2_id))
            
        return contradiction_vertices_id
    

        
if __name__ == "__main__":
    Ethix_dataset = Ethix_data()
    
    debate_data = zip((Ethix_dataset.iloc(12)["Scheme"], Ethix_dataset.iloc(12)["Debate"]))
    print(debate_data)
    # d = Discourse(model)

    # d.add_statements_vertices(t)
    # d.update_stance_graph_edges()
    # fallacy_checker = FallacyChecker(d.stance_graph)
    
    # print(d.stance_graph.get_edgelist())
    # print(d.stance_graph.es["stance"])
    # print()
    # print(fallacy_checker.get_contradiction([0]))
    # print(fallacy_checker.get_cycle([0],[1]))
