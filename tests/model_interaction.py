from sys import argv
import logging

import numpy as np
import pandas as pd

from model import BaseModel
from true_causal_model import TrueCausalModel
from structure_learning import generate_approx_model_from_graph

def run_simulation(config_filename, graphfilename, n):
  model = BaseModel(config_filename)
  tcm = TrueCausalModel(model)
  tcm.model.save_pgm_as_img(graphfilename)
  intervention_vars = model.get_intervention_variables()
  target_value = 1
  target = {
      "variable": model.get_target_variable(),
      "value": target_value
  }
  local_data = dict()
  for i in range(n):
    idx_intervention_var = np.random.randint(len(intervention_vars))
    action = (intervention_vars[idx_intervention_var], np.random.randint(2))
    nature_response = tcm.action_simulator([action[0]], [action[1]])
    for k in nature_response:
      if not k in local_data:
        local_data[k] = []
      local_data[k].append(nature_response[k])

  df = pd.DataFrame.from_dict(local_data)
  model_from_data = generate_approx_model_from_graph(ebunch=model.get_ebunch(), nodes=model.nodes, df=df)
  for cpd in model_from_data.get_cpds():
    logging.info(cpd)

if __name__ == "__main__":
  n = int(argv[1]) if len(argv) > 1 else 1000
  logging.basicConfig(filename="logs/test_model_interaction.log",
                    filemode='w', level=logging.INFO)
  config_filenames = [
    "configs/model_parameters.json",
    "configs/model_parameters_reaction.json",
    "configs/model_parameters_lives.json"
  ]
  graph_names = [
    "gt-graph.png",
    "gt-graph-reaction.png",
    "gt-graph-lives.png"
  ]
  for config, name in zip(config_filenames, graph_names):
    logging.info(f"Testing {name}")
    run_simulation(config, name, n)