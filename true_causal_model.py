import numpy as np
from model import BaseModel

class TrueCausalModel:
	"""
	El objeto de TrueCausalModel contiene el modelo causal de la naturaleza.

	"""
	def __init__(self, model):
		self.model = model

	def action_simulator(self, chosen_actions, values_chosen_actions):
		"""
		Ejecuta las acciones en el modelo causal verdadero (Nature).

		Args:
			chosen_actions (list): lista de las variables de accción a las que se les asignan
			valores.
			values_chosen_actions (list): lista de los valores que se asignan a las variables de acción.
		"""
		response = dict()
		elements = None
		probabilities = None
		for nat_var in self.model.get_nature_variables():
			probabilities = self.model.get_nature_var_prob(nat_var)
			elements = [i for i in range(len(probabilities))]
			res = np.random.choice(elements, p=probabilities)
			response[nat_var] = res
		for idx, variable in enumerate(chosen_actions):
			response[variable] = values_chosen_actions[idx]
		ordered_variables = self.model.get_graph_toposort()
		ordered_variables = [i for i in ordered_variables\
							if i not in response]
		for unknown_variable in ordered_variables:
			if not response.get(unknown_variable):
				response[unknown_variable] = self.model.make_inference(unknown_variable,\
					response)
		return response

def main():
	model = BaseModel('model_parameters.json')
	tcm = TrueCausalModel(model)
	r = tcm.action_simulator(['Tratamiento'], [1])
	print(r)
	r = tcm.action_simulator(['Tratamiento'], [0])
	print(r)
if __name__ == '__main__':
	main()