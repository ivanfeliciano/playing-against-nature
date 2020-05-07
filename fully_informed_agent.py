class fully_informed_agent:
    def __init__(self,model_simulator):
        self.beliefs=dict()
        self.beliefs['Enfermedad']=[0.7, #0
                                    0.3] #1
        
        self.beliefs['Tratamiento']=[0.5, #0
                                     0.5] #1
        
        self.beliefs['Reaccion|Tratamiento']=[[.7,.4], #0 
                                              [.3,.6]] #1
        
        self.beliefs['Final|Enfermedad, Tratamiento, Reaccion']=[[.6,0,.8,0,.4,0,.9,0], #0
                                                                 [.4,1,.2,1,.6,1,.1,1]] #1
        self.recompensa=[0]
        self.simulator=model_simulator.action_simulator
        
    def do_calculus(self,final,treatment): 
        #La distribución P(Final|do(Tratamiento))=P(Final|Enfermedad, Tratamiento, Reaccion)P(Reaccion|Tratamiento)P(Enfermedad)
        #Treatment = 1 ó 0
        #Quiero que me de la probabilidad de vivir=0 ó =1 dado treatment=0 ó =1
        if (treatment==0):
            res=self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][0]*self.beliefs['Reaccion|Tratamiento'][0][0]*self.beliefs['Enfermedad'][0] \
            +self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][1]*self.beliefs['Reaccion|Tratamiento'][1][0]*self.beliefs['Enfermedad'][0] \
            +self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][4]*self.beliefs['Reaccion|Tratamiento'][0][0]*self.beliefs['Enfermedad'][1] \
            +self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][5]*self.beliefs['Reaccion|Tratamiento'][1][0]*self.beliefs['Enfermedad'][1]
        else:
            res=self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][2]*self.beliefs['Reaccion|Tratamiento'][0][1]*self.beliefs['Enfermedad'][0] \
                +self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][3]*self.beliefs['Reaccion|Tratamiento'][1][1]*self.beliefs['Enfermedad'][0] \
                +self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][6]*self.beliefs['Reaccion|Tratamiento'][0][1]*self.beliefs['Enfermedad'][1] \
                +self.beliefs['Final|Enfermedad, Tratamiento, Reaccion'][final][7]*self.beliefs['Reaccion|Tratamiento'][1][1]*self.beliefs['Enfermedad'][1]
        return(res)
    
    def make_decision(self,final=1):
        f_0= self.do_calculus(final,0)
        f_1=self.do_calculus(final,1)
        if (f_0 > f_1):
            return(0)
        else:
            return(1)
        
    def observing_external(self):
        chosen_action=self.make_decision()
        datos=self.simulator(chosen_action)
        self.recompensa.append(datos['Final'])
        return(datos)

    def get_reward(self):
        return(self.recompensa)

def main():
    informed_agent = fully_informed_agent()
