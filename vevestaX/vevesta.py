import pandas
import inspect
import os.path
import ipynbname
from datetime import datetime

def test():
    return 'Test Executed Succesfully'

class Experiment(object):
    def __init__(self ):
        self._dataSourcing = None
        self._featureEngineering = None
    
        self.startlocals = None
        self.variables = {}
                
    def get_filename(self):
        return ipynbname.name()

    @property
    def dataSourcing(self):
        return self._dataSourcing

    @dataSourcing.setter
    def dataSourcing(self, value):
        if type(value) == pandas.core.frame.DataFrame:
            self._dataSourcing = value.columns


    @property
    def ds(self):
        return self._dataSourcing

    @dataSourcing.setter
    def ds(self, value):
        self.dataSourcing = value

        
    @property
    def featureEngineering(self):
        return self._featureEngineering

    @featureEngineering.setter
    def featureEngineering(self, value):
        if self._dataSourcing is None:
            print("Data Sourcing step missed.")
            if type(value) == pandas.core.frame.DataFrame:
                cols = value.columns
                self._featureEngineering = cols

        else:
            if type(value) == pandas.core.frame.DataFrame:
                cols = value.columns
                cols = cols.drop(self.dataSourcing)
                self._featureEngineering = cols


    @property
    def fe(self):
        return self._featureEngineering

    @featureEngineering.setter
    def fe(self, value):
        self.featureEngineering = value
        
    def startModelling(self):
        self.startlocals = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()

    
    def endModelling(self):

        temp = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()
        self.temp = inspect.getmembers(inspect.stack()[1])
       
        self.variables ={**self.variables ,**{i:temp.get(i) for i in temp if i not in self.startlocals and i[0]!='_' and type(temp[i]) in [int, float, bool, str]}}
        
        return self.variables
    
    #create alias of method modellingStart and modellingEnd
    start = startModelling
    end = endModelling
    #Exp = Experiment
    #-------------

        
    def dump(self, techniqueUsed, filename = None, message = None, version = None):
        
        existingData = None
        modelingData = None
        featureEngineeringData = None
        messageData = None
        experimentID = 1
        mode = 'w'

        if(filename==None):
            filename = "vevesta.xlsx"
        print("Dumped the experiment in the file " + filename)

        #check if file already exists
        if(os.path.isfile(filename)):            
            #mode = 'a'
            existingData = pandas.read_excel(filename, sheet_name = 'dataSourcing', index_col=[])
            featureEngineeringData = pandas.read_excel(filename, sheet_name = 'featureEngineering', index_col=[])
            modelingData = pandas.read_excel(filename, sheet_name = 'modelling', index_col=[])
            messageData = pandas.read_excel(filename    , sheet_name = 'messages', index_col=[])
            experimentID = max(modelingData["experimentID"]) + 1
            
        if self._dataSourcing is None:
            df_dataSourcing = pandas.DataFrame(index=[1])
        else:
            df_dataSourcing = pandas.DataFrame(1, index=[1], columns = self._dataSourcing)

        df_dataSourcing.insert(0, 'experimentID', experimentID)
        df_dataSourcing = pandas.concat([existingData, df_dataSourcing], ignore_index=True).fillna(0)
        
        if self.featureEngineering is None:
            df_featureEngineering = pandas.DataFrame(index=[1])     
        else:
            df_featureEngineering = pandas.DataFrame(1, index=[1], columns = self.featureEngineering)
            
        df_featureEngineering.insert(0, 'experimentID', experimentID)
        df_featureEngineering = pandas.concat([featureEngineeringData, df_featureEngineering], ignore_index=True).fillna(0)

        
        if self._dataSourcing is None and self._featureEngineering is None:
            modeling = pandas.DataFrame(data = {**{'experimentID':experimentID, 'timestamp in UTC':datetime.utcnow().isoformat()} , **{ k:[v] for k,v in self.variables.items()}},index=[0])
        elif self._dataSourcing is None:
            modeling = pandas.DataFrame(data = {**{'experimentID':experimentID, 'features':','.join(self.featureEngineering) , 'timestamp in UTC':datetime.utcnow().isoformat()} , **{ k:[v] for k,v in self.variables.items()}},index=[0])
        elif self._featureEngineering is None:
            modeling = pandas.DataFrame(data = {**{'experimentID':experimentID, 'features':','.join(self.dataSourcing) , 'timestamp in UTC':datetime.utcnow().isoformat()} , **{ k:[v] for k,v in self.variables.items()}},index=[0])
        else:
            modeling = pandas.DataFrame(data = {**{'experimentID':experimentID, 'features':','.join(self.dataSourcing)+','+','.join(self.featureEngineering) , 'timestamp in UTC':datetime.utcnow().isoformat()} , **{ k:[v] for k,v in self.variables.items()}},index=[0])
        
        modeling = pandas.concat([modelingData, modeling], ignore_index=True)
                
        #message table       
        data = {
        'experimentID': experimentID,
        'techniqueUsed': techniqueUsed,
        'message': message,
        'version': version,
        'filename' : self.get_filename() + '.ipynb',
        'timestamp in UTC' : datetime.utcnow().isoformat()
        }
        
          
        df_messages = pandas.DataFrame(index =[1], data = data)
        df_messages = pandas.concat([messageData, df_messages], ignore_index=True)
 
        with pandas.ExcelWriter(filename, engine='openpyxl') as writer: 
            
            df_dataSourcing.to_excel(writer, sheet_name = 'dataSourcing', index =False) 
            
            df_featureEngineering.to_excel(writer, sheet_name = 'featureEngineering', index = False)
            modeling.to_excel(writer, sheet_name = 'modelling', index = False)
        
            df_messages.to_excel(writer, sheet_name = 'messages', index = False)     
        
        print("For additional features, explore our tool at www.vevesta.com for free.")
