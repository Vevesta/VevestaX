import pandas
import inspect
import os.path
import os
import ipynbname
#import ipyparams
import ipykernel
from datetime import datetime

def test():
    return 'Test Executed Succesfully'

class V(object):
    def __init__(self ):
        self._dataSourcing = None
        self._featureEngineering = None
    
        self.startlocals = None
        self.vars = {}
                
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
       
        self.vars ={**self.vars ,**{i:temp.get(i) for i in temp if i not in self.startlocals and i[0]!='_' and type(temp[i]) in [int, float, bool, str]}}
        
        return self.vars
    
	#create alias of method modellingStart and modellingEnd
    start = startModelling
    end = endModelling
    #-------------

        
    def dump(self, techniqueUsed, filename = None, message = None, version = None):
        
        existingData = None
        modelingData = None
        featureEngineeringData = None
        messageData = None
        experimentID = 1
        if_sheet_exists = None
        mode = 'w'

        if(filename==None):
            filename = "vevesta.xlsv"
        print("Dumped the experiment in the file" + filename)

        #check if file already exists
        if(os.path.isfile(filename)):            
            #mode = 'a'
            if_sheet_exists = 'replace'
            existingData = pandas.read_excel(filename, sheet_name = 'dataSourcing', index_col=[])
            featureEngineeringData = pandas.read_excel(filename, sheet_name = 'featureEngineering', index_col=[])
            modelingData = pandas.read_excel(filename, sheet_name = 'modelling', index_col=[])
            messageData = pandas.read_excel(filename	, sheet_name = 'messages', index_col=[])
            experimentID = max(modelingData["experimentID"]) + 1
            

        df_dataSourcing = pandas.DataFrame(1, index=[1], columns = self._dataSourcing)
        df_dataSourcing.insert(0, 'experimentID', experimentID)
        df_dataSourcing = pandas.concat([existingData, df_dataSourcing], ignore_index=True).fillna(0)
        
        df_featureEngineering = pandas.DataFrame(1, index=[1], columns = self.featureEngineering)
        df_featureEngineering.insert(0, 'experimentID', experimentID)
        df_featureEngineering = pandas.concat([featureEngineeringData, df_featureEngineering], ignore_index=True).fillna(0)

       
        modeling = pandas.DataFrame(data = {**{'experimentID':experimentID, 'features':','.join(self.dataSourcing) , 'timestamp in UTC':datetime.utcnow().isoformat()} , **{ k:[v] for k,v in self.vars.items()}})
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
 
        with pandas.ExcelWriter(filename, engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer: 
            
            df_dataSourcing.to_excel(writer, sheet_name = 'dataSourcing', index =False) 
            
            df_featureEngineering.to_excel(writer, sheet_name = 'featureEngineering', index = False)
            modeling.to_excel(writer, sheet_name = 'modelling', index = False)
        
            df_messages.to_excel(writer, sheet_name = 'messages', index = False)     
        
        print("Explore our tool at www.vevesta.com")
        
     
             
        
        
        
            
        
